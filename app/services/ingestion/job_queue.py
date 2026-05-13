"""
Single-worker async ingestion queue.

FastAPI BackgroundTasks runs jobs in the request handler's task group,
which means N concurrent uploads = N concurrent bge-m3 jobs in memory.
On a normal box that OOMs. This queue serializes jobs through a single
worker so memory pressure is bounded regardless of upload concurrency.

The status dict doubles as the source of truth for GET /ingest/status/{doc_id}.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# JobFn signature: async def fn(doc_id: str, payload: dict, status_cb: Callable[[dict], None]) -> None
JobFn = Callable[[str, dict, Callable[[dict], None]], Awaitable[None]]


@dataclass
class Job:
    doc_id: str
    payload: dict
    fn: JobFn
    submitted_at: float = field(default_factory=time.time)


class IngestionJobQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._statuses: dict[str, dict[str, Any]] = {}
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())
            logger.info("Ingestion worker started")

    async def stop(self) -> None:
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def submit(self, doc_id: str, payload: dict, fn: JobFn) -> None:
        self._statuses[doc_id] = {
            "state": "queued",
            "submitted_at": time.time(),
            "progress": None,
            "message": None,
            "error": None,
        }
        await self._queue.put(Job(doc_id=doc_id, payload=payload, fn=fn))

    def get_status(self, doc_id: str) -> dict | None:
        return self._statuses.get(doc_id)

    def all_statuses(self) -> dict[str, dict]:
        return dict(self._statuses)

    def _update_status(self, doc_id: str, **fields) -> None:
        if doc_id not in self._statuses:
            self._statuses[doc_id] = {}
        self._statuses[doc_id].update(fields)

    async def _worker(self) -> None:
        while True:
            try:
                job = await self._queue.get()
            except asyncio.CancelledError:
                logger.info("Ingestion worker cancelled")
                raise

            doc_id = job.doc_id
            self._update_status(
                doc_id,
                state="processing",
                started_at=time.time(),
            )

            def status_cb(fields: dict) -> None:
                self._update_status(doc_id, **fields)

            try:
                await job.fn(doc_id, job.payload, status_cb)
                self._update_status(
                    doc_id,
                    state="done",
                    finished_at=time.time(),
                )
            except Exception as e:
                logger.error(f"[{doc_id}] Ingestion job failed: {e}", exc_info=True)
                self._update_status(
                    doc_id,
                    state="failed",
                    error=str(e),
                    finished_at=time.time(),
                )
            finally:
                self._queue.task_done()
