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
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# JobFn signature: async def fn(doc_id: str, payload: dict, status_cb: Callable[[dict], None]) -> None
JobFn = Callable[[str, dict, Callable[[dict], None]], Awaitable[None]]

# Hard cap on retained job status records. Older terminal entries (done/failed)
# are evicted first when this is exceeded; in-flight entries are kept.
MAX_STATUSES = 500


@dataclass
class Job:
    doc_id: str
    payload: dict
    fn: JobFn
    submitted_at: float = field(default_factory=time.time)


class IngestionJobQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        # OrderedDict so we can pop the LRU terminal job in _maybe_evict.
        self._statuses: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
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
        self._statuses.move_to_end(doc_id)
        self._maybe_evict()
        await self._queue.put(Job(doc_id=doc_id, payload=payload, fn=fn))

    def get_status(self, doc_id: str) -> dict | None:
        return self._statuses.get(doc_id)

    def all_statuses(self) -> dict[str, dict]:
        return dict(self._statuses)

    def _update_status(self, doc_id: str, **fields) -> None:
        if doc_id not in self._statuses:
            self._statuses[doc_id] = {}
        self._statuses[doc_id].update(fields)
        # Bump to most-recently-touched on every update so terminal jobs we're
        # actively reporting on stay in the dict.
        self._statuses.move_to_end(doc_id)

    def _maybe_evict(self) -> None:
        """Cap the status dict at MAX_STATUSES. Evict oldest terminal (done /
        failed) entries first; never evict in-flight (queued / processing)."""
        if len(self._statuses) <= MAX_STATUSES:
            return
        terminal = ("done", "failed")
        # Walk from least-recently-touched and remove terminal ones until under cap.
        for doc_id in list(self._statuses.keys()):
            if len(self._statuses) <= MAX_STATUSES:
                return
            if self._statuses[doc_id].get("state") in terminal:
                del self._statuses[doc_id]
        # If still over cap, all remaining entries are in-flight; that's fine,
        # they'll evict naturally once they terminate.

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
