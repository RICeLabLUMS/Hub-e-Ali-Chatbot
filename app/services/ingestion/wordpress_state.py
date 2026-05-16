"""
Per-content-type incremental sync state for the WordPress importer.

State file is a small JSON keyed by content type:

    {
        "posts": "2026-05-15T10:00:00",
        "pages": "2026-05-15T09:00:00",
        "media": "2026-05-15T08:00:00",
        "cpt:lectures": "2026-05-15T07:00:00"
    }

Values are ISO-8601 GMT timestamps (no trailing 'Z') that match the format
the WordPress REST API returns in `modified_gmt` and accepts on `modified_after`.

Writes are atomic (tmp file + rename) so a crash mid-write cannot corrupt
the file.
"""

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# In-process serialization (one Python process, multiple threads e.g. APScheduler).
# Cross-process coordination (CLI + scheduler at the same time) is the next
# level up - see _acquire_cross_process_lock for the OS-level lock file.
_thread_lock = threading.Lock()


class WordPressSyncState:
    def __init__(self, path: str):
        self.path = Path(path)
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                self._data = json.load(f) or {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read state file {self.path}: {e}. Starting fresh.")
            self._data = {}

    def get(self, content_key: str) -> Optional[str]:
        with _thread_lock:
            return self._data.get(content_key)

    def set(self, content_key: str, iso_ts: str) -> None:
        with _thread_lock:
            existing = self._data.get(content_key)
            # Only advance the watermark forward; defensive against out-of-order updates.
            if existing and existing >= iso_ts:
                return
            # Re-read from disk to pick up any concurrent updates from another
            # process (CLI vs scheduler) before we merge our change in.
            self._merge_from_disk_locked()
            existing = self._data.get(content_key)
            if existing and existing >= iso_ts:
                return
            self._data[content_key] = iso_ts
            self._save_locked()

    def all(self) -> dict[str, str]:
        with _thread_lock:
            return dict(self._data)

    def reset(self, content_key: Optional[str] = None) -> None:
        with _thread_lock:
            if content_key is None:
                self._data = {}
            else:
                self._data.pop(content_key, None)
            self._save_locked()

    def get_failed_pdfs(self) -> set[str]:
        """Return URLs that previously failed linked-PDF download/ingest.
        Stored under the reserved '_failed_linked_pdfs' key. Used by the sync
        orchestrator to retry on next run (regardless of watermark)."""
        with _thread_lock:
            raw = self._data.get("_failed_linked_pdfs")
            if isinstance(raw, list):
                return set(raw)
            return set()

    def set_failed_pdfs(self, urls: set[str]) -> None:
        """Replace the persisted failed-PDF list."""
        with _thread_lock:
            self._merge_from_disk_locked()
            if urls:
                self._data["_failed_linked_pdfs"] = sorted(urls)
            else:
                self._data.pop("_failed_linked_pdfs", None)
            self._save_locked()

    # ----- internal -----

    def _merge_from_disk_locked(self) -> None:
        """Re-read disk state so we don't trample updates from another process
        (CLI vs APScheduler). Watermark merge is max() so it's safe with the
        monotonic 'only-advance' property. Caller must hold _thread_lock."""
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                disk = json.load(f) or {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Re-read of {self.path} failed: {e}")
            return
        for k, v in disk.items():
            cur = self._data.get(k)
            # Watermark values (ISO strings) merge by max(); list values
            # (failed PDFs) merge by union; the rest take disk value.
            if isinstance(v, str) and isinstance(cur, str):
                if v > cur:
                    self._data[k] = v
            elif isinstance(v, list) and isinstance(cur, list):
                self._data[k] = sorted(set(cur) | set(v))
            elif k not in self._data:
                self._data[k] = v

    def _save_locked(self) -> None:
        """Atomic write (tmp+rename) plus a co-located lock file to serialize
        across processes. Caller must hold _thread_lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = Path(str(self.path) + ".lock")
        # Spin briefly for cross-process lock. The lock is held only for the
        # tmp-write + atomic rename - microseconds in the normal case.
        acquired = self._acquire_cross_process_lock(lock_path)
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=self.path.name + ".",
                suffix=".tmp",
                dir=str(self.path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2, sort_keys=True)
                os.replace(tmp_path, self.path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        finally:
            if acquired:
                try:
                    os.unlink(lock_path)
                except OSError:
                    pass

    @staticmethod
    def _acquire_cross_process_lock(lock_path: Path, max_wait_s: float = 5.0) -> bool:
        """Best-effort exclusive lock via O_CREAT|O_EXCL on a sibling lock file.
        Returns True if we own the lock (must release), False if we gave up
        waiting (write proceeds anyway - state file is monotonic, so worst
        case is two writes interleave to the same final state)."""
        import time
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            try:
                fd = os.open(
                    str(lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
                os.close(fd)
                return True
            except FileExistsError:
                # Stale-lock guard: if the lock is older than the deadline window,
                # forcibly remove. Prevents deadlock from a crashed prior writer.
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > max_wait_s:
                        os.unlink(lock_path)
                        continue
                except OSError:
                    pass
                time.sleep(0.05)
        logger.warning(f"Could not acquire {lock_path} within {max_wait_s}s; proceeding")
        return False

    # Backwards-compatible facade for existing callers that may still invoke
    # the old method names (none in-tree, but external scripts might).
    def _save(self) -> None:
        with _thread_lock:
            self._save_locked()
