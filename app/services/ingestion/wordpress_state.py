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
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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
        return self._data.get(content_key)

    def set(self, content_key: str, iso_ts: str) -> None:
        existing = self._data.get(content_key)
        # Only advance the watermark forward; defensive against out-of-order updates.
        if existing and existing >= iso_ts:
            return
        self._data[content_key] = iso_ts
        self._save()

    def all(self) -> dict[str, str]:
        return dict(self._data)

    def reset(self, content_key: Optional[str] = None) -> None:
        if content_key is None:
            self._data = {}
        else:
            self._data.pop(content_key, None)
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
