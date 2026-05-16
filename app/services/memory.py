"""
Per-session conversation history with TTL eviction.

Why this exists: the chat /chat endpoint uses recent turns to rewrite
follow-up questions into standalone search queries. Before this rewrite,
a single global deque mixed every user's turns together - two simultaneous
users would see each other's prior turns appear in their query rewrites.

A session is identified by an opaque ID supplied by the client (typically a
cookie or browser-local UUID). Histories live in process memory and are
auto-evicted after SESSION_TTL_SECONDS of inactivity, so an abandoned
session can't grow forever.

For multi-process / multi-server deployments, swap _store with a Redis-backed
implementation; the public API (get_history / store_memory / reset) is stable.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_TURNS = 5
SESSION_TTL_SECONDS = 60 * 60 * 6   # 6 hours of inactivity -> evict
MAX_SESSIONS = 5_000                # hard cap on simultaneous sessions

# Phrases we recognize as "no useful answer" and refuse to store - storing
# them pollutes the rewrite_query context for the user's next turn.
_DONT_KNOW_MARKERS = (
    "i don't know based on the provided sources",
    "i don't know",
    "i do not know",
)


@dataclass
class _Session:
    history: deque = field(default_factory=lambda: deque(maxlen=MAX_TURNS))
    last_access: float = field(default_factory=time.time)


_store: dict[str, _Session] = {}
_lock = threading.Lock()

# Sentinel used by the chat route when the client doesn't supply a session_id.
# Treats every anonymous call as its own "session" of one - no cross-talk
# between anonymous users, but no follow-up context either. The chat layer
# generates and returns a session_id for the client to send on the next call.
ANON_PREFIX = "anon-"


def _is_dont_know(answer: str) -> bool:
    if not answer:
        return True
    lowered = answer.strip().lower()
    return any(lowered.startswith(m) for m in _DONT_KNOW_MARKERS)


def _evict_expired(now: float | None = None) -> None:
    """Remove sessions idle longer than SESSION_TTL_SECONDS. Caller holds lock."""
    now = now if now is not None else time.time()
    cutoff = now - SESSION_TTL_SECONDS
    stale = [sid for sid, s in _store.items() if s.last_access < cutoff]
    for sid in stale:
        del _store[sid]


def _evict_oldest_if_full() -> None:
    """If we're at MAX_SESSIONS, drop the LRU. Caller holds lock."""
    if len(_store) < MAX_SESSIONS:
        return
    oldest = min(_store.items(), key=lambda kv: kv[1].last_access)
    del _store[oldest[0]]


def new_session_id() -> str:
    return uuid.uuid4().hex


def store_memory(session_id: str | None, question: str, answer: str) -> None:
    """Append (Q, A) to this session's history. Skips 'I don't know' answers
    so they don't pollute the rewrite_query context next turn."""
    if not session_id:
        return
    if _is_dont_know(answer):
        return

    now = time.time()
    with _lock:
        _evict_expired(now)
        sess = _store.get(session_id)
        if sess is None:
            _evict_oldest_if_full()
            sess = _Session()
            _store[session_id] = sess
        sess.last_access = now
        sess.history.append(f"Q: {question}\nA: {answer}")


def get_history(session_id: str | None) -> list[str]:
    """Return this session's history (most-recent last), or [] if unknown."""
    if not session_id:
        return []
    now = time.time()
    with _lock:
        _evict_expired(now)
        sess = _store.get(session_id)
        if sess is None:
            return []
        sess.last_access = now
        return list(sess.history)


def reset(session_id: str | None) -> bool:
    """Clear this session's history. Returns True if it existed, False otherwise."""
    if not session_id:
        return False
    with _lock:
        return _store.pop(session_id, None) is not None


def stats() -> dict:
    """Snapshot for diagnostics."""
    with _lock:
        return {
            "active_sessions": len(_store),
            "max_sessions": MAX_SESSIONS,
            "session_ttl_seconds": SESSION_TTL_SECONDS,
            "max_turns_per_session": MAX_TURNS,
        }
