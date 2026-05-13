"""In-memory conversation history. Replace with Redis/DB for multi-user later."""

from collections import deque

MAX_TURNS = 5
_history: deque[str] = deque(maxlen=MAX_TURNS)


def store_memory(question: str, answer: str) -> None:
    _history.append(f"Q: {question}\nA: {answer}")


def get_history() -> list[str]:
    return list(_history)


def reset() -> None:
    _history.clear()
