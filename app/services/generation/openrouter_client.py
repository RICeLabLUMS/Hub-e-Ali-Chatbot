import asyncio
import json
import logging
from typing import Any

import httpx
from json_repair import repair_json

from app.core.config import settings

logger = logging.getLogger(__name__)


class _ProviderError(RuntimeError):
    """OpenRouter returned a parseable error envelope (often with HTTP 200).
    Distinguished from transport errors so the retry loop can treat it as a
    transient failure but the final message preserves the provider's error
    text for ops debugging."""


# Retry config for transient LLM failures (5xx, network blips, rate limits).
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.5  # seconds; doubled on each retry

LANG_NAMES = {"en": "English", "ar": "Arabic", "ur": "Urdu"}

ANSWER_SYSTEM_TEMPLATE = (
    "You answer questions about Islamic texts. Reply STRICTLY as JSON matching:\n"
    '{{"answer": "<answer in {lang_name}>", "citations": [{{"chunk_id": "..."}}]}}\n\n'
    "Rules:\n"
    "- Use ONLY the provided context. Never invent facts.\n"
    "- If the context does not answer the question, return:\n"
    '  {{"answer": "I don\'t know based on the provided sources.", "citations": []}}\n'
    "- Cite ONLY chunk_ids that are listed in the context.\n"
    "- Output JSON only. No prose, no markdown fences."
)


class OpenRouterClient:
    """Thin async wrapper over the OpenRouter chat completions endpoint."""

    def __init__(self) -> None:
        self.api_key = settings.OPENROUTER_API_KEY
        self.model = settings.OPENROUTER_MODEL
        self.base_url = settings.OPENROUTER_BASE_URL.rstrip("/")
        self._client = httpx.AsyncClient(timeout=60.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate_answer(
        self,
        query: str,
        reranked_chunks: list[dict],
        language: str,
    ) -> dict[str, Any]:
        """Build the strict-JSON prompt, call the model, parse the result."""
        if not reranked_chunks:
            return {
                "answer": "I don't know based on the provided sources.",
                "citations": [],
            }

        lang_name = LANG_NAMES.get(language, "English")
        system = ANSWER_SYSTEM_TEMPLATE.format(lang_name=lang_name)
        context = self._format_context(reranked_chunks)

        user = (
            f"QUESTION:\n{query}\n\n"
            f"CONTEXT (cite chunk_ids verbatim):\n{context}"
        )

        try:
            raw = await self._chat(
                system=system,
                user=user,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
        except _ProviderError as e:
            # OpenRouter returned an error envelope (often as HTTP 200). The
            # message includes the provider's own error text - log it intact
            # so ops can tell "insufficient credits" from "model overloaded"
            # from "content moderated" etc.
            logger.error(f"OpenRouter answer call rejected by provider: {e}")
            return {
                "answer": (
                    "The answer service is temporarily unavailable. "
                    "Please try again in a moment."
                ),
                "citations": [],
            }
        except Exception as e:
            # Network / transport / unexpected error.
            logger.error(
                f"OpenRouter answer call failed: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return {
                "answer": (
                    "The answer service is temporarily unavailable. "
                    "Please try again in a moment."
                ),
                "citations": [],
            }

        parsed = self._parse_json(raw)
        if not isinstance(parsed, dict) or "answer" not in parsed:
            logger.warning(f"OpenRouter returned malformed JSON: {raw[:200]}...")
            return {
                "answer": "I don't know based on the provided sources.",
                "citations": [],
            }
        parsed.setdefault("citations", [])
        return parsed

    async def rewrite_query(self, question: str, history: list[str]) -> str:
        """Make a follow-up question standalone using prior turns."""
        if not history:
            return question

        context = "\n".join(history[-3:])
        system = (
            "Rewrite the user's question as a standalone search query "
            "that captures necessary context from prior turns. Output ONLY "
            "the rewritten query, no preamble."
        )
        user = f"PRIOR TURNS:\n{context}\n\nQUESTION: {question}"

        try:
            rewritten = await self._chat(system=system, user=user, temperature=0.0)
            rewritten = rewritten.strip().strip('"').strip("'")
            return rewritten or question
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}; using original")
            return question

    async def _chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        response_format: dict | None = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_format is not None:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://hubeali.local",
            "X-Title": "HubeAli RAG",
        }

        # Retry on transient failures: network errors, 429 rate limit, 5xx,
        # and the OpenRouter "200 with error body" quirk where the provider
        # routes the call to a backend that returns an error JSON inline.
        # 4xx other than 429 are non-retryable (bad request, auth).
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_exc = e
                logger.warning(
                    f"OpenRouter transport error (attempt {attempt}/{_MAX_RETRIES}): {e!r}"
                )
            else:
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", _BACKOFF_BASE))
                    logger.warning(
                        f"OpenRouter rate-limited (attempt {attempt}/{_MAX_RETRIES}); "
                        f"sleeping {retry_after}s"
                    )
                    await asyncio.sleep(retry_after)
                    continue
                if 500 <= resp.status_code < 600:
                    logger.warning(
                        f"OpenRouter {resp.status_code} (attempt {attempt}/{_MAX_RETRIES})"
                    )
                else:
                    resp.raise_for_status()
                    try:
                        return self._extract_message_content(resp.json())
                    except _ProviderError as pe:
                        # 200-with-error - retryable (often transient provider
                        # routing issues). Capture and retry; surfaces if all
                        # attempts exhaust.
                        last_exc = pe
                        logger.warning(
                            f"OpenRouter returned a 200 with error body "
                            f"(attempt {attempt}/{_MAX_RETRIES}): {pe}"
                        )

            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_BACKOFF_BASE * (2 ** (attempt - 1)))

        # Exhausted retries.
        if last_exc:
            raise last_exc
        raise RuntimeError("OpenRouter call failed after retries (last response was 5xx)")

    @staticmethod
    def _extract_message_content(data: dict) -> str:
        """Pull the assistant message text out of an OpenRouter /chat/completions
        response. Raises _ProviderError when the response is shaped like an
        error envelope rather than a normal completion - the retry loop in
        _chat treats this as a transient failure."""
        # OpenRouter sometimes returns errors as HTTP 200 with one of these
        # shapes instead of the normal {"choices": [...]} envelope:
        #   {"error": {"message": "...", "code": 402}}
        #   {"error": "string message"}
        err = data.get("error") if isinstance(data, dict) else None
        if err is not None:
            if isinstance(err, dict):
                msg = err.get("message") or str(err)
                code = err.get("code")
                code_part = f" [{code}]" if code is not None else ""
                raise _ProviderError(f"OpenRouter API error{code_part}: {msg}")
            raise _ProviderError(f"OpenRouter API error: {err}")

        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            # Shape we don't recognize - surface a snippet so ops can diagnose.
            snippet = str(data)[:300]
            raise _ProviderError(f"OpenRouter response missing 'choices': {snippet}")

        try:
            return choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise _ProviderError(
                f"OpenRouter response shape unexpected: {e!r}; data={str(data)[:300]}"
            ) from e

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        blocks = []
        for c in chunks:
            header_parts = [f"chunk_id={c.get('chunk_id')}"]
            if c.get("title"):
                header_parts.append(f"title={c['title']!r}")
            if c.get("content_type"):
                header_parts.append(f"type={c['content_type']}")
            if c.get("page") is not None:
                header_parts.append(f"page={c['page']}")
            header_parts.append(f"source={c.get('source')}")
            blocks.append(
                f"[{' | '.join(header_parts)}]\n{c.get('text', '')}"
            )
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _parse_json(raw: str) -> Any:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                return json.loads(repair_json(raw))
            except Exception:
                return None
