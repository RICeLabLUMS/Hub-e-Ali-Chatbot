import json
import logging
from typing import Any

import httpx
from json_repair import repair_json

from app.core.config import settings

logger = logging.getLogger(__name__)

LANG_NAMES = {"en": "English", "ar": "Arabic", "ur": "Urdu"}

ANSWER_SYSTEM_TEMPLATE = (
    "You answer questions about Islamic texts. Reply STRICTLY as JSON matching:\n"
    '{{"answer": "<answer in {lang_name}>", "citations": [{{"chunk_id": "...", "source": "...", "page": <int>}}]}}\n\n'
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

        raw = await self._chat(
            system=system,
            user=user,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

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

        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        blocks = []
        for c in chunks:
            blocks.append(
                f"[chunk_id={c.get('chunk_id')} | source={c.get('source')} | page={c.get('page')}]\n"
                f"{c.get('text', '')}"
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
