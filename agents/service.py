"""Публичный фасад агентского слоя.

Бэкенду и внешним скриптам не нужно знать про LangGraph, узлы и состояние.
Достаточно создать ``AgentService`` и вызвать ``answer`` или ``aanswer``.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from agents.config import Settings, get_settings
from agents.graph import build_graph
from agents.llm import ensure_llm_configured
from agents.state import AgentResult
from rag.rag_pipeline import RAG

__all__ = ["AgentService"]

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class AgentService:
    """Сервис, который превращает вопрос пользователя в готовый ответ.

    Настройки LLM обязательны: без ``LLM_BASE_URL`` / ``LLM_API_KEY`` /
    ``LLM_MODEL`` конструктор падает с понятной ошибкой при старте.

    Parameters
    ----------
    rag_client:
        Готовый RAG с методом ``search_chunks``.
    settings:
        Настройки агента. Если не переданы, читаются из окружения.
    """

    def __init__(
        self,
        rag_client: RAG,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        ensure_llm_configured(self._settings)
        self._graph = build_graph(rag_client, self._settings)

    def _to_result(self, result: dict[str, Any]) -> AgentResult:
        answer = (result.get("answer") or "").strip()
        status = result.get("status") or "refused"
        if not answer:
            answer = "Не удалось сформировать ответ."
            status = "refused"

        return AgentResult(
            answer=answer,
            status=status,
            clarification=result.get("clarification"),
        )

    def _answer_sync(self, question: str) -> AgentResult:
        question = (question or "").strip()
        if not question:
            raise ValueError("Вопрос не может быть пустым")

        result = self._graph.invoke(
            {"query": question},
            config={"recursion_limit": 30},
        )
        return self._to_result(result)

    async def aanswer(self, question: str) -> AgentResult:
        """Асинхронно ответить на вопрос.

        Синхронный граф выполняется в отдельном потоке, поэтому event loop
        не блокируется.

        Parameters
        ----------
        question:
            Вопрос пользователя. Может содержать историю в формате
            ``Пользователь: ...`` / ``Ассистент: ...``.

        Returns
        -------
        AgentResult
            Ответ агента.
        """
        return await asyncio.to_thread(self._answer_sync, question)

    def answer(self, question: str) -> str:
        """Синхронно ответить на вопрос.

        Удобна для скриптов и batch-запуска. Внутри уже работающего event
        loop (например, в Jupyter или FastAPI) использовать нельзя — там
        нужен ``await`` к :meth:`aanswer`.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._answer_sync(question).answer

        raise RuntimeError(
            "answer() нельзя вызывать внутри работающего event loop. "
            "Используйте: await service.aanswer(question)."
        )
