from __future__ import annotations

import asyncio
import unittest
from unittest.mock import Mock, patch

from agents.config import Settings
from agents.service import AgentService
from rag.index.contracts import RetrievedChunk


def make_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        doc_id="doc-1",
        chunk_id="chunk-1",
        text="Текст документа про налоги.",
        metadata={"title": "Договор"},
        score=0.9,
    )


class FakeRAG:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[str] = []

    def search_chunks(self, query: str) -> list[RetrievedChunk]:
        self.calls.append(query)
        return self.chunks


def completion_responses() -> list[str]:
    return [
        (
            '{"standalone_query":"налоги","in_scope":true,'
            '"scope_reason":"ok","is_complex":false,'
            '"question_parts":[{"id":1,"text":"налоги","kind":"fact"}],'
            '"subqueries":[{"id":1,"text":"налоги","part_ids":[1],'
            '"kind":"fact","required":true}]}'
        ),
        (
            '{"subqueries":[{"subquery_id":1,"coverage":"full",'
            '"missing_aspects":[],"reason":"ok"}],'
            '"overall":"full","missing_aspects":[]}'
        ),
        "Ответ по документу [doc_id: doc-1]",
        (
            '{"verdict":"pass","grounded":true,"answers_question":true,'
            '"citation_ok":true,"premise_ok":true,"failure_type":"none",'
            '"missing_evidence":[],"fix_hint":"","reason":"ok"}'
        ),
    ]


class AgentServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rag = FakeRAG([make_chunk()])
        completion = patch(
            "agents.llm._chat_completion",
            new=Mock(side_effect=completion_responses()),
        )
        self.completion = completion.start()
        self.addCleanup(completion.stop)

        builder = patch(
            "agents.llm._build_llm",
            side_effect=AssertionError("real LLM must not be called"),
        )
        builder.start()
        self.addCleanup(builder.stop)

        self.service = AgentService(
            self.rag,
            settings=Settings(
                llm_base_url="http://llm.local/v1",
                llm_api_key="test-key",
                llm_model="test-model",
            ),
        )

    def test_missing_llm_settings_raise_value_error(self) -> None:
        with self.assertRaises(ValueError) as raised:
            AgentService(
                self.rag,
                settings=Settings(llm_base_url="", llm_api_key="", llm_model=""),
            )

        self.assertIn("LLM_API_KEY", str(raised.exception))

    def test_sync_answer_returns_generated_answer(self) -> None:
        answer = self.service.answer("налоги")

        self.assertIn("[Договор]", answer)
        self.assertIn("Ответ по документу", answer)
        self.assertNotIn("Источники:", answer)
        self.assertNotIn("doc_id", answer)

    def test_async_answer_returns_result(self) -> None:
        result = asyncio.run(self.service.aanswer("налоги"))

        self.assertIn("Ответ по документу", result.answer)

    def test_search_receives_standalone_subquery(self) -> None:
        asyncio.run(self.service.aanswer("налоги"))

        self.assertEqual(self.rag.calls, ["налоги"])

    def test_answer_uses_fresh_minimal_state(self) -> None:
        with patch.object(
            self.service._graph,
            "invoke",
            return_value={"answer": "ok", "status": "answered"},
        ) as invoke:
            answer = self.service.answer("налоги")

        invoke.assert_called_once_with(
            {"query": "налоги"},
            config={"recursion_limit": 30},
        )
        self.assertEqual(answer, "ok")

    def test_empty_question_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.service.answer("   ")

    def test_sync_answer_inside_event_loop_raises(self) -> None:
        async def call_sync() -> None:
            with self.assertRaises(RuntimeError):
                self.service.answer("вопрос")

        asyncio.run(call_sync())

    def test_async_answer_works_inside_running_loop(self) -> None:
        async def call_async() -> None:
            result = await self.service.aanswer("налоги")
            self.assertIn("Ответ по документу", result.answer)

        asyncio.run(call_async())


if __name__ == "__main__":
    unittest.main()
