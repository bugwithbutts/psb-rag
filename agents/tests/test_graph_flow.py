from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from agents.config import Settings
from agents.nodes import (
    make_generate_node,
    make_retrieve_node,
    make_rewrite_node,
    make_understand_node,
    route_after_grade,
    route_after_verify,
)
from agents.service import AgentService
from agents.state import (
    ClarificationRequest,
    ContextChunk,
    DialogTurn,
    RetrievalGradeResult,
    SubQuery,
    SubqueryGrade,
    UnderstandResult,
    VerificationResult,
)
from agents.tokens import TokenUsage
from rag.index.contracts import RetrievedChunk


class FakeRAG:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search_chunks(self, query: str) -> list[RetrievedChunk]:
        self.calls.append(query)
        return [
            RetrievedChunk(
                doc_id="doc-1",
                chunk_id="chunk-1",
                text="Текст про налоги.",
                metadata={"title": "Документ"},
                score=0.9,
            )
        ]


class TwoContractRAG(FakeRAG):
    def search_chunks(self, query: str) -> list[RetrievedChunk]:
        self.calls.append(query)
        return [
            RetrievedChunk(
                doc_id="документы компании__Договор поставки П-17-2024 Альфа-Трейд.txt",
                chunk_id="alfa-1",
                text="Гарантийный срок — 24 месяца.",
                metadata={
                    "title": "документы компании__Договор поставки П-17-2024 Альфа-Трейд.txt"
                },
                score=0.9,
            ),
            RetrievedChunk(
                doc_id="документы компании__Договор поставки ПТ-05-2023 Альфа-Трейдинг.txt",
                chunk_id="trading-1",
                text="Гарантийный срок — 12 месяцев.",
                metadata={
                    "title": "документы компании__Договор поставки ПТ-05-2023 Альфа-Трейдинг.txt"
                },
                score=0.8,
            ),
        ]


def make_understand(in_scope: bool = True) -> UnderstandResult:
    subquery = SubQuery(id=1, text="налоги", part_ids=[1], kind="fact")
    return UnderstandResult(
        standalone_query="налоги",
        in_scope=in_scope,
        scope_reason="тест",
        is_complex=False,
        subqueries=[subquery],
    )


def make_grade(coverage: str) -> RetrievalGradeResult:
    return RetrievalGradeResult(
        subqueries=[
            SubqueryGrade(
                subquery_id=1,
                coverage=coverage,
                missing_aspects=[] if coverage == "full" else ["нет данных"],
            )
        ],
        overall=coverage,
        missing_aspects=[] if coverage == "full" else ["нет данных"],
    )


def make_verify(verdict: str) -> VerificationResult:
    return VerificationResult(
        verdict=verdict,
        grounded=verdict == "pass",
        answers_question=True,
        citation_ok=True,
        premise_ok=True,
        failure_type="none" if verdict == "pass" else "retrieval",
    )


class RetrieveNodeTest(unittest.TestCase):
    def test_increments_attempts_after_search(self) -> None:
        rag = FakeRAG()
        state = {
            "subqueries": [SubQuery(id=1, text="налоги")],
            "contexts": [],
            "retrieval_attempts": 0,
        }

        result = make_retrieve_node(rag)(state)

        self.assertEqual(rag.calls, ["налоги"])
        self.assertEqual(result["subqueries"][0].attempts, 1)
        self.assertEqual(result["subqueries"][0].tried_queries, ["налоги"])
        self.assertEqual(result["retrieval_attempts"], 1)

    def test_skips_tried_query(self) -> None:
        rag = FakeRAG()
        state = {
            "subqueries": [
                SubQuery(id=1, text="налоги", tried_queries=["налоги"], attempts=1)
            ],
            "contexts": [],
            "retrieval_attempts": 1,
        }

        result = make_retrieve_node(rag)(state)

        self.assertEqual(rag.calls, [])
        self.assertEqual(result["subqueries"][0].attempts, 1)
        self.assertEqual(result["subqueries"][0].tried_queries, ["налоги"])
        self.assertEqual(result["retrieval_attempts"], 2)
        self.assertEqual(result["contexts"], [])

    def test_retries_query_after_rag_exception(self) -> None:
        class FlakyRAG:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def search_chunks(self, query: str) -> list[RetrievedChunk]:
                self.calls.append(query)
                if len(self.calls) == 1:
                    raise RuntimeError("boom")
                return [
                    RetrievedChunk(
                        doc_id="doc-1",
                        chunk_id="chunk-1",
                        text="Текст про налоги.",
                        metadata={"title": "Документ"},
                        score=0.9,
                    )
                ]

        rag = FlakyRAG()
        state = {
            "subqueries": [SubQuery(id=1, text="налоги")],
            "contexts": [],
            "retrieval_attempts": 0,
        }
        node = make_retrieve_node(rag)

        with self.assertLogs("agents.nodes", level="WARNING"):
            first = node(state)
        self.assertEqual(first["subqueries"][0].tried_queries, [])
        self.assertEqual(first["subqueries"][0].attempts, 0)

        second_state = {**state, **first}
        second = node(second_state)

        self.assertEqual(rag.calls, ["налоги", "налоги"])
        self.assertEqual(second["subqueries"][0].tried_queries, ["налоги"])
        self.assertEqual(second["subqueries"][0].attempts, 1)
        self.assertEqual(
            [chunk.chunk_id for chunk in second["contexts"]],
            ["chunk-1"],
        )

    def test_keeps_old_chunks_for_subquery(self) -> None:
        rag = FakeRAG()
        old_chunk = ContextChunk(
            doc_id="doc-1",
            chunk_id="old-1",
            text="Старый текст.",
            subquery_ids=[1],
        )
        state = {
            "subqueries": [
                SubQuery(id=1, text="налоги", tried_queries=["налоги"], attempts=1)
            ],
            "contexts": [old_chunk],
            "retrieval_attempts": 1,
        }

        result = make_retrieve_node(rag)(state)

        self.assertEqual(rag.calls, [])
        self.assertEqual(
            [chunk.chunk_id for chunk in result["contexts"]],
            ["old-1"],
        )


class GenerateNodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            llm_base_url="http://llm.local/v1",
            llm_api_key="test-key",
            llm_model="test-model",
        )

    def test_generates_answer_from_context(self) -> None:
        generate = Mock(return_value="Ответ по контексту.")
        history = [DialogTurn(role="user", text="раньше")]
        contexts = [ContextChunk(doc_id="doc-1", chunk_id="chunk-1", text="Налоги.")]
        state = {"query": "вопрос", "history": history, "contexts": contexts}

        with patch("agents.nodes.generate_answer", new=generate):
            result = make_generate_node(self.settings)(state)

        self.assertEqual(result["answer_draft"], "Ответ по контексту.")
        self.assertEqual(result["answer_attempts"], 1)
        self.assertEqual(generate.call_args.args[0], "вопрос")
        self.assertEqual(generate.call_args.args[1], contexts)
        self.assertEqual(generate.call_args.args[2], history)

    def test_no_context_does_not_call_llm(self) -> None:
        generate = Mock(side_effect=AssertionError("LLM must not be called"))
        state = {"query": "вопрос", "contexts": []}

        with patch("agents.nodes.generate_answer", new=generate):
            result = make_generate_node(self.settings)(state)

        self.assertIn("нет документов", result["answer_draft"])
        self.assertEqual(result["answer_attempts"], 1)
        generate.assert_not_called()


class RouteAfterVerifyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            llm_base_url="http://llm.local/v1",
            llm_api_key="test-key",
            llm_model="test-model",
        )

    def test_retries_when_core_answer_unsupported(self) -> None:
        state = {
            "verification": VerificationResult(core_answer_supported=False),
            "contexts": [ContextChunk(doc_id="doc-1", chunk_id="chunk-1")],
            "answer_attempts": 1,
            "total_steps": 1,
        }

        self.assertEqual(route_after_verify(state, self.settings), "generate")

    def test_finalizes_when_core_answer_supported(self) -> None:
        state = {
            "verification": VerificationResult(core_answer_supported=True),
            "contexts": [ContextChunk(doc_id="doc-1", chunk_id="chunk-1")],
            "answer_attempts": 1,
            "total_steps": 1,
        }

        self.assertEqual(route_after_verify(state, self.settings), "finalize")

    def test_finalizes_when_answer_attempts_exhausted(self) -> None:
        settings = replace(self.settings, max_answer_attempts=1)
        state = {
            "verification": VerificationResult(core_answer_supported=False),
            "contexts": [ContextChunk(doc_id="doc-1", chunk_id="chunk-1")],
            "answer_attempts": 1,
            "total_steps": 1,
        }

        self.assertEqual(route_after_verify(state, settings), "finalize")


class GraphFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rag = FakeRAG()
        self.settings = Settings(
            llm_base_url="http://llm.local/v1",
            llm_api_key="test-key",
            llm_model="test-model",
        )

    def _patch(self, name: str, value: object) -> None:
        patcher = patch(f"agents.nodes.{name}", new=value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_out_of_scope_does_not_retrieve(self) -> None:
        self._patch("understand_query", Mock(return_value=make_understand(False)))
        service = AgentService(self.rag, settings=self.settings)

        answer = asyncio.run(service.aanswer("кто выиграет чемпионат?"))

        self.assertIn("выходит за тематику", answer.answer)
        self.assertEqual(self.rag.calls, [])

    def test_clarification_before_retrieval(self) -> None:
        understand = make_understand()
        understand.clarification = ClarificationRequest(
            needed=True,
            question="Уточните, пожалуйста, номер договора.",
            missing_slots=["номер договора"],
        )
        self._patch("understand_query", Mock(return_value=understand))
        service = AgentService(self.rag, settings=self.settings)

        result = asyncio.run(
            service.aanswer(
                "Мы арендуем склад у «Северного Склада». Планируем съезжать.",
            )
        )

        self.assertEqual(result.status, "clarification")
        self.assertIn("номер договора", result.answer)
        self.assertIsNotNone(result.clarification)
        self.assertEqual(self.rag.calls, [])

    def test_searches_complete_question_before_asking_for_contract_number(self) -> None:
        understand = make_understand()
        understand.standalone_query = (
            "По договору с «Северным Складом» нас должны пропускать ночью?"
        )
        understand.subqueries = [
            SubQuery(
                id=1,
                text=understand.standalone_query,
                part_ids=[1],
                kind="fact",
            )
        ]
        understand.clarification = ClarificationRequest(
            needed=True,
            question="Уточните, пожалуйста, номер договора.",
            missing_slots=["номер договора"],
        )
        self._patch("understand_query", Mock(return_value=understand))
        self._patch("grade_retrieval", Mock(return_value=make_grade("full")))
        self._patch(
            "generate_answer",
            Mock(return_value="Ответ [doc_id: doc-1]"),
        )
        self._patch("verify_answer", Mock(return_value=make_verify("pass")))
        service = AgentService(self.rag, settings=self.settings)

        result = asyncio.run(
            service.aanswer(
                "По договору с «Северным Складом» нас должны пропускать ночью?",
            )
        )

        self.assertNotEqual(result.status, "clarification")
        self.assertEqual(self.rag.calls, [understand.standalone_query])

    def test_clarification_is_not_repeated_before_retrieval(self) -> None:
        understand = make_understand()
        understand.clarification = ClarificationRequest(
            needed=True,
            question="Уточните, пожалуйста, номер договора.",
        )
        self._patch("understand_query", Mock(return_value=understand))
        self._patch("grade_retrieval", Mock(return_value=make_grade("full")))
        self._patch(
            "generate_answer",
            Mock(return_value="Ответ [doc_id: doc-1]"),
        )
        self._patch("verify_answer", Mock(return_value=make_verify("pass")))
        service = AgentService(self.rag, settings=self.settings)

        result = asyncio.run(
            service.aanswer(
                "Пользователь: Мы арендуем склад. Планируем съезжать.\n"
                "Ассистент: Уточните номер договора?\n"
                "Пользователь: Не знаю.",
            )
        )

        self.assertNotEqual(result.status, "clarification")
        self.assertEqual(self.rag.calls, ["налоги"])

    def test_clarification_after_retrieval_ambiguity(self) -> None:
        rag = TwoContractRAG()
        understand = make_understand()
        understand.standalone_query = "Подскажите гарантию по договору с «Альфой»."
        understand.subqueries = [
            SubQuery(
                id=1,
                text=understand.standalone_query,
                part_ids=[1],
                kind="fact",
            )
        ]
        generate = Mock(return_value="Не должно вызываться.")
        self._patch("understand_query", Mock(return_value=understand))
        self._patch("grade_retrieval", Mock(return_value=make_grade("full")))
        self._patch("generate_answer", generate)
        service = AgentService(rag, settings=self.settings)

        result = asyncio.run(
            service.aanswer("Подскажите гарантию по договору с «Альфой».")
        )

        self.assertEqual(result.status, "clarification")
        self.assertIn("Альфа-Трейд", result.answer)
        self.assertIn("Альфа-Трейдинг", result.answer)
        self.assertEqual(rag.calls, ["Подскажите гарантию по договору с «Альфой»."])
        generate.assert_not_called()

    def test_skipped_understand_does_not_consume_step(self) -> None:
        def skipped(_history, _question, *, usage_sink, **_kwargs) -> UnderstandResult:
            usage_sink.append(TokenUsage(skipped=True))
            return make_understand()

        self._patch("understand_query", Mock(side_effect=skipped))
        node = make_understand_node(self.settings)

        result = node({"query": "налоги", "total_steps": 3, "used_tokens": 0})

        self.assertEqual(result["total_steps"], 3)

    def test_understand_clarification_survives_empty_retrieval(self) -> None:
        understand = make_understand()
        understand.standalone_query = "По договору с «Альфой» какие налоги?"
        understand.subqueries = [
            SubQuery(
                id=1,
                text=understand.standalone_query,
                part_ids=[1],
                kind="fact",
            )
        ]
        understand.clarification = ClarificationRequest(
            needed=True,
            question="Уточните, пожалуйста, номер договора.",
            missing_slots=["номер договора"],
        )
        self._patch("understand_query", Mock(return_value=understand))
        self._patch("grade_retrieval", Mock(return_value=make_grade("none")))
        service = AgentService(self.rag, settings=self.settings)

        result = asyncio.run(service.aanswer("По договору с «Альфой» какие налоги?"))

        self.assertEqual(result.status, "clarification")
        self.assertIn("номер договора", result.answer)

    def test_retrieval_rewrite_cycle(self) -> None:
        self._patch("understand_query", Mock(return_value=make_understand()))
        self._patch(
            "grade_retrieval",
            Mock(side_effect=[make_grade("none"), make_grade("full")]),
        )
        self._patch("rewrite_subquery", Mock(return_value="налоги и сборы"))
        self._patch(
            "generate_answer",
            Mock(return_value="Ответ [doc_id: doc-1]"),
        )
        self._patch("verify_answer", Mock(return_value=make_verify("pass")))
        service = AgentService(self.rag, settings=self.settings)

        answer = asyncio.run(service.aanswer("налоги"))

        self.assertIn("Ответ", answer.answer)
        self.assertEqual(
            self.rag.calls,
            [
                "налоги",
                "налоги и сборы",
            ],
        )

    def test_finalize_humanizes_sources(self) -> None:
        self._patch("understand_query", Mock(return_value=make_understand()))
        self._patch("grade_retrieval", Mock(return_value=make_grade("full")))
        self._patch(
            "generate_answer",
            Mock(return_value="Ответ [doc_id: doc-1]"),
        )
        self._patch("verify_answer", Mock(return_value=make_verify("pass")))
        service = AgentService(self.rag, settings=self.settings)

        answer = asyncio.run(service.aanswer("налоги"))

        self.assertIn("[Документ]", answer.answer)
        self.assertNotIn("doc_id", answer.answer)
        self.assertNotIn("Источники:", answer.answer)

    def test_rewrite_is_limited_per_round(self) -> None:
        subqueries = [
            SubQuery(
                id=index,
                text=f"подзапрос {index}",
                part_ids=[1],
                kind="fact",
            )
            for index in range(1, 5)
        ]
        understand = UnderstandResult(
            standalone_query="сложный вопрос",
            in_scope=True,
            scope_reason="тест",
            is_complex=True,
            subqueries=subqueries,
        )
        self._patch("understand_query", Mock(return_value=understand))
        self._patch(
            "grade_retrieval",
            Mock(side_effect=[make_grade("none"), make_grade("full")]),
        )
        rewrite = Mock(return_value="новый запрос")
        self._patch("rewrite_subquery", rewrite)
        self._patch(
            "generate_answer",
            Mock(return_value="Ответ [doc_id: doc-1]"),
        )
        self._patch("verify_answer", Mock(return_value=make_verify("pass")))
        settings = Settings(
            llm_base_url="http://llm.local/v1",
            llm_api_key="test-key",
            llm_model="test-model",
            max_rewrite_subqueries_per_round=2,
        )
        service = AgentService(self.rag, settings=settings)

        asyncio.run(service.aanswer("сложный вопрос"))

        self.assertEqual(rewrite.call_count, 2)

    def test_soft_budget_skips_optional_rewrite(self) -> None:
        state = {
            "coverage": "partial",
            "total_steps": 2,
            "retrieval_attempts": 1,
            "used_tokens": self.settings.max_total_tokens_soft,
        }

        self.assertEqual(route_after_grade(state, self.settings), "generate")

        state["coverage"] = "none"
        self.assertEqual(route_after_grade(state, self.settings), "finalize")

    def test_context_is_generated_even_when_grade_is_none(self) -> None:
        state = {
            "coverage": "none",
            "contexts": [ContextChunk(text="По договору плата 450 000 рублей.")],
            "total_steps": 0,
            "retrieval_attempts": 1,
        }

        self.assertEqual(route_after_grade(state, self.settings), "generate")

    def test_rewrite_is_not_repeated_for_subquery(self) -> None:
        subquery = SubQuery(
            id=1,
            text="налоги и сборы",
            coverage="none",
            attempts=5,
        )
        rewrite = Mock(side_effect=AssertionError("rewrite must not be called"))
        self._patch("rewrite_subquery", rewrite)

        result = make_rewrite_node(self.settings)(
            {
                "subqueries": [subquery],
                "contexts": [],
                "total_steps": 0,
                "used_tokens": 0,
            }
        )

        self.assertEqual(result["subqueries"], [subquery])
        rewrite.assert_not_called()
        self.assertEqual(
            route_after_grade(
                {
                    "coverage": "none",
                    "subqueries": [subquery],
                    "retrieval_attempts": 5,
                    "total_steps": 2,
                },
                self.settings,
            ),
            "generate",
        )

    def test_rewrite_records_old_query_without_incrementing_attempts(self) -> None:
        subquery = SubQuery(
            id=1,
            text="налоги и сборы",
            coverage="none",
            attempts=1,
        )
        rewrite = Mock(return_value="налоги сборы синоним")
        self._patch("rewrite_subquery", rewrite)

        result = make_rewrite_node(self.settings)(
            {
                "subqueries": [subquery],
                "contexts": [],
                "total_steps": 0,
                "used_tokens": 0,
            }
        )
        updated = result["subqueries"][0]

        self.assertEqual(updated.text, "налоги сборы синоним")
        self.assertEqual(updated.tried_queries, ["налоги и сборы"])
        self.assertEqual(updated.attempts, 1)
        self.assertEqual(result["total_steps"], 1)
        self.assertNotIn("tried_queries", result)
        self.assertEqual(rewrite.call_args.args[3], [])

    def test_route_rewrites_only_none_subqueries(self) -> None:
        state = {
            "coverage": "none",
            "subqueries": [
                SubQuery(id=1, coverage="none", attempts=1),
                SubQuery(id=2, coverage="partial", attempts=1),
            ],
            "contexts": [ContextChunk(text="Есть часть сведений.")],
            "retrieval_attempts": 1,
            "total_steps": 1,
            "used_tokens": 0,
        }

        self.assertEqual(route_after_grade(state, self.settings), "rewrite")

        state["coverage"] = "partial"
        state["subqueries"][0] = state["subqueries"][0].model_copy(
            update={"coverage": "partial"}
        )
        self.assertEqual(route_after_grade(state, self.settings), "generate")

    def test_partial_subquery_goes_to_generate(self) -> None:
        state = {
            "coverage": "partial",
            "subqueries": [SubQuery(id=1, coverage="partial", attempts=1)],
            "contexts": [ContextChunk(text="Есть часть сведений.")],
            "retrieval_attempts": 1,
            "total_steps": 1,
            "used_tokens": 0,
        }

        self.assertEqual(route_after_grade(state, self.settings), "generate")

    def test_hard_budget_prevents_llm_calls(self) -> None:
        builder = patch(
            "agents.llm._build_llm",
            side_effect=AssertionError("real LLM must not be called"),
        )
        builder.start()
        self.addCleanup(builder.stop)
        settings = Settings(
            llm_base_url="http://llm.local/v1",
            llm_api_key="test-key",
            llm_model="test-model",
            max_total_tokens_per_question=100,
            max_total_tokens_soft=50,
        )
        service = AgentService(self.rag, settings=settings)

        answer = asyncio.run(service.aanswer("налоги"))

        self.assertIn("нет документов", answer.answer)

    def test_no_data_refuses_after_retrieval_limit(self) -> None:
        self._patch("understand_query", Mock(return_value=make_understand()))
        self._patch(
            "grade_retrieval",
            Mock(side_effect=[make_grade("none"), make_grade("none")]),
        )
        self._patch("rewrite_subquery", Mock(return_value="другой запрос"))
        settings = replace(self.settings, max_retrieval_attempts=2)
        service = AgentService(self.rag, settings=settings)

        answer = asyncio.run(service.aanswer("неизвестный вопрос"))

        self.assertIn("нет документов", answer.answer)
        self.assertEqual(len(self.rag.calls), 2)


if __name__ == "__main__":
    unittest.main()
