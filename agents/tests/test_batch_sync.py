from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from agents.config import Settings
from agents.service import AgentService
from agents.state import (
    RetrievalGradeResult,
    SubQuery,
    SubqueryGrade,
    UnderstandResult,
    VerificationResult,
)
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


def make_understand(question: str) -> UnderstandResult:
    return UnderstandResult(
        standalone_query=question,
        in_scope=True,
        scope_reason="тест",
        is_complex=False,
        subqueries=[
            SubQuery(id=1, text=question, part_ids=[1], kind="fact", required=True)
        ],
    )


def make_grade() -> RetrievalGradeResult:
    return RetrievalGradeResult(
        subqueries=[
            SubqueryGrade(
                subquery_id=1,
                coverage="full",
                missing_aspects=[],
                reason="ok",
            )
        ],
        overall="full",
        missing_aspects=[],
    )


def make_verify() -> VerificationResult:
    return VerificationResult(
        verdict="pass",
        grounded=True,
        answers_question=True,
        citation_ok=True,
        premise_ok=True,
        failure_type="none",
        reason="ok",
    )


class BatchSyncTest(unittest.TestCase):
    def test_parallel_answers_do_not_share_loop_or_history(self) -> None:
        rag = FakeRAG()
        settings = Settings(
            llm_base_url="http://llm.local/v1",
            llm_api_key="test-key",
            llm_model="test-model",
        )
        questions = [f"вопрос {index}" for index in range(8)]

        with (
            patch(
                "agents.nodes.understand_query",
                Mock(side_effect=lambda _history, question, **_: make_understand(question)),
            ),
            patch("agents.nodes.grade_retrieval", Mock(return_value=make_grade())),
            patch(
                "agents.nodes.generate_answer",
                Mock(
                    side_effect=lambda question, *_args, **_kwargs: (
                        f"Ответ на {question} [doc_id: doc-1]"
                    )
                ),
            ),
            patch("agents.nodes.verify_answer", Mock(return_value=make_verify())),
        ):
            service = AgentService(rag, settings=settings)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(service.answer, question) for question in questions
                ]
                answers = [future.result() for future in futures]

        for question, answer in zip(questions, answers):
            self.assertIn(question, answer)

        self.assertEqual(len(rag.calls), len(questions))
        self.assertCountEqual(rag.calls, questions)


if __name__ == "__main__":
    unittest.main()
