from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage, SystemMessage

from agents.config import Settings
from agents.llm import _chat_completion, generate_answer
from agents.state import ContextChunk
from agents.tokens import (
    TokenUsage,
    estimate_messages_tokens,
    estimate_tokens,
    extract_usage,
    sum_usage,
)


def make_settings(**overrides) -> Settings:
    data = {
        "llm_base_url": "http://llm.local/v1",
        "llm_api_key": "test-key",
        "llm_model": "test-model",
    }
    data.update(overrides)
    return Settings(**data)


def make_context() -> ContextChunk:
    return ContextChunk(
        doc_id="doc-1",
        chunk_id="chunk-1",
        text="Сотрудник обязан уплатить налог до 1 декабря.",
        metadata={"title": "Налоговый кодекс"},
        score=0.9,
        subquery_ids=[1],
    )


class EstimateTokensTest(unittest.TestCase):
    def test_empty_text(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)

    def test_estimates_conservatively(self) -> None:
        self.assertEqual(estimate_tokens("а" * 170), 100)

    def test_estimates_messages(self) -> None:
        messages = [SystemMessage(content="а" * 170), HumanMessage(content="б" * 85)]

        self.assertEqual(estimate_messages_tokens(messages), 150)

    def test_sum_usage(self) -> None:
        usages = [
            TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            TokenUsage(input_tokens=20, output_tokens=0, total_tokens=20),
        ]

        self.assertEqual(sum_usage(usages), 35)


class ExtractUsageTest(unittest.TestCase):
    def test_prefers_usage_metadata(self) -> None:
        response = SimpleNamespace(
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        )

        usage = extract_usage(response, [], "ответ")

        self.assertEqual(usage.total_tokens, 150)
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.output_tokens, 50)

    def test_reads_response_metadata_token_usage(self) -> None:
        response = SimpleNamespace(
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 10,
                    "total_tokens": 40,
                }
            }
        )

        usage = extract_usage(response, [], "ответ")

        self.assertEqual(usage.total_tokens, 40)
        self.assertEqual(usage.input_tokens, 30)
        self.assertEqual(usage.output_tokens, 10)

    def test_falls_back_to_estimate(self) -> None:
        response = SimpleNamespace()
        messages = [SystemMessage(content="а" * 170)]
        output = "б" * 34

        usage = extract_usage(response, messages, output)

        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.output_tokens, 20)
        self.assertEqual(usage.total_tokens, 120)


class ChatCompletionBudgetTest(unittest.TestCase):
    def test_records_actual_usage(self) -> None:
        fake_llm = SimpleNamespace(
            invoke=Mock(
                return_value=SimpleNamespace(
                    content="ответ",
                    usage_metadata={
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                )
            )
        )
        usage: list[TokenUsage] = []

        with patch("agents.llm._build_llm", return_value=fake_llm):
            result = _chat_completion(
                settings=make_settings(),
                messages=[SystemMessage(content="промпт")],
                max_tokens=100,
                usage_sink=usage,
            )

        self.assertEqual(result, "ответ")
        self.assertEqual(usage[0].total_tokens, 15)

    def test_skips_call_when_budget_is_exhausted(self) -> None:
        usage: list[TokenUsage] = []

        with patch(
            "agents.llm._build_llm",
            side_effect=AssertionError("real LLM must not be called"),
        ):
            result = _chat_completion(
                settings=make_settings(),
                messages=[SystemMessage(content="а" * 1000)],
                max_tokens=500,
                usage_sink=usage,
                budget_remaining=100,
            )

        self.assertEqual(result, "")
        self.assertTrue(usage[0].skipped)
        self.assertEqual(usage[0].total_tokens, 0)

    def test_generate_returns_no_data_when_budget_skipped(self) -> None:
        usage: list[TokenUsage] = []

        result = generate_answer(
            "вопрос",
            [make_context()],
            [],
            settings=make_settings(),
            usage_sink=usage,
            budget_remaining=1,
        )

        self.assertIn("нет документов", result)
        self.assertTrue(usage[0].skipped)


if __name__ == "__main__":
    unittest.main()
