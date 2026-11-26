from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import SystemMessage

from agents.config import Settings
from agents.llm import (
    _chat_completion,
    generate_answer,
    grade_retrieval,
    rewrite_subquery,
    understand_query,
    verify_answer,
)
from agents.prompts import UNDERSTAND_SYSTEM_PROMPT
from agents.state import ContextChunk, DialogTurn, SubQuery


def make_settings(**overrides) -> Settings:
    data = {
        "llm_base_url": "http://llm.local/v1",
        "llm_api_key": "test-key",
        "llm_model": "test-model",
    }
    data.update(overrides)
    return Settings(**data)


def make_context(subquery_id: int = 1) -> ContextChunk:
    return ContextChunk(
        doc_id="doc-1",
        chunk_id="chunk-1",
        text="Сотрудник обязан уплатить налог до 1 декабря.",
        metadata={"title": "Налоговый кодекс"},
        score=0.9,
        subquery_ids=[subquery_id],
    )


def make_subquery(text: str = "налоги", subquery_id: int = 1) -> SubQuery:
    return SubQuery(id=subquery_id, text=text, part_ids=[1], kind="fact")


class UnderstandQueryTest(unittest.TestCase):
    def test_parses_and_normalizes_complex_question(self) -> None:
        raw = (
            '{"standalone_query":"Какие налоги платит ИП по договору № 1?",'
            '"in_scope":true,"scope_reason":"налоги","is_complex":true,'
            '"question_parts":[{"id":1,"text":"налоги","kind":"fact"}],'
            '"subqueries":['
            '{"id":1,"text":"налоги ИП","part_ids":[1],"kind":"fact"},'
            '{"id":2,"text":"срок уплаты налогов ИП","part_ids":[1],"kind":"fact"}]}'
        )
        with patch("agents.llm._chat_completion", new=Mock(return_value=raw)):
            result = understand_query(
                [DialogTurn(role="user", text="Я ИП")],
                "Какие налоги?",
                settings=make_settings(),
            )

        self.assertEqual(result.standalone_query, "Какие налоги платит ИП по договору № 1?")
        self.assertTrue(result.is_complex)
        self.assertEqual(len(result.subqueries), 2)

    def test_accepts_prompt_shaped_json(self) -> None:
        self.assertIn('"standalone_query"', UNDERSTAND_SYSTEM_PROMPT)
        self.assertIn('"is_complex"', UNDERSTAND_SYSTEM_PROMPT)
        self.assertNotIn('"premise_check"', UNDERSTAND_SYSTEM_PROMPT)
        self.assertIn("ШАГ 4. Разбей вопрос на подзапросы.", UNDERSTAND_SYSTEM_PROMPT)
        self.assertIn("ШАГ 1. Сформируй standalone_query.", UNDERSTAND_SYSTEM_PROMPT)
        self.assertIn("всей истории", UNDERSTAND_SYSTEM_PROMPT)
        self.assertIn('" Контекст:', UNDERSTAND_SYSTEM_PROMPT)
        self.assertIn("ШАГ 2. Определи in_scope.", UNDERSTAND_SYSTEM_PROMPT)
        self.assertIn(" Отсутствие данных — задача поиска", UNDERSTAND_SYSTEM_PROMPT)
        self.assertIn(
            "ШАГ 3. Реши, нужно ли уточнение до поиска.",
            UNDERSTAND_SYSTEM_PROMPT,
        )
        self.assertIn(
            "критичной сущности нет ни в последнем вопросе",
            UNDERSTAND_SYSTEM_PROMPT,
        )
        raw = (
            '{"standalone_query":"Какие налоги платит ИП?",'
            '"in_scope":true,"scope_reason":"налоги","is_complex":true,'
            '"clarification":{"needed":false,"question":"","reason":""},'
            '"subqueries":['
            '{"id":1,"text":"налоги ИП"},'
            '{"id":2,"text":"сроки уплаты налогов ИП"}]}'
        )
        with patch("agents.llm._chat_completion", new=Mock(return_value=raw)):
            result = understand_query(
                [DialogTurn(role="user", text="Я ИП")],
                "Какие налоги?",
                settings=make_settings(),
            )

        self.assertEqual(result.standalone_query, "Какие налоги платит ИП?")
        self.assertEqual(len(result.subqueries), 2)

    def test_preserves_entity_context_tail(self) -> None:
        raw = (
            '{"standalone_query":"Какие сроки уплаты НДС? Контекст: '
            'договор № A-08/2024; ООО «Ромашка»; аренда склада.",'
            '"in_scope":true,"scope_reason":"налоги","is_complex":false,'
            '"subqueries":[{"id":1,"text":"сроки уплаты НДС"}]}'
        )
        with patch("agents.llm._chat_completion", new=Mock(return_value=raw)):
            result = understand_query(
                [
                    DialogTurn(
                        role="user",
                        text="Договор № A-08/2024 с ООО «Ромашка» на аренду склада.",
                    )
                ],
                "Какие сроки уплаты НДС?",
                settings=make_settings(),
            )

        self.assertEqual(
            result.standalone_query,
            "Какие сроки уплаты НДС? Контекст: договор № A-08/2024; "
            "ООО «Ромашка»; аренда склада.",
        )
        self.assertEqual(result.subqueries[0].text, "сроки уплаты НДС")

    def test_empty_response_logs_fallback_reason(self) -> None:
        with patch("agents.llm._chat_completion", new=Mock(return_value="")):
            with self.assertLogs("agents.llm", level="WARNING") as captured:
                result = understand_query([], "налоги", settings=make_settings())

        self.assertEqual(result.standalone_query, "налоги")
        self.assertTrue(any("пустой ответ" in line for line in captured.output))

    def test_simple_question_keeps_one_subquery(self) -> None:
        raw = (
            '{"standalone_query":"налоги","in_scope":true,'
            '"is_complex":false,"subqueries":['
            '{"id":1,"text":"налоги","part_ids":[1]},'
            '{"id":2,"text":"штрафы","part_ids":[1]}]}'
        )
        with patch("agents.llm._chat_completion", new=Mock(return_value=raw)):
            result = understand_query([], "налоги", settings=make_settings())

        self.assertEqual(len(result.subqueries), 1)

    def test_parses_clarification_request(self) -> None:
        raw = (
            '{"standalone_query":"гарантия по договору с Альфой",'
            '"in_scope":true,"is_complex":false,'
            '"clarification":{"needed":true,'
            '"question":"Уточните, какая именно компания имеется в виду.",'
            '"missing_slots":["компания"],"options":["Альфа-Трейд","Альфа-Трейдинг"],'
            '"reason":"неоднозначная компания"},'
            '"subqueries":[{"id":1,"text":"гарантия Альфа",'
            '"part_ids":[1],"kind":"fact"}]}'
        )
        with patch("agents.llm._chat_completion", new=Mock(return_value=raw)):
            result = understand_query(
                [],
                "Подскажите гарантию по договору с Альфой.",
                settings=make_settings(),
            )

        self.assertTrue(result.clarification.needed)
        self.assertIn("какая именно компания", result.clarification.question)
        self.assertEqual(len(result.clarification.options), 2)

    def test_builds_clarification_question_from_missing_slots(self) -> None:
        raw = (
            '{"standalone_query":"возврат помещения","in_scope":true,'
            '"is_complex":false,'
            '"clarification":{"needed":true,"question":"",'
            '"missing_slots":["номер договора"]},'
            '"subqueries":[{"id":1,"text":"возврат помещения",'
            '"part_ids":[1],"kind":"fact"}]}'
        )
        with patch("agents.llm._chat_completion", new=Mock(return_value=raw)):
            result = understand_query(
                [],
                "Мы арендуем склад. Планируем съезжать.",
                settings=make_settings(),
            )

        self.assertTrue(result.clarification.needed)
        self.assertIn("номер договора", result.clarification.question)

    def test_falls_back_to_last_question_on_bad_json(self) -> None:
        with patch("agents.llm._chat_completion", new=Mock(return_value="не json")):
            result = understand_query([], "налоги", settings=make_settings())

        self.assertEqual(result.standalone_query, "налоги")
        self.assertEqual(result.subqueries[0].text, "налоги")
        self.assertFalse(result.clarification.needed)


class GradeRetrievalTest(unittest.TestCase):
    def test_parses_subquery_grades_and_computes_overall(self) -> None:
        raw = (
            '{"subqueries":[{"subquery_id":1,"coverage":"partial",'
            '"missing_aspects":["срок"],"reason":"есть норма"}],'
            '"overall":"full","missing_aspects":[]}'
        )
        with patch("agents.llm._chat_completion", new=Mock(return_value=raw)):
            result = grade_retrieval(
                "налоги",
                [make_subquery()],
                [make_context()],
                settings=make_settings(),
            )

        self.assertEqual(result.overall, "partial")
        self.assertEqual(result.subqueries[0].missing_aspects, ["срок"])

    def test_empty_context_does_not_call_llm(self) -> None:
        completion = Mock(return_value="")
        with patch("agents.llm._chat_completion", new=completion):
            result = grade_retrieval(
                "налоги",
                [make_subquery()],
                [],
                settings=make_settings(),
            )

        self.assertEqual(result.overall, "none")
        completion.assert_not_called()

    def test_preserves_contradiction_and_required_missing_fields(self) -> None:
        raw = (
            '{"subqueries":[{"subquery_id":1,"coverage":"contradiction",'
            '"evidence_type":"contradiction",'
            '"missing_required_aspects":[],"missing_optional_details":'
            '["форма уведомления"],'
            '"unsupported_user_assumptions":["сумма 500 000"],'
            '"reason":"контекст указывает другую сумму"}],'
            '"overall":"contradiction","evidence_type":"contradiction",'
            '"missing_required_aspects":[]}'
        )
        with patch("agents.llm._chat_completion", new=Mock(return_value=raw)):
            result = grade_retrieval(
                "Почему сумма 500 000 рублей?",
                [make_subquery()],
                [make_context()],
                settings=make_settings(),
            )

        self.assertEqual(result.overall, "partial")
        self.assertEqual(result.evidence_type, "contradiction")
        self.assertEqual(
            result.subqueries[0].unsupported_user_assumptions,
            ["сумма 500 000"],
        )
        self.assertEqual(
            result.subqueries[0].missing_optional_details,
            ["форма уведомления"],
        )

    def test_missing_subquery_grade_uses_heuristic(self) -> None:
        raw = '{"subqueries":[],"overall":"none","missing_aspects":[]}'
        with patch("agents.llm._chat_completion", new=Mock(return_value=raw)):
            result = grade_retrieval(
                "налоги",
                [make_subquery()],
                [make_context()],
                settings=make_settings(),
            )

        self.assertEqual(result.overall, "partial")


class RewriteSubqueryTest(unittest.TestCase):
    def test_returns_rewritten_query(self) -> None:
        with patch("agents.llm._chat_completion", new=Mock(return_value="новый запрос")):
            result = rewrite_subquery(
                "вопрос",
                make_subquery("старый запрос"),
                "нужно точнее",
                [],
                settings=make_settings(),
            )

        self.assertEqual(result, "новый запрос")

    def test_falls_back_to_current_query(self) -> None:
        with patch("agents.llm._chat_completion", new=Mock(return_value="")):
            result = rewrite_subquery(
                "вопрос",
                make_subquery("старый запрос"),
                "нужно точнее",
                [],
                settings=make_settings(),
            )

        self.assertEqual(result, "старый запрос")

    def test_avoids_near_duplicate_of_tried_query(self) -> None:
        with patch("agents.llm._chat_completion", new=Mock(return_value="старый запрос")):
            result = rewrite_subquery(
                "вопрос",
                make_subquery("старый запрос"),
                "нужно точнее",
                ["старый запрос"],
                settings=make_settings(),
            )

        self.assertEqual(result, "старый запрос")


class GenerateAnswerTest(unittest.TestCase):
    def test_returns_llm_answer_when_available(self) -> None:
        with patch("agents.llm._chat_completion", new=Mock(return_value="Ответ LLM")):
            answer = generate_answer(
                "вопрос",
                [make_context()],
                [],
                settings=make_settings(),
            )

        self.assertEqual(answer, "Ответ LLM")

    def test_returns_no_data_when_llm_empty(self) -> None:
        with patch("agents.llm._chat_completion", new=Mock(return_value="")):
            answer = generate_answer(
                "вопрос",
                [make_context()],
                [],
                settings=make_settings(),
            )

        self.assertIn("нет документов", answer)
        self.assertNotIn("налог", answer)

    def test_empty_context_does_not_call_llm(self) -> None:
        completion = Mock(return_value="Ответ LLM")
        with patch("agents.llm._chat_completion", new=completion):
            answer = generate_answer(
                "вопрос",
                [],
                [],
                settings=make_settings(),
            )

        self.assertIn("нет документов", answer)
        completion.assert_not_called()


class VerifyAnswerTest(unittest.TestCase):
    def test_parses_valid_verification(self) -> None:
        raw = (
            '{"verdict":"pass","grounded":true,"answers_question":true,'
            '"citation_ok":true,"premise_ok":true,"failure_type":"none",'
            '"missing_evidence":[],"fix_hint":"","reason":"ok"}'
        )
        with patch("agents.llm._chat_completion", new=Mock(return_value=raw)):
            result = verify_answer(
                "вопрос",
                "Ответ [doc_id: doc-1]",
                [make_context()],
                settings=make_settings(),
            )

        self.assertEqual(result.verdict, "pass")

    def test_unsupported_core_answer_becomes_refusal(self) -> None:
        raw = (
            '{"verdict":"pass","grounded":false,"answers_question":true,'
            '"citation_ok":true,"premise_ok":true,"failure_type":"generation",'
            '"missing_evidence":[],"fix_hint":"убрать выдумку","reason":"не заземлён"}'
        )
        with patch("agents.llm._chat_completion", new=Mock(return_value=raw)):
            result = verify_answer(
                "вопрос",
                "Ответ",
                [make_context()],
                settings=make_settings(),
            )

        self.assertEqual(result.verdict, "refuse")
        self.assertFalse(result.core_answer_supported)

    def test_bad_citation_becomes_revision_on_fallback(self) -> None:
        with patch("agents.llm._chat_completion", new=Mock(return_value="не json")):
            result = verify_answer(
                "вопрос",
                "Ответ [doc_id: doc-999]",
                [make_context()],
                settings=make_settings(),
            )

        self.assertEqual(result.verdict, "revise_answer")
        self.assertEqual(result.failure_type, "citation")

    def test_empty_context_refuses_without_llm(self) -> None:
        completion = Mock(return_value="")
        with patch("agents.llm._chat_completion", new=completion):
            result = verify_answer(
                "вопрос",
                "Ответ",
                [],
                settings=make_settings(),
            )

        self.assertEqual(result.verdict, "refuse")
        completion.assert_not_called()


class ChatCompletionTest(unittest.TestCase):
    def test_missing_settings_raise_value_error(self) -> None:
        with self.assertRaises(ValueError) as raised:
            _chat_completion(settings=Settings(), messages=[])

        self.assertIn("LLM_BASE_URL", str(raised.exception))
        self.assertIn("LLM_API_KEY", str(raised.exception))
        self.assertIn("LLM_MODEL", str(raised.exception))

    def test_settings_repr_hides_api_key(self) -> None:
        settings = make_settings()
        self.assertNotIn("test-key", repr(settings))


class ChatCompletionLoggingTest(unittest.TestCase):
    def test_logs_prompt_and_raw_response(self) -> None:
        fake_llm = SimpleNamespace(
            invoke=Mock(return_value=SimpleNamespace(content="сырой ответ"))
        )
        with (
            patch("agents.llm._build_llm", return_value=fake_llm),
            self.assertLogs("agents.llm", level="INFO") as captured,
        ):
            result = _chat_completion(
                settings=make_settings(),
                messages=[SystemMessage(content="системный промпт")],
                label="test",
            )

        self.assertEqual(result, "сырой ответ")
        logs = "\n".join(captured.output)
        self.assertIn("[test] LLM запрос", logs)
        self.assertIn("системный промпт", logs)
        self.assertIn("сырой ответ", logs)
        self.assertIn("LLM вызов завершён", logs)

    def test_api_error_is_soft_and_logged(self) -> None:
        fake_llm = SimpleNamespace(
            invoke=Mock(side_effect=RuntimeError("сеть недоступна"))
        )
        with (
            patch("agents.llm._build_llm", return_value=fake_llm),
            self.assertLogs("agents.llm", level="WARNING") as captured,
        ):
            result = _chat_completion(
                settings=make_settings(),
                messages=[SystemMessage(content="промпт")],
            )

        self.assertEqual(result, "")
        self.assertIn("RAG-fallback", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
