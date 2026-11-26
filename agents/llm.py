"""Доступ к LLM через OpenAI-совместимый API.

Провайдер, модель и ключ берутся из настроек (переменные окружения).
В коде нет фиксированного имени модели: на платформе используется то,
что передала платформа, локально — то, что указано в ``.env``.

Незаданные настройки LLM — громкая ошибка конфигурации (ValueError):
проблему видно сразу при старте. Если же запрос упал или модель вернула
пустой ответ, функции мягко деградируют до детерминированных fallback'ов
и пишут предупреждение в лог.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.config import Settings
from agents.context import format_context, format_grouped_context
from agents.dialog import history_records, queries_similar
from agents.prompts import (
    GENERATE_SYSTEM_PROMPT,
    GRADE_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    UNDERSTAND_SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
    generate_user_prompt,
    grade_user_prompt,
    rewrite_user_prompt,
    understand_user_prompt,
    verify_user_prompt,
)
from agents.state import (
    ClarificationRequest,
    ContextChunk,
    DialogTurn,
    QuestionPart,
    RetrievalGradeResult,
    SubQuery,
    SubqueryGrade,
    UnderstandResult,
    VerificationResult,
)
from agents.tokens import TokenUsage, estimate_messages_tokens, extract_usage

__all__ = [
    "clarification_answer",
    "ensure_llm_configured",
    "fallback_grade",
    "fallback_verification",
    "generate_answer",
    "grade_retrieval",
    "no_data_answer",
    "out_of_scope_answer",
    "rewrite_subquery",
    "understand_query",
    "verify_answer",
]

logger = logging.getLogger(__name__)

_LLM_TIMEOUT_SECONDS = 30.0
_LLM_MAX_RETRIES = 1



def _normalize_text(value: object) -> str:
    """Привести содержимое ответа модели к строке."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    parts.append(text.strip())
            elif isinstance(item, str):
                parts.append(item.strip())
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text.strip())
        return " ".join(part for part in parts if part).strip()
    return str(value).strip()


def _missing_settings(settings: Settings) -> list[str]:
    """Перечислить незаданные параметры, без которых LLM не вызвать."""
    missing: list[str] = []
    if not (settings.llm_base_url or "").strip():
        missing.append("LLM_BASE_URL")
    if not (settings.llm_api_key or "").strip():
        missing.append("LLM_API_KEY")
    if not (settings.llm_model or "").strip():
        missing.append("LLM_MODEL")
    return missing


def ensure_llm_configured(settings: Settings) -> None:
    """Проверить настройки LLM; незаданные переменные — громкая ошибка."""
    missing = _missing_settings(settings)
    if missing:
        raise ValueError(
            "LLM не настроена: не заданы "
            + ", ".join(missing)
            + ". Впишите их в .env (шаблон — .env.example, описание — agents/README.md)."
        )


def _build_llm(
    settings: Settings,
    *,
    temperature: float,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """Создать клиент ChatOpenAI по настройкам окружения."""
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "api_key": settings.llm_api_key,
        "base_url": settings.llm_base_url,
        "temperature": temperature,
        "timeout": _LLM_TIMEOUT_SECONDS,
        "max_retries": _LLM_MAX_RETRIES,
    }
    tokens = settings.llm_max_tokens if max_tokens is None else max_tokens
    if tokens is not None:
        kwargs["max_tokens"] = tokens
    return ChatOpenAI(**kwargs)


def _chat_completion(
    *,
    settings: Settings,
    messages: list[BaseMessage],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    label: str = "llm",
    usage_sink: list[TokenUsage] | None = None,
    budget_remaining: int | None = None,
) -> str:
    """Отправить запрос в LLM и вернуть текст ответа.

    Незаданные настройки — ValueError: проблему конфигурации видно сразу.
    Ошибка API, сбой сети или пустой ответ обрабатываются мягко: пустая
    строка, вызывающий код сам выбирает безопасный fallback.

    Промпт и сырой ответ модели логируются на уровне INFO.
    """
    ensure_llm_configured(settings)

    estimated_input = estimate_messages_tokens(messages)
    output_reserve = max_tokens or settings.llm_max_tokens or 0
    if (
        budget_remaining is not None
        and estimated_input + output_reserve > budget_remaining
    ):
        logger.warning(
            "[%s] Пропуск LLM-вызова: бюджет токенов исчерпан "
            "(нужно ~%s, осталось %s).",
            label,
            estimated_input + output_reserve,
            budget_remaining,
        )
        if usage_sink is not None:
            usage_sink.append(TokenUsage(skipped=True))
        return ""

    logger.info(
        "[%s] LLM запрос: model=%s base_url=%s",
        label,
        settings.llm_model,
        settings.llm_base_url,
    )
    for message in messages:
        logger.info("[%s] промпт [%s]:\n%s", label, message.type, message.content)

    try:
        llm = _build_llm(settings, temperature=temperature, max_tokens=max_tokens)
    except Exception as exc:
        logger.warning(
            "Не удалось создать LLM-клиент: %s: %s. Используется RAG-fallback.",
            type(exc).__name__,
            exc,
        )
        return ""

    started_at = time.monotonic()
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        logger.warning(
            "[%s] LLM request failed: %s: %s. Используется RAG-fallback.",
            label,
            type(exc).__name__,
            exc,
        )
        if usage_sink is not None:
            usage_sink.append(
                TokenUsage(
                    input_tokens=estimated_input,
                    total_tokens=estimated_input,
                )
            )
        return ""

    logger.info(
        "[%s] LLM ответ (сырой):\n%s",
        label,
        getattr(response, "content", ""),
    )
    logger.info("[%s] LLM вызов завершён за %.2f c", label, time.monotonic() - started_at)

    content = _normalize_text(getattr(response, "content", ""))
    if usage_sink is not None:
        usage = extract_usage(response, messages, content)
        usage_sink.append(usage)
        logger.info(
            "[%s] токены: input=%s output=%s total=%s",
            label,
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
        )
    if not content:
        logger.warning("LLM вернула пустой ответ. Используется RAG-fallback.")
    return content


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Достать JSON-объект из ответа модели."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        data = json.loads(cleaned[start : end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


def _as_text_list(value: Any) -> list[str]:
    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


def _normalize_clarification_data(data: dict[str, Any]) -> None:
    raw = data.get("clarification")
    clarification = dict(raw) if isinstance(raw, dict) else {}

    flat_fields = {
        "needs_clarification": "needed",
        "clarification_question": "question",
        "clarification_missing_slots": "missing_slots",
        "clarification_options": "options",
        "clarification_reason": "reason",
    }
    for source_key, target_key in flat_fields.items():
        if target_key not in clarification and source_key in data:
            clarification[target_key] = data[source_key]

    if not clarification:
        return

    question = str(clarification.get("question") or "").strip()
    needed = clarification.get("needed")
    if isinstance(needed, str):
        needed = needed.strip().lower() in {"true", "1", "yes", "да"}
    if needed is None and question:
        needed = True
    clarification["needed"] = bool(needed)

    source = str(clarification.get("source") or "understand").strip().lower()
    if source not in {"understand", "retrieval"}:
        source = "understand"
    clarification["source"] = source
    clarification["question"] = question
    clarification["missing_slots"] = _as_text_list(
        clarification.get("missing_slots")
    )
    clarification["options"] = _as_text_list(clarification.get("options"))
    clarification["reason"] = str(clarification.get("reason") or "").strip()
    data["clarification"] = clarification


def _normalize_clarification(
    clarification: ClarificationRequest | None,
) -> ClarificationRequest:
    if clarification is None:
        return ClarificationRequest()

    question = (clarification.question or "").strip()
    missing_slots = list(
        dict.fromkeys(_as_text_list(clarification.missing_slots))
    )
    options = list(dict.fromkeys(_as_text_list(clarification.options)))[:5]

    if not question and missing_slots:
        question = "Уточните, пожалуйста: " + ", ".join(missing_slots) + "."

    if not clarification.needed or not question:
        return ClarificationRequest()

    return ClarificationRequest(
        needed=True,
        question=question,
        missing_slots=missing_slots,
        options=options,
        source=clarification.source,
        reason=(clarification.reason or "").strip(),
    )


def clarification_answer(clarification: ClarificationRequest | None) -> str:
    """Сформировать текст уточняющего вопроса."""
    if clarification is None:
        return "Уточните, пожалуйста, вопрос."
    question = (clarification.question or "").strip()
    return question or "Уточните, пожалуйста, вопрос."


def _clean_rewrite(value: str) -> str:
    """Убрать кавычки и служебные подписи из переформулировки."""
    cleaned = (value or "").strip()
    if not cleaned:
        return ""

    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    first_line = first_line.strip("`\"'«»")
    first_line = re.sub(
        r"^(поисковый запрос|запрос|подзапрос)\s*[:\-]\s*",
        "",
        first_line,
        flags=re.IGNORECASE,
    )
    return first_line.strip()


def no_data_answer(question: str) -> str:
    """Честный отказ, когда в базе знаний нет подтверждённых сведений."""
    return (
        f"В предоставленной базе знаний нет документов, подтверждающих ответ "
        f"на вопрос «{question}». Я не могу отвечать по памяти. "
        "Уточните вопрос или обратитесь к профильному специалисту."
    )


def out_of_scope_answer() -> str:
    """Ответ для вопросов вне тематики базы знаний."""
    return (
        "Этот вопрос выходит за тематику базы знаний ПСБ. "
        "Я отвечаю только на банковские, финансовые, налоговые, трудовые "
        "и связанные с ними юридические вопросы по предоставленным документам."
    )


def _fallback_understand(last_question: str) -> UnderstandResult:
    question = (last_question or "").strip()
    return UnderstandResult(
        standalone_query=question,
        in_scope=True,
        scope_reason="LLM недоступна, вопрос принят как есть.",
        is_complex=False,
        question_parts=[QuestionPart(id=1, text=question, kind="fact")],
        subqueries=[
            SubQuery(
                id=1,
                text=question,
                part_ids=[1],
                kind="fact",
                required=True,
            )
        ],
    )


def _normalize_understand(
    result: UnderstandResult,
    last_question: str,
    max_subqueries: int,
) -> UnderstandResult:
    standalone = (result.standalone_query or last_question or "").strip()

    parts: list[QuestionPart] = []
    for index, part in enumerate(result.question_parts, start=1):
        text = (part.text or "").strip()
        if not text:
            continue
        parts.append(QuestionPart(id=index, text=text, kind=part.kind or "fact"))
    if not parts:
        parts = [QuestionPart(id=1, text=standalone, kind="fact")]

    part_ids = {part.id for part in parts}
    subqueries: list[SubQuery] = []
    for raw_subquery in result.subqueries:
        text = (raw_subquery.text or "").strip()
        if not text:
            continue
        if any(queries_similar(text, existing.text) for existing in subqueries):
            continue
        selected_parts = [
            part_id for part_id in raw_subquery.part_ids if part_id in part_ids
        ]
        subqueries.append(
            SubQuery(
                id=len(subqueries) + 1,
                text=text,
                part_ids=selected_parts or [1],
                kind=raw_subquery.kind or "fact",
                required=raw_subquery.required,
            )
        )
        if len(subqueries) >= max_subqueries:
            break

    if not subqueries:
        subqueries = [
            SubQuery(id=1, text=standalone, part_ids=[1], kind="fact", required=True)
        ]

    if not result.is_complex and len(subqueries) > 1:
        subqueries = subqueries[:1]

    return UnderstandResult(
        standalone_query=standalone,
        in_scope=result.in_scope,
        scope_reason=result.scope_reason,
        is_complex=result.is_complex,
        question_parts=parts,
        subqueries=subqueries,
        clarification=_normalize_clarification(result.clarification),
    )


def understand_query(
    history: list[DialogTurn],
    last_user_question: str,
    *,
    settings: Settings,
    usage_sink: list[TokenUsage] | None = None,
    budget_remaining: int | None = None,
) -> UnderstandResult:
    """Проанализировать вопрос, историю, тематику и декомпозицию."""
    history_data = history_records(history, settings.history_limit)
    messages = [
        SystemMessage(
            content=UNDERSTAND_SYSTEM_PROMPT.format(
                max_subqueries=settings.max_subqueries
            )
        ),
        HumanMessage(
            content=understand_user_prompt(history_data, last_user_question)
        ),
    ]
    raw = _chat_completion(
        settings=settings,
        messages=messages,
        temperature=0.0,
        max_tokens=500,
        label="understand",
        usage_sink=usage_sink,
        budget_remaining=budget_remaining,
    )

    data = _extract_json_object(raw)
    parsed: UnderstandResult | None = None
    if data:
        if "query" in data and "standalone_query" not in data:
            data["standalone_query"] = data.pop("query")
        if "parts" in data and "question_parts" not in data:
            data["question_parts"] = data.pop("parts")
        _normalize_clarification_data(data)
        if "subqueries" in data:
            for item in _as_list(data["subqueries"]):
                if not isinstance(item, dict):
                    continue
                if "query" in item and "text" not in item:
                    item["text"] = item.pop("query")
                if "part_ids" in item:
                    item["part_ids"] = _as_list(item["part_ids"])
        try:
            parsed = UnderstandResult.model_validate(data)
        except Exception as exc:
            logger.warning("LLM вернула анализ вопроса не по схеме: %s", exc)

    if parsed is not None:
        return _normalize_understand(parsed, last_user_question, settings.max_subqueries)

    if not raw:
        skipped = bool(usage_sink) and any(item.skipped for item in usage_sink)
        if not skipped:
            logger.warning("LLM вернула пустой ответ анализа вопроса. Используется fallback.")
    else:
        logger.warning("LLM вернула анализ вопроса не в JSON-формате. Используется fallback.")
    return _fallback_understand(last_user_question)


def fallback_grade(
    subqueries: list[SubQuery],
    contexts: list[ContextChunk],
) -> RetrievalGradeResult:
    grades: list[SubqueryGrade] = []
    for subquery in subqueries:
        has_chunks = any(subquery.id in chunk.subquery_ids for chunk in contexts)
        grades.append(
            SubqueryGrade(
                subquery_id=subquery.id,
                coverage="partial" if has_chunks else "none",
                evidence_type="related" if has_chunks else "none",
                missing_required_aspects=(
                    [] if has_chunks else ["Нет релевантных фрагментов."]
                ),
                reason="Эвристика по наличию фрагментов.",
            )
        )

    overall = "none"
    if grades and all(grade.coverage == "full" for grade in grades):
        overall = "full"
    elif any(grade.coverage in {"full", "partial"} for grade in grades):
        overall = "partial"

    missing = [
        aspect
        for grade in grades
        for aspect in grade.missing_required_aspects
    ]
    return RetrievalGradeResult(
        subqueries=grades,
        overall=overall,
        evidence_type=(
            "direct" if overall == "full" else "related" if overall == "partial" else "none"
        ),
        missing_required_aspects=list(dict.fromkeys(missing)),
    )


def _normalize_grade(
    parsed: RetrievalGradeResult,
    subqueries: list[SubQuery],
    contexts: list[ContextChunk],
) -> RetrievalGradeResult:
    fallback = fallback_grade(subqueries, contexts)
    by_id = {grade.subquery_id: grade for grade in parsed.subqueries}

    normalized: list[SubqueryGrade] = []
    for subquery in subqueries:
        grade = by_id.get(subquery.id)
        if grade is None:
            heuristic = next(
                (item for item in fallback.subqueries if item.subquery_id == subquery.id),
                None,
            )
            if heuristic is not None:
                normalized.append(heuristic)
            continue

        coverage = grade.coverage
        evidence_type = grade.evidence_type
        if coverage not in {"full", "partial", "none"}:
            if coverage == "direct":
                coverage = "full"
                evidence_type = "direct"
            elif coverage == "contradiction":
                coverage = "partial"
                evidence_type = "contradiction"
            elif coverage == "related":
                coverage = "partial"
                evidence_type = "related"
        if coverage not in {"full", "partial", "none"}:
            coverage = "none"
        if evidence_type not in {"direct", "contradiction", "related", "none"}:
            evidence_type = "none"
        if evidence_type == "none" and coverage in {"full", "partial"}:
            evidence_type = "direct" if coverage == "full" else "related"
        required = _as_text_list(
            grade.missing_required_aspects or grade.missing_aspects
        )
        normalized.append(
            SubqueryGrade(
                subquery_id=subquery.id,
                coverage=coverage,
                evidence_type=evidence_type,
                missing_aspects=required,
                missing_required_aspects=required,
                missing_optional_details=_as_text_list(
                    grade.missing_optional_details
                ),
                unsupported_user_assumptions=_as_text_list(
                    grade.unsupported_user_assumptions
                ),
                reason=grade.reason,
            )
        )

    overall = "none"
    if normalized and all(grade.coverage == "full" for grade in normalized):
        overall = "full"
    elif any(grade.coverage in {"full", "partial"} for grade in normalized):
        overall = "partial"

    missing = [
        aspect
        for grade in normalized
        for aspect in grade.missing_required_aspects
    ]
    optional = [
        aspect
        for grade in normalized
        for aspect in grade.missing_optional_details
    ]
    assumptions = [
        assumption
        for grade in normalized
        for assumption in grade.unsupported_user_assumptions
    ]
    evidence_types = {grade.evidence_type for grade in normalized}
    overall_evidence = (
        "contradiction"
        if "contradiction" in evidence_types
        else "direct"
        if overall == "full"
        else "related"
        if overall == "partial"
        else "none"
    )
    return RetrievalGradeResult(
        subqueries=normalized,
        overall=overall,
        evidence_type=overall_evidence,
        missing_required_aspects=list(dict.fromkeys(missing)),
        missing_optional_details=list(dict.fromkeys(optional)),
        unsupported_user_assumptions=list(dict.fromkeys(assumptions)),
    )


def grade_retrieval(
    question: str,
    subqueries: list[SubQuery],
    contexts: list[ContextChunk],
    *,
    settings: Settings,
    usage_sink: list[TokenUsage] | None = None,
    budget_remaining: int | None = None,
) -> RetrievalGradeResult:
    """Оценить, насколько найденный контекст покрывает подзапросы."""
    if not subqueries:
        subqueries = [
            SubQuery(id=1, text=question, part_ids=[1], kind="fact", required=True)
        ]
    if not contexts:
        return fallback_grade(subqueries, contexts)

    context_text = format_grouped_context(
        subqueries,
        contexts,
    )
    subqueries_data = [
        {"id": subquery.id, "text": subquery.text, "kind": subquery.kind}
        for subquery in subqueries
    ]
    messages = [
        SystemMessage(content=GRADE_SYSTEM_PROMPT),
        HumanMessage(content=grade_user_prompt(question, subqueries_data, context_text)),
    ]
    raw = _chat_completion(
        settings=settings,
        messages=messages,
        temperature=0.0,
        max_tokens=400,
        label="grade",
        usage_sink=usage_sink,
        budget_remaining=budget_remaining,
    )

    data = _extract_json_object(raw)
    if data:
        if "results" in data and "subqueries" not in data:
            data["subqueries"] = data.pop("results")
        for item in _as_list(data.get("subqueries")):
            if isinstance(item, dict):
                if "id" in item and "subquery_id" not in item:
                    item["subquery_id"] = item.pop("id")
                if "missing_aspects" in item:
                    item["missing_aspects"] = _as_text_list(item["missing_aspects"])
                for field in (
                    "missing_required_aspects",
                    "missing_optional_details",
                    "unsupported_user_assumptions",
                ):
                    if field in item:
                        item[field] = _as_text_list(item[field])
                coverage = item.get("coverage")
                if isinstance(coverage, str):
                    coverage = coverage.strip().lower()
                    if coverage in {"direct", "contradiction", "related"}:
                        item.setdefault("evidence_type", coverage)
                        coverage = (
                            "full" if coverage == "direct" else "partial"
                        )
                    item["coverage"] = coverage
                if item.get("coverage") not in {"full", "partial", "none"}:
                    item["coverage"] = "none"
                evidence_type = item.get("evidence_type")
                if isinstance(evidence_type, str):
                    item["evidence_type"] = evidence_type.strip().lower()
                if item.get("evidence_type") not in {
                    "direct",
                    "contradiction",
                    "related",
                    "none",
                }:
                    item["evidence_type"] = "none"
        if "missing_aspects" in data:
            data["missing_aspects"] = _as_text_list(data["missing_aspects"])
        for field in (
            "missing_required_aspects",
            "missing_optional_details",
            "unsupported_user_assumptions",
        ):
            if field in data:
                data[field] = _as_text_list(data[field])
        overall = data.get("overall")
        if isinstance(overall, str):
            overall = overall.strip().lower()
            if overall in {"direct", "contradiction", "related"}:
                data.setdefault("evidence_type", overall)
                data["overall"] = "full" if overall == "direct" else "partial"
        if data.get("overall") not in {None, "full", "partial", "none"}:
            data["overall"] = "none"
        evidence_type = data.get("evidence_type")
        if isinstance(evidence_type, str):
            data["evidence_type"] = evidence_type.strip().lower()
        if data.get("evidence_type") not in {
            "direct",
            "contradiction",
            "related",
            "none",
        }:
            data["evidence_type"] = "none"
        try:
            parsed = RetrievalGradeResult.model_validate(data)
        except Exception as exc:
            logger.warning("LLM вернула оценку retrieval не по схеме: %s", exc)
            parsed = None
    else:
        parsed = None

    if parsed is None:
        if raw:
            logger.warning("LLM вернула оценку retrieval не в JSON-формате. Используется эвристика.")
        return fallback_grade(subqueries, contexts)

    return _normalize_grade(parsed, subqueries, contexts)


def rewrite_subquery(
    question: str,
    subquery: SubQuery,
    reason: str,
    tried_queries: list[str],
    *,
    settings: Settings,
    usage_sink: list[TokenUsage] | None = None,
    budget_remaining: int | None = None,
) -> str:
    """Переформулировать один подзапрос без потери сущностей."""
    base = (subquery.text or "").strip()
    if not base:
        return ""

    messages = [
        SystemMessage(content=REWRITE_SYSTEM_PROMPT),
        HumanMessage(
            content=rewrite_user_prompt(
                question,
                base,
                reason,
                tried_queries[-8:],
            )
        ),
    ]
    raw = _chat_completion(
        settings=settings,
        messages=messages,
        temperature=0.1,
        max_tokens=128,
        label="rewrite",
        usage_sink=usage_sink,
        budget_remaining=budget_remaining,
    )

    cleaned = _clean_rewrite(raw)
    if not cleaned:
        return base
    if any(queries_similar(cleaned, query, threshold=0.95) for query in tried_queries):
        logger.info("Переформулировка почти совпала с уже использованным запросом.")
        return base
    return cleaned


def generate_answer(
    question: str,
    contexts: list[ContextChunk],
    history: list[DialogTurn],
    *,
    settings: Settings,
    usage_sink: list[TokenUsage] | None = None,
    budget_remaining: int | None = None,
) -> str:
    """Сгенерировать ответ строго по RAG-контексту."""
    context_text = format_context(contexts)
    if not context_text:
        return no_data_answer(question)

    history_data = history_records(history, settings.history_limit)
    messages = [
        SystemMessage(content=GENERATE_SYSTEM_PROMPT),
        HumanMessage(
            content=generate_user_prompt(
                question,
                history_data,
                context_text,
            )
        ),
    ]
    usage_start = len(usage_sink) if usage_sink is not None else 0
    answer = _chat_completion(
        settings=settings,
        messages=messages,
        temperature=0.0,
        max_tokens=600,
        label="generate",
        usage_sink=usage_sink,
        budget_remaining=budget_remaining,
    )
    if (
        usage_sink is not None
        and len(usage_sink) > usage_start
        and usage_sink[usage_start].skipped
    ):
        return no_data_answer(question)
    return answer or no_data_answer(question)


def _known_sources(contexts: list[ContextChunk]) -> set[str]:
    known: set[str] = set()
    for chunk in contexts:
        chunk_ids = chunk.metadata.get("chunk_ids")
        values: list[object] = [
            chunk.doc_id,
            chunk.chunk_id,
            chunk.metadata.get("title"),
            chunk.metadata.get("head_doc"),
            chunk.metadata.get("document_title"),
        ]
        if isinstance(chunk_ids, (list, tuple)):
            values.extend(chunk_ids)
        for value in values:
            if value:
                known.add(str(value).strip())
    return known


def _citations_valid(answer: str, contexts: list[ContextChunk]) -> bool:
    known = _known_sources(contexts)
    if not known:
        return False

    citations = re.findall(
        r"(?:doc_id|chunk_id)\s*[:=]\s*([^\s,;\]\)]+)",
        answer,
        flags=re.IGNORECASE,
    )
    for citation in citations:
        cleaned = citation.strip("`\"'«»[](){}.,;:")
        if cleaned and cleaned not in known:
            return False
    return True


def fallback_verification(
    answer: str,
    contexts: list[ContextChunk],
) -> VerificationResult:
    if not (answer or "").strip():
        return VerificationResult(
            verdict="refuse",
            grounded=False,
            answers_question=False,
            citation_ok=False,
            core_answer_supported=False,
            core_answer_relevant=False,
            premise_corrected=False,
            failure_type="policy",
            reason="Пустой ответ.",
        )

    if not _citations_valid(answer, contexts):
        return VerificationResult(
            verdict="revise_answer",
            grounded=True,
            answers_question=True,
            citation_ok=False,
            core_answer_supported=True,
            core_answer_relevant=True,
            premise_corrected=True,
            failure_type="citation",
            fix_hint="Исправь ссылки на источники по doc_id из контекста.",
            reason="Ссылки не соответствуют контексту.",
        )

    return VerificationResult(
        verdict="pass",
        grounded=True,
        answers_question=True,
        citation_ok=True,
        premise_ok=True,
        core_answer_supported=True,
        core_answer_relevant=True,
        premise_corrected=True,
        failure_type="none",
        reason="LLM недоступна, использована эвристика.",
    )


def verify_answer(
    question: str,
    answer: str,
    contexts: list[ContextChunk],
    *,
    settings: Settings,
    usage_sink: list[TokenUsage] | None = None,
    budget_remaining: int | None = None,
) -> VerificationResult:
    """Проверить заземлённость, полноту и корректность ссылок."""
    context_text = format_context(contexts)
    if not context_text:
        return VerificationResult(
            verdict="refuse",
            grounded=False,
            answers_question=False,
            citation_ok=False,
            core_answer_supported=False,
            core_answer_relevant=False,
            premise_corrected=False,
            failure_type="retrieval",
            missing_evidence=["В базе знаний нет подтверждённых фрагментов."],
            reason="Контекст пуст.",
        )

    messages = [
        SystemMessage(content=VERIFY_SYSTEM_PROMPT),
        HumanMessage(
            content=verify_user_prompt(
                question,
                answer,
                context_text,
            )
        ),
    ]
    raw = _chat_completion(
        settings=settings,
        messages=messages,
        temperature=0.0,
        max_tokens=300,
        label="verify",
        usage_sink=usage_sink,
        budget_remaining=budget_remaining,
    )

    data = _extract_json_object(raw)
    if data:
        for key in ("missing_evidence",):
            if key in data:
                data[key] = _as_text_list(data[key])
        if "optional_details_missing" in data:
            data["optional_details_missing"] = _as_text_list(
                data["optional_details_missing"]
            )
        for field, fallback in (
            ("core_answer_supported", "grounded"),
            ("core_answer_relevant", "answers_question"),
            ("premise_corrected", "premise_ok"),
        ):
            if field not in data and fallback in data:
                data[field] = bool(data[fallback])
        verdict = data.get("verdict")
        if isinstance(verdict, str):
            verdict = verdict.strip().lower()
            data["verdict"] = verdict
        if data.get("verdict") not in {"pass", "revise_answer", "re_retrieve", "refuse"}:
            data["verdict"] = "revise_answer"
        failure_type = data.get("failure_type")
        if isinstance(failure_type, str):
            failure_type = failure_type.strip().lower()
            data["failure_type"] = failure_type
        if data.get("failure_type") not in {
            "none",
            "retrieval",
            "generation",
            "citation",
            "premise",
            "policy",
        }:
            data["failure_type"] = "generation"
        try:
            parsed = VerificationResult.model_validate(data)
        except Exception as exc:
            logger.warning("LLM вернула проверку ответа не по схеме: %s", exc)
            parsed = None
    else:
        parsed = None

    if parsed is None:
        if raw:
            logger.warning("LLM вернула проверку ответа не в JSON-формате. Используется эвристика.")
        return fallback_verification(answer, contexts)

    if parsed.verdict == "pass" and not (
        parsed.grounded
        and parsed.answers_question
        and parsed.citation_ok
        and parsed.premise_ok
        and parsed.core_answer_relevant
        and parsed.premise_corrected
    ):
        if not parsed.core_answer_supported:
            parsed = parsed.model_copy(update={"verdict": "refuse"})
        elif parsed.failure_type == "retrieval":
            parsed = parsed.model_copy(update={"verdict": "re_retrieve"})
        else:
            parsed = parsed.model_copy(update={"verdict": "revise_answer"})

    if parsed.core_answer_supported and parsed.verdict == "refuse":
        parsed = parsed.model_copy(update={"verdict": "revise_answer"})

    return parsed
