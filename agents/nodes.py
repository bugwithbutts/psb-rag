"""Узлы агентского графа.


Узлы — обычные функции: получают состояние и возвращают только изменённые
поля. Вся тяжёлая работа делегируется RAG и функциям из ``llm.py``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from agents.clarification import (
    detect_context_ambiguity,
    last_assistant_asked_question,
)
from agents.citations import humanize_citations
from agents.config import Settings
from agents.context import merge_contexts, normalize_chunk
from agents.dialog import parse_dialog, queries_similar
from agents.llm import (
    clarification_answer,
    fallback_grade,
    fallback_verification,
    generate_answer,
    grade_retrieval,
    no_data_answer,
    out_of_scope_answer,
    rewrite_subquery,
    understand_query,
    verify_answer,
)
from agents.state import AgentState, SubQuery, VerificationResult
from agents.tokens import TokenUsage, sum_usage
from rag.rag_pipeline import RAG

__all__ = [
    "make_clarify_node",
    "make_finalize_node",
    "make_generate_node",
    "make_grade_node",
    "make_parse_dialog_node",
    "make_refuse_node",
    "make_retrieve_node",
    "make_rewrite_node",
    "make_understand_node",
    "make_verify_node",
    "route_after_grade",
    "route_after_understand",
    "route_after_verify",
]

logger = logging.getLogger(__name__)

Node = Callable[[AgentState], dict[str, Any]]


def _question(state: AgentState) -> str:
    return (
        state.get("standalone_query")
        or state.get("last_user_question")
        or state.get("query")
        or ""
    ).strip()


def _has_searchable_anchor(question: str) -> bool:
    """Проверить, есть ли в вопросе опора для пробного поиска."""
    text = (question or "").strip()
    if not text.endswith("?"):
        return False
    return bool(
        re.search(
            r"(?:ООО|ОАО|ЗАО|ИП)\s*[«\"']?[\wЁА-я-]+"
            r"|«[^»]+»|"
            r"(?:договор|контракт|соглашен\w*|склад|помещен\w*)",
            text,
            re.IGNORECASE,
        )
    )


def _budget_remaining(state: AgentState, settings: Settings) -> int:
    """Сколько токенов ещё можно потратить на LLM в этом вопросе."""
    used = state.get("used_tokens", 0)
    return max(0, settings.max_total_tokens_per_question - used)


def _add_usage(state: AgentState, usage: list[TokenUsage]) -> int:
    """Вернуть обновлённый суммарный расход токенов."""
    return state.get("used_tokens", 0) + sum_usage(usage)


def _subqueries(state: AgentState) -> list[SubQuery]:
    subqueries = list(state.get("subqueries") or [])
    if subqueries:
        return subqueries

    question = _question(state)
    if not question:
        return []
    return [SubQuery(id=1, text=question, part_ids=[1], kind="fact", required=True)]


def make_parse_dialog_node() -> Node:
    """Создать узел детерминированного разбора истории диалога."""

    def parse_dialog_node(state: AgentState) -> dict[str, Any]:
        question = (state.get("query") or "").strip()
        history, last_question = parse_dialog(question)
        return {
            "history": history,
            "last_user_question": last_question,
            "query": last_question,
        }

    return parse_dialog_node


def make_understand_node(settings: Settings) -> Node:
    """Создать узел анализа вопроса и декомпозиции."""

    def understand(state: AgentState) -> dict[str, Any]:
        last_question = (
            state.get("last_user_question")
            or state.get("query")
            or ""
        ).strip()
        usage: list[TokenUsage] = []
        result = understand_query(
            state.get("history") or [],
            last_question,
            settings=settings,
            usage_sink=usage,
            budget_remaining=_budget_remaining(state, settings),
        )
        subqueries = list(result.subqueries)
        clarification = result.clarification
        if clarification.needed and last_assistant_asked_question(
            state.get("history") or []
        ):
            clarification = clarification.model_copy(
                update={"needed": False, "question": ""}
            )
        step_increment = 1 if any(not item.skipped for item in usage) else 0
        return {
            "query": last_question,
            "standalone_query": result.standalone_query,
            "in_scope": result.in_scope,
            "clarification": clarification,
            "subqueries": subqueries,
            "contexts": [],
            "retrieval_attempts": 0,
            "answer_attempts": 0,
            "total_steps": state.get("total_steps", 0) + step_increment,
            "used_tokens": _add_usage(state, usage),
        }

    return understand


def _interleave_chunks(groups: list[list[Any]]) -> list[Any]:
    """Смешать результаты запросов по рангам, сохраняя порядок выдачи."""
    result: list[Any] = []
    seen: set[str] = set()
    max_len = max((len(group) for group in groups), default=0)
    for rank in range(max_len):
        for group in groups:
            if rank >= len(group):
                continue
            chunk = group[rank]
            key = (
                getattr(chunk, "chunk_id", "")
                or getattr(chunk, "doc_id", "")
                or str(id(chunk))
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(chunk)
    return result


def make_retrieve_node(rag: RAG) -> Node:
    """Создать узел поиска в RAG по всем подзапросам."""

    def retrieve(state: AgentState) -> dict[str, Any]:
        subqueries = _subqueries(state)
        previous_contexts = list(state.get("contexts") or [])
        per_subquery: list[list[Any]] = []
        updated_subqueries: list[SubQuery] = []

        for subquery in subqueries:
            new_groups: list[list[Any]] = []
            tried_queries = list(subquery.tried_queries)
            attempts = subquery.attempts
            query = (subquery.text or "").strip()
            if query and query not in tried_queries:
                search_failed = False
                try:
                    chunks = rag.search_chunks(query)
                except Exception as exc:
                    logger.warning(
                        "RAG поиск упал для подзапроса %s: %s: %s",
                        subquery.id,
                        type(exc).__name__,
                        exc,
                    )
                    chunks = []
                    search_failed = True

                logger.info(
                    "RAG поиск: подзапрос=%s, запрос=%r, найдено=%s, chunk_ids=%s",
                    subquery.id,
                    query,
                    len(chunks or []),
                    [
                        getattr(chunk, "chunk_id", "")
                        for chunk in (chunks or [])
                    ],
                )
                query_chunks: list[Any] = []
                for chunk in chunks or []:
                    normalized = normalize_chunk(chunk, subquery.id)
                    if normalized is None:
                        logger.debug(
                            "Пропущен пустой RAG-чанк: doc_id=%r, chunk_id=%r",
                            getattr(chunk, "doc_id", None),
                            getattr(chunk, "chunk_id", None),
                        )
                        continue
                    query_chunks.append(normalized)
                if query_chunks:
                    new_groups.append(query_chunks)
                if not search_failed:
                    tried_queries.append(query)
                    attempts += 1

            old_chunks = [
                chunk
                for chunk in previous_contexts
                if subquery.id in chunk.subquery_ids
            ]
            per_subquery.append(_interleave_chunks([old_chunks, *new_groups]))
            updated_subqueries.append(
                subquery.model_copy(
                    update={"tried_queries": tried_queries, "attempts": attempts}
                )
            )

        collected: list[Any] = []
        for subquery_chunks in per_subquery:
            collected.extend(subquery_chunks)
        collected.extend(
            chunk for chunk in previous_contexts if not chunk.subquery_ids
        )

        merged_contexts = merge_contexts(collected)
        logger.info(
            "Контекст агента: подзапросы=%s, chunk_ids=%s",
            [subquery.id for subquery in updated_subqueries],
            [chunk.chunk_id for chunk in merged_contexts],
        )

        return {
            "subqueries": updated_subqueries,
            # RAG/reranker уже определили состав и порядок выдачи. Агент
            # только дедуплицирует одинаковые chunk_id и ничего не отбрасывает.
            "contexts": merged_contexts,
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

    return retrieve


def make_grade_node(settings: Settings) -> Node:
    """Создать узел оценки покрытия контекста."""

    def grade(state: AgentState) -> dict[str, Any]:
        question = _question(state)
        subqueries = _subqueries(state)
        contexts = list(state.get("contexts") or [])
        total_steps = state.get("total_steps", 0)
        usage: list[TokenUsage] = []

        if contexts and total_steps < settings.max_total_llm_calls:
            result = grade_retrieval(
                question,
                subqueries,
                contexts,
                settings=settings,
                usage_sink=usage,
                budget_remaining=_budget_remaining(state, settings),
            )
            step_increment = 1
        else:
            result = fallback_grade(subqueries, contexts)
            step_increment = 0

        by_id = {item.subquery_id: item for item in result.subqueries}
        updated: list[SubQuery] = []
        for subquery in subqueries:
            item = by_id.get(subquery.id)
            if item is None:
                updated.append(subquery)
                continue
            updated.append(
                subquery.model_copy(
                    update={
                        "coverage": item.coverage,
                        "evidence_type": item.evidence_type,
                        "missing_aspects": item.missing_required_aspects,
                        "missing_required_aspects": item.missing_required_aspects,
                        "missing_optional_details": item.missing_optional_details,
                        "unsupported_user_assumptions": (
                            item.unsupported_user_assumptions
                        ),
                    }
                )
            )

        if result.overall == "none":
            clarification = state.get("clarification")
        else:
            clarification = detect_context_ambiguity(
                question,
                state.get("history") or [],
                contexts,
            )

        return {
            "subqueries": updated,
            "coverage": result.overall,
            "missing_required_aspects": result.missing_required_aspects,
            "clarification": clarification,
            "total_steps": total_steps + step_increment,
            "used_tokens": _add_usage(state, usage),
        }

    return grade


def make_rewrite_node(settings: Settings) -> Node:
    """Создать узел переформулировки проблемных подзапросов."""

    def rewrite(state: AgentState) -> dict[str, Any]:
        question = _question(state)
        verification = state.get("verification")
        subqueries = [
            subquery.model_copy(deep=True)
            for subquery in _subqueries(state)
        ]
        target_ids = {
            subquery.id
            for subquery in subqueries
            if subquery.coverage == "none" or subquery.coverage is None
        }

        if verification is not None and verification.missing_evidence:
            next_id = max((subquery.id for subquery in subqueries), default=0) + 1
            for evidence in verification.missing_evidence:
                evidence = (evidence or "").strip()
                if not evidence:
                    continue
                if len(subqueries) >= settings.max_subqueries:
                    break
                if any(
                    queries_similar(evidence, subquery.text)
                    for subquery in subqueries
                ):
                    continue
                subqueries.append(
                    SubQuery(
                        id=next_id,
                        text=evidence,
                        part_ids=[],
                        kind="fact",
                        required=True,
                    )
                )
                target_ids.add(next_id)
                next_id += 1

        available = settings.max_total_llm_calls - state.get("total_steps", 0)
        rewrite_limit = min(available, settings.max_rewrite_subqueries_per_round)
        rewrite_calls = 0
        usage: list[TokenUsage] = []
        updated: list[SubQuery] = []

        for subquery in subqueries:
            if (
                subquery.id not in target_ids
                or subquery.attempts >= settings.max_retrieval_attempts
                or rewrite_calls >= rewrite_limit
            ):
                updated.append(subquery)
                continue

            reason = "; ".join(subquery.missing_aspects)
            if not reason and verification is not None:
                reason = verification.fix_hint or verification.reason
            if not reason and subquery.coverage == "none":
                reason = (
                    "Нет релевантных чанков с ответом на подзапрос. "
                    "Попробуйте синонимы и другую формулировку."
                )
            if not reason:
                reason = "Текущий контекст не покрывает подзапрос."

            tried_queries = list(subquery.tried_queries)
            new_text = rewrite_subquery(
                question,
                subquery,
                reason,
                list(tried_queries),
                settings=settings,
                usage_sink=usage,
                budget_remaining=max(
                    0,
                    settings.max_total_tokens_per_question
                    - state.get("used_tokens", 0)
                    - sum_usage(usage),
                ),
            )
            rewrite_calls += 1
            new_text = (new_text or "").strip() or subquery.text

            old_text = (subquery.text or "").strip()
            if old_text and old_text not in tried_queries:
                tried_queries.append(old_text)

            if new_text != subquery.text:
                updated.append(
                    subquery.model_copy(
                        update={
                            "text": new_text,
                            "coverage": None,
                            "missing_aspects": [],
                            "tried_queries": tried_queries,
                        }
                    )
                )
            else:
                updated.append(
                    subquery.model_copy(
                        update={
                            "tried_queries": tried_queries,
                            "attempts": subquery.attempts + 1,
                        }
                    )
                )

        return {
            "subqueries": updated,
            "contexts": list(state.get("contexts") or []),
            "total_steps": state.get("total_steps", 0) + rewrite_calls,
            "used_tokens": _add_usage(state, usage),
        }

    return rewrite


def make_generate_node(settings: Settings) -> Node:
    """Создать узел генерации черновика ответа."""

    def generate(state: AgentState) -> dict[str, Any]:
        question = _question(state)
        contexts = list(state.get("contexts") or [])
        answer_attempts = state.get("answer_attempts", 0) + 1
        usage: list[TokenUsage] = []

        if not contexts:
            return {
                "answer_draft": no_data_answer(question),
                "answer_attempts": answer_attempts,
                "used_tokens": _add_usage(state, usage),
            }

        answer = generate_answer(
            question,
            contexts,
            state.get("history") or [],
            settings=settings,
            usage_sink=usage,
            budget_remaining=_budget_remaining(state, settings),
        )
        return {
            "answer_draft": answer,
            "answer_attempts": answer_attempts,
            "total_steps": state.get("total_steps", 0) + 1,
            "used_tokens": _add_usage(state, usage),
        }

    return generate


def make_verify_node(settings: Settings) -> Node:
    """Создать узел проверки черновика ответа."""

    def verify(state: AgentState) -> dict[str, Any]:
        question = _question(state)
        draft = (state.get("answer_draft") or "").strip()
        contexts = list(state.get("contexts") or [])

        if not draft:
            return {
                "verification": VerificationResult(
                    verdict="refuse",
                    grounded=False,
                    answers_question=False,
                    citation_ok=False,
                    failure_type="policy",
                    reason="Пустой черновик ответа.",
                )
            }

        total_steps = state.get("total_steps", 0)
        usage: list[TokenUsage] = []
        if not contexts:
            result = verify_answer(
                question,
                draft,
                contexts,
                settings=settings,
                usage_sink=usage,
                budget_remaining=_budget_remaining(state, settings),
            )
            step_increment = 0
        elif total_steps >= settings.max_total_llm_calls:
            result = fallback_verification(draft, contexts)
            step_increment = 0
        else:
            result = verify_answer(
                question,
                draft,
                contexts,
                settings=settings,
                usage_sink=usage,
                budget_remaining=_budget_remaining(state, settings),
            )
            step_increment = 1

        return {
            "verification": result,
            "total_steps": total_steps + step_increment,
            "used_tokens": _add_usage(state, usage),
        }

    return verify


def make_clarify_node() -> Node:
    """Создать узел уточняющего вопроса."""

    def clarify(state: AgentState) -> dict[str, Any]:
        answer = clarification_answer(state.get("clarification"))
        return {
            "answer": answer,
            "status": "clarification",
        }

    return clarify


def make_finalize_node() -> Node:
    """Создать узел финального ответа."""

    def finalize(state: AgentState) -> dict[str, Any]:
        question = _question(state)
        coverage = state.get("coverage") or "none"
        verification = state.get("verification")
        draft = (state.get("answer_draft") or "").strip()

        if not state.get("in_scope", True):
            answer = out_of_scope_answer()
            status = "refused"
        elif (
            verification is not None
            and verification.verdict == "refuse"
            and not verification.core_answer_supported
        ):
            answer = no_data_answer(question)
            status = "refused"
        elif not draft:
            answer = no_data_answer(question)
            status = "refused"
        else:
            answer = draft
            status = "answered"
            if (
                coverage == "partial"
                or (
                    verification is not None
                    and (
                        verification.verdict != "pass"
                        or verification.optional_details_missing
                    )
                )
            ):
                status = "partial"

        if status in {"answered", "partial"}:
            answer, _ = humanize_citations(
                answer,
                list(state.get("contexts") or []),
            )

        logger.info(
            "Итог по вопросу: status=%s, LLM-вызовов=%s, токенов≈%s",
            status,
            state.get("total_steps", 0),
            state.get("used_tokens", 0),
        )
        return {
            "answer": answer,
            "status": status,
        }

    return finalize


def make_refuse_node() -> Node:
    """Создать узел отказа для вопросов вне тематики."""

    def refuse(state: AgentState) -> dict[str, Any]:
        answer = out_of_scope_answer()
        return {
            "answer": answer,
            "status": "refused",
        }

    return refuse


def route_after_understand(state: AgentState) -> str:
    """Маршрут после анализа вопроса."""
    if not state.get("in_scope", True):
        return "refuse"
    clarification = state.get("clarification")
    if (
        clarification is not None
        and clarification.needed
        and not _has_searchable_anchor(_question(state))
    ):
        return "clarify"
    return "retrieve"


def route_after_grade(state: AgentState, settings: Settings) -> str:
    """Маршрут после оценки покрытия контекста."""
    coverage = state.get("coverage") or "none"
    total_steps = state.get("total_steps", 0)
    retrieval_attempts = state.get("retrieval_attempts", 0)

    clarification = state.get("clarification")
    if clarification is not None and clarification.needed:
        return "clarify"

    if coverage == "full":
        return "generate" if total_steps < settings.max_total_llm_calls else "finalize"

    if state.get("used_tokens", 0) >= settings.max_total_tokens_soft:
        if coverage == "partial" and total_steps < settings.max_total_llm_calls:
            return "generate"
        return "finalize"

    if coverage == "none" and not state.get("subqueries") and not state.get("contexts"):
        return "finalize"

    can_retry = (
        retrieval_attempts < settings.max_retrieval_attempts
        and total_steps < settings.max_total_llm_calls
    )
    can_rewrite = any(
        subquery.coverage == "none"
        and subquery.attempts < settings.max_retrieval_attempts
        for subquery in state.get("subqueries", [])
    )
    if coverage == "none" and can_retry and can_rewrite:
        return "rewrite"
    if (
        coverage == "none"
        and retrieval_attempts >= settings.max_retrieval_attempts
        and (state.get("contexts") or not state.get("subqueries"))
    ):
        return "finalize"
    if coverage in {"partial", "none"} and total_steps < settings.max_total_llm_calls:
        return "generate"
    return "finalize"


def route_after_verify(state: AgentState, settings: Settings) -> str:
    """Повторить генерацию, если основной ответ не подтверждён."""
    verification = state.get("verification")
    if (
        verification is not None
        and not verification.core_answer_supported
        and state.get("contexts")
        and state.get("answer_attempts", 0) < settings.max_answer_attempts
        and state.get("total_steps", 0) < settings.max_total_llm_calls
    ):
        return "generate"
    return "finalize"

