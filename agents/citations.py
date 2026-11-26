"""Постобработка ссылок на источники в финальном ответе.

Модель цитирует внутренние ``doc_id`` / ``chunk_id``. Этот модуль заменяет
их на человекочитаемые названия.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from agents.context import document_title, source_label
from agents.state import ContextChunk

__all__ = [
    "SourceRef",
    "build_source_index",
    "humanize_citations",
]

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(
    r"\[(?P<kind>doc_id|chunk_id)\s*[:=]\s*(?P<ids>[^\]]+)\]",
    re.IGNORECASE,
)
_SOURCE_CITATION_RE = re.compile(
    r"\[source\s*[:=]\s*(?P<labels>[^\]]+)\]",
    re.IGNORECASE,
)
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_ID_SPLIT_RE = re.compile(r"[,;]")
_ID_CLEAN_CHARS = " \t\r\n`\"'«»[](){}.,;:"


@dataclass(frozen=True)
class SourceRef:
    """Источник, на который ссылается финальный ответ."""

    label: str
    title: str = ""


def _clean_id(value: str) -> str:
    return value.strip().strip(_ID_CLEAN_CHARS)


def build_source_index(
    contexts: list[ContextChunk],
) -> dict[str, SourceRef]:
    """Сопоставить технические идентификаторы с человекочитаемым источником."""
    index: dict[str, SourceRef] = {}
    for chunk in contexts:
        label = source_label(chunk)
        if not label:
            continue
        ref = SourceRef(label=label, title=document_title(chunk) or label)
        values: list[object] = [
            chunk.doc_id,
            chunk.chunk_id,
            chunk.metadata.get("title"),
            chunk.metadata.get("head_doc"),
            chunk.metadata.get("document_title"),
            label,
        ]
        chunk_ids = chunk.metadata.get("chunk_ids")
        if isinstance(chunk_ids, (list, tuple)):
            values.extend(chunk_ids)

        for value in values:
            key = str(value).strip() if value is not None else ""
            if key and key not in index:
                index[key] = ref
    return index


def humanize_citations(
    answer: str,
    contexts: list[ContextChunk],
) -> tuple[str, list[SourceRef]]:
    """Заменить ``[doc_id: ...]`` и ``[chunk_id: ...]`` на названия источников.

    Возвращает обновлённый текст и список источников в порядке первого
    упоминания. Неизвестные идентификаторы остаются без изменений.
    """
    if not answer:
        return answer, []

    index = build_source_index(contexts)
    ordered: list[SourceRef] = []
    seen: set[str] = set()

    def collect(ref: SourceRef) -> None:
        if ref.label not in seen:
            seen.add(ref.label)
            ordered.append(ref)

    def replace(match: re.Match[str]) -> str:
        labels: list[str] = []
        unknown: list[str] = []
        for raw in _ID_SPLIT_RE.split(match.group("ids")):
            key = _clean_id(raw)
            if not key:
                continue
            ref = index.get(key)
            if ref is None:
                unknown.append(key)
                continue
            if ref.label not in labels:
                labels.append(ref.label)
            collect(ref)

        if unknown:
            logger.warning(
                "Не удалось сопоставить источники из ответа: %s",
                ", ".join(unknown),
            )
        if not labels:
            return match.group(0)

        result = "[" + "; ".join(labels) + "]"
        if unknown:
            result += " [doc_id: " + ", ".join(unknown) + "]"
        return result

    def replace_source(match: re.Match[str]) -> str:
        parts = [
            _clean_id(raw)
            for raw in match.group("labels").split(";")
            if _clean_id(raw)
        ]
        refs = [index.get(part) for part in parts]
        if not parts or any(ref is None for ref in refs):
            return match.group(0)
        labels: list[str] = []
        for ref in refs:
            assert ref is not None
            if ref.label not in labels:
                labels.append(ref.label)
            collect(ref)
        return "[" + "; ".join(labels) + "]"

    result = _CITATION_RE.sub(replace, answer)
    result = _SOURCE_CITATION_RE.sub(replace_source, result)

    for bracket in _BRACKET_RE.finditer(result):
        for raw in bracket.group(1).split(";"):
            ref = index.get(_clean_id(raw))
            if ref is not None:
                collect(ref)

    return result, ordered
