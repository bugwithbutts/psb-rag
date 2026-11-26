from __future__ import annotations

import unittest

from agents.context import (
    format_context,
    merge_contexts,
    metadata_hints,
    normalize_chunk,
    source_label,
)
from agents.state import ContextChunk
from rag.index.contracts import RetrievedChunk


class NormalizeChunkTest(unittest.TestCase):
    def test_converts_retrieved_chunk(self) -> None:
        chunk = normalize_chunk(
            RetrievedChunk(
                doc_id="doc-1",
                chunk_id="chunk-1",
                text="Текст",
                metadata={"title": "Договор", "info": "Описание раздела"},
                score=0.5,
            ),
            2,
        )

        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.doc_id, "doc-1")
        self.assertEqual(chunk.chunk_id, "chunk-1")
        self.assertEqual(chunk.metadata["title"], "Договор")
        self.assertEqual(chunk.metadata["info"], "Описание раздела")
        self.assertEqual(chunk.metadata["document_title"], "Договор")
        self.assertIsNone(chunk.metadata["doc_number"])
        self.assertEqual(chunk.subquery_ids, [2])

    def test_chunk_id_falls_back_to_doc_id(self) -> None:
        chunk = normalize_chunk(
            RetrievedChunk(
                doc_id="doc-1",
                chunk_id="",
                text="Текст",
                metadata={},
                score=0.1,
            ),
            1,
        )

        self.assertEqual(chunk.chunk_id, "doc-1")

    def test_discards_empty_text(self) -> None:
        chunk = normalize_chunk(
            RetrievedChunk(
                doc_id="doc-1",
                chunk_id="chunk-1",
                text="   \n\t",
                metadata={"head_doc": "Договор № 1"},
                score=0.5,
            ),
            1,
        )

        self.assertIsNone(chunk)


class MetadataParsingTest(unittest.TestCase):
    def test_parses_head_doc_fields(self) -> None:
        chunk = normalize_chunk(
            RetrievedChunk(
                doc_id="doc-1",
                chunk_id="chunk-1",
                text="Текст",
                metadata={
                    "head_doc": (
                        "Договор № 123/2024-АБ от 01.03.2024 "
                        "— 1 августа 2024 г."
                    ),
                    "chunk_index": "0",
                    "char_start": 0,
                    "char_end": 10,
                },
                score=0.5,
            ),
            1,
        )

        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.metadata["document_title"], (
            "Договор № 123/2024-АБ от 01.03.2024 — 1 августа 2024 г."
        ))
        self.assertEqual(chunk.metadata["doc_number"], "123/2024-АБ")
        self.assertEqual(chunk.metadata["document_date"], "01.08.2024")
        self.assertEqual(chunk.metadata["chunk_index"], 0)
        self.assertEqual(chunk.metadata["char_start"], 0)
        self.assertEqual(chunk.metadata["char_end"], 10)

    def test_addendum_uses_parent_contract_number(self) -> None:
        chunk = normalize_chunk(
            RetrievedChunk(
                doc_id="addendum",
                chunk_id="chunk-1",
                text="Текст",
                metadata={
                    "head_doc": (
                        "Дополнительное соглашение № 2 к Договору "
                        "№ 123/2024-АБ от 01.03.2024"
                    )
                },
                score=0.5,
            ),
            1,
        )

        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.metadata["doc_number"], "123/2024-АБ")
        self.assertEqual(chunk.metadata["document_date"], "01.03.2024")

    def test_earliest_status_marker_wins(self) -> None:
        active_first = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="Текст",
            metadata={"head_doc": "Договор действует, старая редакция утратила силу"},
            subquery_ids=[1],
        )
        invalid_first = ContextChunk(
            doc_id="doc-2",
            chunk_id="chunk-2",
            text="Текст",
            metadata={"head_doc": "Договор утратил силу, новый документ действует"},
            subquery_ids=[1],
        )

        self.assertEqual(metadata_hints(active_first)["status"], "действует")
        self.assertEqual(metadata_hints(invalid_first)["status"], "утратил силу")

    def test_numeric_edition_and_date_normalized(self) -> None:
        chunk = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="Текст",
            metadata={"head_doc": "Кодекс, редакция от 01.02.2024"},
            subquery_ids=[1],
        )

        hints = metadata_hints(chunk)

        self.assertEqual(hints["edition"], "01.02.2024")
        self.assertEqual(hints["document_date"], "01.02.2024")


class MergeContextsTest(unittest.TestCase):
    def test_deduplicates_by_chunk_id_and_merges_subqueries(self) -> None:
        first = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="Текст",
            subquery_ids=[1],
        )
        second = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="Текст",
            subquery_ids=[2],
        )

        merged = merge_contexts([first, second], max_chunks=10)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].subquery_ids, [1, 2])

    def test_preserves_all_chunks_despite_legacy_max_chunks(self) -> None:
        chunks = [
            ContextChunk(doc_id=f"doc-{index}", chunk_id=f"chunk-{index}", text="Т")
            for index in range(5)
        ]

        merged = merge_contexts(chunks, max_chunks=2)

        self.assertEqual(
            [chunk.doc_id for chunk in merged],
            ["doc-0", "doc-1", "doc-2", "doc-3", "doc-4"],
        )


class SourceLabelTest(unittest.TestCase):
    def test_law_label(self) -> None:
        chunk = ContextChunk(
            doc_id="законы__152ФЗ_ст9_действующая.txt",
            chunk_id="law-1",
            text="Текст",
            metadata={
                "head_doc": (
                    "Статья 9. Согласие субъекта персональных данных. "
                    'Федеральный закон от 27.07.2006 N 152-ФЗ "О персональных данных"'
                )
            },
            subquery_ids=[1],
        )

        self.assertEqual(source_label(chunk), "152-ФЗ, ст. 9")

    def test_contract_label_from_head(self) -> None:
        chunk = ContextChunk(
            doc_id="документы компании__Договор аренды А-08-2024 Северный Склад.txt",
            chunk_id="contract-1",
            text="Текст",
            metadata={
                "head_doc": (
                    "ДОГОВОР АРЕНДЫ НЕЖИЛОГО ПОМЕЩЕНИЯ № А-08/2024 "
                    "Московская область, г. Химки — 1 августа 2024 г."
                )
            },
            subquery_ids=[1],
        )

        self.assertEqual(source_label(chunk), "Договор аренды № А-08/2024")

    def test_addendum_label_from_head(self) -> None:
        chunk = ContextChunk(
            doc_id="документы компании__Доп соглашение 1 к договору П-17-2024.txt",
            chunk_id="addendum-1",
            text="Текст",
            metadata={
                "head_doc": (
                    "ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ № 1 к договору поставки "
                    "№ П-17/2024 от 12 февраля 2024 г."
                )
            },
            subquery_ids=[1],
        )

        self.assertEqual(
            source_label(chunk),
            "Доп. соглашение № 1 к договору № П-17/2024",
        )

    def test_regulation_label_from_doc_id(self) -> None:
        chunk = ContextChunk(
            doc_id=(
                "документы компании__Регламент согласования договоров "
                "ред 2023 (утратил силу).txt"
            ),
            chunk_id="reg-1",
            text="Текст",
            metadata={},
            subquery_ids=[1],
        )

        self.assertEqual(
            source_label(chunk),
            "Регламент согласования договоров (ред. 2023, утратил силу)",
        )

    def test_falls_back_to_document_title(self) -> None:
        chunk = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="Текст",
            metadata={"title": "Налоговый кодекс"},
            subquery_ids=[1],
        )

        self.assertEqual(source_label(chunk), "Налоговый кодекс")

    def test_uses_stored_source_label(self) -> None:
        chunk = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="Текст",
            metadata={"source_label": "ТК РФ, ст. 70"},
            subquery_ids=[1],
        )

        self.assertEqual(source_label(chunk), "ТК РФ, ст. 70")


class FormatContextTest(unittest.TestCase):
    def test_includes_sources_and_metadata(self) -> None:
        chunk = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="Текст нормы",
            metadata={"title": "Налоговый кодекс", "edition": "2024"},
            score=0.9,
            subquery_ids=[1],
        )

        formatted = format_context([chunk])

        self.assertIn("title: Налоговый кодекс", formatted)
        self.assertIn("doc_id: doc-1", formatted)
        self.assertIn("source_label: Налоговый кодекс", formatted)
        self.assertIn("edition: 2024", formatted)
        self.assertIn("Текст нормы", formatted)

    def test_includes_context_info_without_repeating_same_value(self) -> None:
        chunks = [
            ContextChunk(
                doc_id="doc-1",
                chunk_id="chunk-1",
                text="Первый пункт",
                metadata={"info": "Описание списка"},
                subquery_ids=[1],
            ),
            ContextChunk(
                doc_id="doc-1",
                chunk_id="chunk-2",
                text="Второй пункт",
                metadata={"info": "Описание списка"},
                subquery_ids=[1],
            ),
        ]

        formatted = format_context(chunks)

        self.assertIn("[Контекст фрагмента: Описание списка]", formatted)
        self.assertEqual(formatted.count("[Контекст фрагмента: Описание списка]"), 1)
        self.assertIn("Первый пункт", formatted)
        self.assertIn("Второй пункт", formatted)

    def test_ignores_legacy_max_chars_without_losing_text(self) -> None:
        chunk = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="Т" * 200,
            metadata={},
            subquery_ids=[1],
        )

        formatted = format_context([chunk], max_chars=80)

        self.assertIn("Т" * 200, formatted)

    def test_preserves_contiguous_chunks_and_order(self) -> None:
        first = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="0123456789",
            metadata={"chunk_index": 0, "char_start": 0, "char_end": 10},
            subquery_ids=[1],
        )
        second = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-2",
            text="56789abcde",
            metadata={"chunk_index": 1, "char_start": 5, "char_end": 15},
            subquery_ids=[1],
        )

        formatted = format_context([first, second])

        self.assertIn("0123456789", formatted)
        self.assertIn("56789abcde", formatted)
        self.assertLess(formatted.index("chunk-1"), formatted.index("chunk-2"))

    def test_keeps_disjoint_chunks_separate_without_loss_marker(self) -> None:
        first = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="abcde",
            metadata={"chunk_index": 0, "char_start": 0, "char_end": 5},
            subquery_ids=[1],
        )
        second = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-2",
            text="fghij",
            metadata={"chunk_index": 1, "char_start": 10, "char_end": 15},
            subquery_ids=[1],
        )

        formatted = format_context([first, second])

        self.assertIn("abcde", formatted)
        self.assertIn("fghij", formatted)
        self.assertNotIn("[...]", formatted)
        self.assertIn("chunk_id: chunk-1", formatted)
        self.assertIn("chunk_id: chunk-2", formatted)

    def test_does_not_merge_across_block_offset_reset(self) -> None:
        first = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-0",
            text="А" * 100,
            metadata={"chunk_index": 0, "char_start": 0, "char_end": 100},
            subquery_ids=[1],
        )
        second = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-2",
            text="Б" * 100,
            metadata={"chunk_index": 2, "char_start": 0, "char_end": 100},
            subquery_ids=[1],
        )

        formatted = format_context([first, second])

        self.assertNotIn("[...]", formatted)

    def test_metadata_hints_extract_status_and_edition(self) -> None:
        chunk = ContextChunk(
            doc_id="doc-1",
            chunk_id="chunk-1",
            text="Текст",
            metadata={"title": "Договор утратил силу, редакция 2024"},
            subquery_ids=[1],
        )

        hints = metadata_hints(chunk)

        self.assertEqual(hints["status"], "утратил силу")
        self.assertEqual(hints["edition"], "2024")


if __name__ == "__main__":
    unittest.main()
