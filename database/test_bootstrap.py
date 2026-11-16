"""Интеграционные проверки PostgreSQL в отдельной временной схеме.

Запуск: python -m unittest database.test_bootstrap -v
Используется DATABASE_URL; существующая таблица chunks не затрагивается.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from database import bootstrap
from database.connection import engine
from database.create_tables import ChunkModel


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        schema = "test_bootstrap_" + uuid4().hex
        with engine.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        self.addCleanup(self.drop_schema, schema)
        self.test_engine = engine.execution_options(schema_translate_map={None: schema})
        self.sessions = sessionmaker(bind=self.test_engine, expire_on_commit=False)
        for target in (
            "database.bootstrap.engine",
            "database.create_tables.engine",
        ):
            patcher = patch(target, self.test_engine)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch("database.import_csv.SessionLocal", self.sessions)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.frame = pd.DataFrame([
            {"title": "doc", "chunk": f"text {i}", "metadata": "info", "head": "head"}
            for i in range(3)
        ])
        self.embedder = Mock()
        self.embedder.encode_passages.side_effect = lambda texts: np.full((len(texts), 384), 0.1)
        self.config = SimpleNamespace(embedder=self.embedder, batch_size=2)

    def drop_schema(self, schema):
        with engine.begin() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')

    def count(self):
        with self.sessions() as session:
            return session.scalar(select(func.count()).select_from(ChunkModel))

    def run_bootstrap(self):
        with (
            patch("rag.config.RAGConfig", return_value=self.config),
            patch("database.import_csv.load_csv", return_value=self.frame),
        ):
            bootstrap.main()

    def test_first_start_imports_and_second_start_skips(self):
        self.run_bootstrap()
        self.assertEqual(self.count(), 3)
        with patch("database.import_csv.import_chunks") as importer:
            bootstrap.main()
        importer.assert_not_called()
        self.assertEqual(self.count(), 3)

    def test_failed_import_rolls_back_and_retry_succeeds(self):
        self.embedder.encode_passages.side_effect = [
            np.full((2, 384), 0.1), RuntimeError("embedding failed"),
        ]
        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
            self.run_bootstrap()
        self.assertEqual(self.count(), 0)
        self.embedder.encode_passages.side_effect = lambda texts: np.full((len(texts), 384), 0.1)
        self.run_bootstrap()
        self.assertEqual(self.count(), 3)

    def test_empty_csv_does_not_allow_startup(self):
        self.frame = self.frame.iloc[:0]
        with self.assertRaisesRegex(RuntimeError, "осталась пустой"):
            self.run_bootstrap()


if __name__ == "__main__":
    unittest.main()
