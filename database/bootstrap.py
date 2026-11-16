"""Подготовить PostgreSQL и один раз импортировать исходный корпус."""

from pathlib import Path

from sqlalchemy import func, select, text

from database.connection import engine
from database.create_tables import ChunkModel, main as create_tables


def main():
    # Общая блокировка для одновременного старта нескольких backend.
    # Транзакционная блокировка автоматически снимается и при ошибке.
    with engine.begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(7348201601)"))
        create_tables()
        count = connection.scalar(select(func.count()).select_from(ChunkModel))
        if count:
            print(f"В БД уже есть {count} чанков, импорт пропущен", flush=True)
            return

        # Модели нужны только для первоначального импорта.
        from database.import_csv import import_chunks
        from rag.config import RAGConfig

        model_path = Path("/app/models/multilingual-e5-small")
        if not (model_path / "config.json").is_file():
            model_path = Path("/opt/models/multilingual-e5-small")
        config = RAGConfig(embedding_model=str(model_path), use_reranker=False)
        print("БД пуста, загружаем исходный корпус CSV", flush=True)
        import_chunks("/app/dataset/result_chunks.csv", config)
        count = connection.scalar(select(func.count()).select_from(ChunkModel))
        if not count:
            raise RuntimeError("После импорта таблица chunks осталась пустой")


if __name__ == "__main__":
    main()
