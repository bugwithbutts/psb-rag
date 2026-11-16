import argparse

import pandas as pd
from sqlalchemy import func, select

from rag.config import RAGConfig
from database.connection import SessionLocal
from database.create_tables import ChunkModel


def load_csv(path: str) -> pd.DataFrame:
    """Загрузка и проверка CSV."""
    df = pd.read_csv(path, keep_default_na=False)

    required_columns = {"chunk", "metadata", "title", "head"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Отсутствуют столбцы: {missing}")

    df = df[df["chunk"].astype(str).str.strip() != ""].copy()
    df = df.reset_index(drop=True)

    return df
def import_chunks(
    csv_path: str,
    config: RAGConfig,
) -> None:
    """Вычисление эмбеддингов и загрузка чанков в PostgreSQL."""

    df = load_csv(csv_path)

    print(f"Найдено чанков: {len(df)}")

    if df.empty:
        print("Нет чанков для загрузки")
        return

    # Проверяем, что таблица пустая
    with SessionLocal() as session:
        count = session.scalar(
            select(func.count()).select_from(ChunkModel)
        )

        if count:
            raise RuntimeError(
                f"В базе уже есть {count} чанков. "
                "Повторная загрузка отменена."
            )

    embedder = config.embedder
    batch_size = config.batch_size

    # Весь импорт атомарен: при сбое следующий запуск повторит его целиком.
    with SessionLocal.begin() as session:

        for start in range(0, len(df), batch_size):

            # 1. Получаем очередной батч
            batch = df.iloc[start:start + batch_size]

            # 2. Формируем список текстов
            texts = [
                str(row.chunk)
                for row in batch.itertuples(index=False)
            ]

            # 3. Вычисляем эмбеддинги
            embeddings = embedder.encode_passages(texts)

            # 4. Проверяем размерность
            if embeddings.ndim != 2 or embeddings.shape != (len(texts), 384):
                raise ValueError(
                    f"Неверная размерность эмбеддингов: {embeddings.shape}. "
                    f"Ожидается ({len(texts)}, 384)"
                )

            # 5. Создаём объекты для БД
            chunks = []

            for row, embedding in zip(
                batch.itertuples(index=False),
                embeddings,
                strict=True,
            ):
                chunks.append(
                    ChunkModel(
                        doc_id=str(row.title),
                        text=str(row.chunk),
                        metadata_={
                            "info": str(row.metadata),
                            "head_doc": str(row.head),
                        },
                        embedding=embedding.tolist(),
                    )
                )

            # 6. Сохраняем батч
            session.add_all(chunks)
            session.flush()

            print(
                f"Подготовлено: "
                f"{min(start + batch_size, len(df))}/{len(df)}"
            )

    print("Загрузка завершена!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="Путь к CSV с чанками",
    )

    args = parser.parse_args()

    config = RAGConfig()

    import_chunks(
        csv_path=args.csv,
        config=config,
    )
