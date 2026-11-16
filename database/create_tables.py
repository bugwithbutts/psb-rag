from sqlalchemy import BigInteger, Integer, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import VECTOR
from database.connection import engine

class Base(DeclarativeBase):
    pass


class ChunkModel(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    doc_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    embedding: Mapped[list[float]] = mapped_column(
        VECTOR(384),
        nullable=False,
    )



def main():
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(bind=connection)
    print("Таблицы успешно созданы")


if __name__ == "__main__":
    main()
