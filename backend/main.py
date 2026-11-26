"""FastAPI-бэкенд чат-бота «Юридический ассистент ПСБ».

Диалог живёт в памяти процесса: фронт присылает только новый вопрос и
``conversation_id``, а бэкенд сам собирает историю в строку для агента.
Ответ отдаётся потоком SSE: ``meta`` -> ``token*`` -> ``done`` -> ``[DONE]``.

Запуск из каталога ``backend``::

    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

# .env грузим до импорта rag/agents: они читают настройки на уровне модулей.
load_dotenv(_REPO_ROOT / ".env", override=False)

# Логи не нужны: глушим всё, включая uvicorn и print-логи библиотек.
logging.disable(logging.CRITICAL)

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException, Response  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from agents.service import AgentService  # noqa: E402
from rag.config import RAGConfig  # noqa: E402
from rag.rag_pipeline import RAG  # noqa: E402

DOCS_PATH = _REPO_ROOT / "dataset" / "result_chunks.csv"
# Модель эмбеддингов: сначала смонтированная с хоста, затем запасная из образа.
MODEL_PATH = _REPO_ROOT / "models" / "multilingual-e5-small"
BAKED_MODEL_PATH = Path("/opt/models/multilingual-e5-small")

MAX_TURNS = 8              # реплик в запросе к агенту
MAX_CHARS = 6000           # суммарная длина истории
IDLE_TTL = 30 * 60         # секунд без активности до удаления
CLOSE_GRACE = 15           # секунд между /close и удалением
WARMUP_TIMEOUT = 240       # сколько ждать готовности индекса в /api/chat

ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")
ROLE = {"user": "Пользователь", "assistant": "Ассистент"}
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# Метка роли в начале строки: двоеточие заменяем, чтобы пользователь не мог
# подделать реплику ассистента через перевод строки.
_ROLE_LINE = re.compile(
    r"(?im)^(\s*)(Пользователь|Ассистент|Система|User|Assistant|System)(\s*):"
)
_SOURCES_RE = re.compile(r"(?im)^\s*Источники\s*:")


@dataclass
class Conversation:
    turns: list[dict[str, str]] = field(default_factory=list)
    closing_at: float | None = None
    touched_at: float = field(default_factory=time.monotonic)
    last_seen_ms: int = 0


AGENT: AgentService | None = None
_READY = asyncio.Event()
_WARMUP_ERROR: str | None = None
_WARMUP_LOCK = threading.Lock()
_HISTORY: dict[str, Conversation] = {}
_CHAT_LOCKS: dict[str, asyncio.Lock] = {}


# ---------------------------------------------------------------------------
# Старт приложения
# ---------------------------------------------------------------------------


def _resolve_model_path() -> Path:
    """Выбрать каталог модели эмбеддингов.

    На хосте ``./models`` может быть пустым: Compose всё равно монтирует его
    в ``/app/models`` (только для чтения), поэтому проверяем сначала монтирование,
    а затем копию, загруженную в образ при сборке.
    """
    for candidate in (MODEL_PATH, BAKED_MODEL_PATH):
        if (candidate / "config.json").is_file():
            return candidate
    # Ничего не нашли — вернуть ожидаемый путь, чтобы ошибка была понятной.
    return MODEL_PATH


def warm_up() -> None:
    """Прочитать корпус, собрать локальный индекс и создать агента."""
    global AGENT, _WARMUP_ERROR
    if AGENT is not None:
        return

    with _WARMUP_LOCK:
        if AGENT is not None:
            return

        rag = RAG(
            RAGConfig(
                embedding_type="local",
                embedding_model=str(_resolve_model_path()),
                use_reranker=False,
            )
        )
        rag.build(pd.read_csv(DOCS_PATH))
        AGENT = AgentService(rag)
        _WARMUP_ERROR = None


async def prepare() -> None:
    """Построить индекс в фоне, не блокируя открытие порта.

    uvicorn открывает сокет только после ``lifespan.startup()``, поэтому
    долгая инициализация в ``lifespan`` приводила к 502 от nginx. Здесь
    сервер начинает принимать запросы сразу, а готовность отмечается
    событием ``_READY``.
    """
    global _WARMUP_ERROR
    try:
        await asyncio.to_thread(warm_up)
    except Exception as exc:  # noqa: BLE001 — показываем причину в /api/health
        _WARMUP_ERROR = f"{type(exc).__name__}: {exc}"
    finally:
        _READY.set()


@asynccontextmanager
async def lifespan(_: FastAPI):
    warmup_task = asyncio.create_task(prepare())
    sweeper_task = asyncio.create_task(sweeper())
    try:
        yield
    finally:
        sweeper_task.cancel()
        warmup_task.cancel()


app = FastAPI(title="PSB Legal Assistant API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# История диалога
# ---------------------------------------------------------------------------


def neutralize(text: str) -> str:
    return _ROLE_LINE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{match.group(3)}\u2236",
        text,
    )


def _for_prompt(turn: dict[str, str]) -> str:
    """Ответ ассистента без финального блока «Источники:»."""
    if turn["role"] != "assistant":
        return turn["content"]
    match = _SOURCES_RE.search(turn["content"])
    return turn["content"][: match.start()].rstrip() if match else turn["content"]


def render_dialog(turns: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{ROLE[turn['role']]}: {neutralize(_for_prompt(turn))}"
        for turn in turns
        if turn["content"].strip()
    )


def trim(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    """Оставить последние реплики в пределах лимитов."""
    turns = list(turns)[-MAX_TURNS:]
    total = sum(len(turn["content"]) for turn in turns)
    while len(turns) > 1 and total > MAX_CHARS:
        total -= len(turns.pop(0)["content"])
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    while turns and turns[-1]["role"] != "user":
        turns.pop()
    return turns


def _conversation(cid: str) -> Conversation:
    return _HISTORY.setdefault(cid, Conversation())


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _touch(conversation: Conversation, client_ts: int | None) -> None:
    """Отметить активность и отменить отложенное удаление."""
    stamp = client_ts if client_ts and client_ts > 0 else _now_ms()
    conversation.last_seen_ms = max(conversation.last_seen_ms, stamp)
    conversation.closing_at = None
    conversation.touched_at = time.monotonic()


def _expired(now: float) -> list[str]:
    return [
        cid
        for cid, conversation in _HISTORY.items()
        if now - conversation.touched_at > IDLE_TTL
        or (
            conversation.closing_at is not None
            and now - conversation.closing_at > CLOSE_GRACE
        )
    ]


async def sweeper() -> None:
    while True:
        await asyncio.sleep(10)
        for cid in _expired(time.monotonic()):
            _HISTORY.pop(cid, None)
            _CHAT_LOCKS.pop(cid, None)


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def chunk_text(text: str, size: int = 40) -> list[str]:
    """Нарезать готовый ответ на куски по словам — эффект печати."""
    chunks: list[str] = []
    buffer = ""
    for token in re.findall(r"\S+\s*", text):
        buffer += token
        if len(buffer) >= size:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    ts: int | None = None


async def _stream(
    cid: str, query: str, ts: int | None = None
) -> AsyncGenerator[str, None]:
    started = time.monotonic()
    yield sse({"type": "meta", "conversationId": cid})

    # Индекс может ещё строиться: сообщаем об этом и ждём готовности,
    # вместо того чтобы отдавать nginx 502 из-за незапущенного порта.
    if AGENT is None:
        if _WARMUP_ERROR is not None:
            yield sse(
                {
                    "type": "error",
                    "message": f"Модель не инициализирована: {_WARMUP_ERROR}",
                }
            )
            yield sse({"type": "done", "answer": "", "latencyMs": _elapsed_ms(started)})
            yield "data: [DONE]\n\n"
            return
        yield sse(
            {
                "type": "status",
                "message": "Инициализация модели, это займёт около 1–2 минут…",
            }
        )
        try:
            await asyncio.wait_for(_READY.wait(), timeout=WARMUP_TIMEOUT)
        except asyncio.TimeoutError:
            yield sse(
                {
                    "type": "error",
                    "message": "Модель не успела инициализироваться. "
                    "Попробуйте ещё раз через минуту.",
                }
            )
            yield sse({"type": "done", "answer": "", "latencyMs": _elapsed_ms(started)})
            yield "data: [DONE]\n\n"
            return
        if AGENT is None:
            detail = f": {_WARMUP_ERROR}" if _WARMUP_ERROR else ""
            yield sse(
                {"type": "error", "message": f"Модель не инициализирована{detail}"}
            )
            yield sse({"type": "done", "answer": "", "latencyMs": _elapsed_ms(started)})
            yield "data: [DONE]\n\n"
            return

    async with _CHAT_LOCKS.setdefault(cid, asyncio.Lock()):
        conversation = _conversation(cid)
        _touch(conversation, ts)

        user_turn = {"role": "user", "content": query}
        if not conversation.turns or conversation.turns[-1] != user_turn:
            conversation.turns.append(user_turn)

        try:
            if AGENT is None:
                raise RuntimeError("Агент не инициализирован")
            result = await AGENT.aanswer(render_dialog(trim(conversation.turns)))
            answer = (result.answer or "").strip() or "Не удалось сформировать ответ."
        except Exception as exc:  # noqa: BLE001 — показываем причину во фронте
            yield sse({"type": "error", "message": str(exc)})
            yield sse({"type": "done", "answer": "", "latencyMs": _elapsed_ms(started)})
            yield "data: [DONE]\n\n"
            return

        conversation.turns.append({"role": "assistant", "content": answer})
        conversation.touched_at = time.monotonic()

        for piece in chunk_text(answer):
            yield sse({"type": "token", "text": piece})
            await asyncio.sleep(0)

        yield sse({"type": "done", "answer": answer, "latencyMs": _elapsed_ms(started)})
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> dict:
    return {"service": "PSB Legal Assistant API", "docs": "/docs"}


@app.get("/api/health")
def health() -> dict:
    if AGENT is not None:
        return {"status": "ok", "agent": True}
    if _WARMUP_ERROR is not None:
        return {"status": "error", "agent": False, "detail": _WARMUP_ERROR}
    return {"status": "starting", "agent": False}


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Пустой вопрос")

    cid = (req.conversation_id or "").strip() or uuid.uuid4().hex
    if not ID_RE.match(cid):
        raise HTTPException(status_code=422, detail="Некорректный conversation_id")

    return StreamingResponse(
        _stream(cid, query, req.ts),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.get("/api/conversations/{cid}")
def get_conversation(cid: str, ts: int | None = None) -> dict:
    conversation = _HISTORY.get(cid)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Диалог не найден")
    _touch(conversation, ts)
    return {"conversationId": cid, "turns": conversation.turns}


@app.delete("/api/conversations/{cid}", status_code=204)
def delete_conversation(cid: str) -> Response:
    _HISTORY.pop(cid, None)
    _CHAT_LOCKS.pop(cid, None)
    return Response(status_code=204)


@app.post("/api/conversations/{cid}/close")
def close_conversation(cid: str, ts: int | None = None) -> dict:
    conversation = _HISTORY.get(cid)
    if conversation is None:
        return {"status": "ok"}
    # Beacon после перезагрузки старше последней активности — игнорируем.
    if ts is not None and ts < conversation.last_seen_ms:
        return {"status": "stale"}
    conversation.closing_at = time.monotonic()
    return {"status": "ok"}
