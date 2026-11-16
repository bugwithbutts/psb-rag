# Юридический ассистент ПСБ — чат-бот

Браузерный чат по внутренней базе знаний: вопрос уходит в FastAPI, ответ
приходит потоком SSE и печатается по мере поступления. Диалог помнит контекст
внутри вкладки, переживает F5 и удаляется вместе с вкладкой.

## Быстрый старт

Нужен Docker Compose. Перед запуском заполните `.env`: `POSTGRES_PASSWORD`
и настройки чат-модели `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.

Из корня репозитория запустите одну команду:

```bash
docker compose up -d --build
```

Backend дождётся PostgreSQL, создаст расширение `vector` и таблицу `chunks`.
Если таблица пустая, скрипт `database.bootstrap` автоматически вычислит
эмбеддинги и загрузит `dataset/result_chunks.csv`, после чего запустит API
и RAG. Если данные уже есть, импорт будет пропущен. Затем запустится frontend.

Первый запуск занимает больше времени из-за сборки образа и импорта.
Импорт выполняется одной транзакцией: при ошибке записи откатываются,
и следующий запуск может повторить загрузку. Одновременные старты backend
согласуются блокировкой PostgreSQL, чтобы не импортировать корпус дважды.
При ошибке инициализации API не запускается; причина видна в логах:

```bash
docker compose logs -f backend
```

Данные сохраняются в томе `postgres_data`; повторный запуск той же команды
использует существующую БД. Автоматический импорт не дополняет заполненную
таблицу при изменении CSV. `docker compose down -v` удаляет том с данными.
При `VECTOR_STORE=faiss` подготовка БД пропускается.

- чат — http://localhost:8080
- API — http://localhost:8000/api/health
- документация — http://localhost:8000/docs

Сборка образов, установка зависимостей и сборка фронтенда происходят внутри
этой команды. Значения `LLM_*` Compose берёт из `.env` рядом с
`docker-compose.yml`; сам файл в образ не копируется. Модель эмбеддингов
монтируется из `./models`, а не копируется в образ.

Проверка готовности:

```bash
curl http://localhost:8080/api/health
# {"status":"ok","agent":true}
```

Эмбеддинги считаются локально (`models/multilingual-e5-small`), поэтому
embedding-эндпоинт и его ключ не нужны — только чат-модель.

## Что умеет

- потоковый ответ (`meta` → `token*` → `done`), остановка генерации, копирование;
- отдельный диалог на каждую вкладку: `conversation_id` в `sessionStorage`;
- восстановление истории после перезагрузки страницы;
- удаление диалога при закрытии вкладки (beacon + 15-секундная задержка);
- очистка диалога кнопкой «Очистить» (удаляет историю на сервере и выдаёт новый id);
- локальные эмбеддинги без внешнего embedding-эндпоинта.

## Архитектура

```
psb-rag/
├── frontend/                 # React + TypeScript + Vite
│   ├── src/App.tsx           # интерфейс и состояние чата
│   ├── src/api.ts            # SSE, история, id вкладки
│   ├── Dockerfile            # сборка и раздача через nginx
│   └── nginx.conf            # прокси /api → backend без буферизации
├── backend/                  # FastAPI: HTTP-слой и память диалогов
│   ├── main.py               # /api/chat, /api/conversations/*, warm_up()
│   ├── test_main.py          # pytest: поток и история
│   └── Dockerfile
├── rag/                      # поиск: BM25, FAISS, гибрид, реранкер
├── agents/                   # LangGraph-агент: разбор диалога, генерация
├── models/multilingual-e5-small # локальная модель эмбеддингов
├── dataset/result_chunks.csv # готовый корпус чанков (680 строк, 21 документ)
├── docker-compose.yml
└── .env.example              # только имена переменных, без значений
```

Поток запроса:

1. фронтенд отправляет только новое сообщение и `conversation_id`;
2. бэкенд достаёт историю из памяти процесса, обрезает её (8 реплик / 6000
   символов) и склеивает в строку с метками `Пользователь:` / `Ассистент:`;
3. агент детерминированно разбирает строку (`agents/dialog.py`), ищет чанки
   в RAG и генерирует ответ;
4. бэкенд сохраняет ответ в историю и отдаёт его потоком SSE.

## API

| Метод | Путь | Назначение |
| --- | --- | --- |
| `GET` | `/` | описание сервиса |
| `GET` | `/api/health` | статус, режим поиска, число диалогов |
| `POST` | `/api/chat` | поток SSE: `meta` → `token*` → `done` |
| `GET` | `/api/conversations/{id}` | история диалога |
| `DELETE` | `/api/conversations/{id}` | удалить диалог |
| `POST` | `/api/conversations/{id}/close` | сигнал «вкладка закрылась» |

```bash
curl -N http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Какая арендная плата по договору аренды?","conversation_id":"tab-1"}'
```

Подробнее о событиях и жизненном цикле диалога — `backend/README.md`.

## Переменные окружения

Шаблон — `.env.example` (только имена). Локальный `.env` в корне:

```env
LLM_BASE_URL=https://api.siliconflow.com/v1
LLM_API_KEY=...
LLM_MODEL=deepseek-ai/DeepSeek-V3
```

- `LLM_*` (или `OPENAI_*`) — чат-модель;
- эмбеддинги локальные и настраиваются в `backend/main.py`;
- остальные лимиты (`HISTORY_LIMIT`, `MAX_*`) необязательны, значения по
  умолчанию заданы в коде.

`.env` не коммитится и не попадает в образ. На стенде значения подставляет
платформа; переменные окружения имеют приоритет над `.env`.

## Локальная разработка

Бэкенд:

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows, или source .venv/bin/activate
pip install -r backend/requirements-dev.txt
cd backend
uvicorn main:app --reload --port 8000
```

Фронтенд (Node.js 18+):

```bash
cd frontend
npm install
npm run dev                      # http://localhost:5173, /api проксируется на :8000
```

## Тесты

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q
```

Тесты не поднимают реальный RAG: агент подменяется заглушкой, проверяются
поток SSE, накопление и изоляция истории, восстановление после перезагрузки,
закрытие вкладки и вырезание `Источники:`.

## Docker: детали

```bash
docker compose up -d --build     # собрать, подготовить БД и запустить
docker compose logs -f backend   # логи RAG и агента
docker compose down              # остановить
```

Логи приложения отключены. Healthcheck backend в Compose опрашивает
`/api/health` каждые 10 секунд и проверяет `status: ok`. Frontend ждёт
готовности backend. Для ожидания всех сервисов в терминале добавьте `--wait`.

## Стек

React · TypeScript · Vite · FastAPI · LangGraph · BM25 + FAISS · SSE · Docker · nginx
