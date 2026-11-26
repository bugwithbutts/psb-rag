# Backend — FastAPI-сервис чат-бота «Юридический ассистент ПСБ»

Сервис принимает новый вопрос и `conversation_id`, хранит историю диалога в
памяти процесса и отдаёт ответ агента потоком SSE.

История живёт в памяти процесса: при рестарте контейнера диалоги теряются.
Несколько воркеров uvicorn тоже не подходят — у каждого процесса свой словарь.

## Файлы

- `main.py` — приложение, эндпоинты, память диалогов, SSE;
- `test_main.py` — pytest-тесты потока и памяти;
- `requirements.txt` / `requirements-dev.txt` — зависимости.

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

Эмбеддинги считаются локально моделью `models/multilingual-e5-small`
(путь задан в `main.py`). Нужен только `.env` в корне репозитория с
`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` — это чат-модель.

Если каталога `models/multilingual-e5-small` нет, бэкенд берёт копию из
`/opt/models/multilingual-e5-small`, которая скачивается при сборке образа
(см. раздел «Docker»).

Документация — http://127.0.0.1:8000/docs.

## API

| Метод | Путь | Назначение |
| --- | --- | --- |
| `GET` | `/` | описание сервиса |
| `GET` | `/api/health` | `{"status":"ok","agent":true}` |
| `POST` | `/api/chat` | поток SSE: `meta` → `token*` → `done` |
| `GET` | `/api/conversations/{id}` | история диалога |
| `DELETE` | `/api/conversations/{id}` | сброс диалога |
| `POST` | `/api/conversations/{id}/close` | сигнал «вкладка закрылась» |

### `POST /api/chat`

```json
{ "query": "А до какого числа её вносить?", "conversation_id": "9f1c…" }
```

`conversation_id` необязателен: если его нет, сервер сгенерирует свой и вернёт
в первом событии. Допустимы латиница, цифры и дефис, 8–64 символа.

Ответ — `text/event-stream`:

```
data: {"type": "meta", "conversationId": "9f1c…"}
data: {"type": "token", "text": "Арендная плата "}
data: {"type": "done", "answer": "…", "latencyMs": 4210}
data: [DONE]
```

При ошибке агента вместо `token*` приходит `{"type": "error", "message": "…"}`.

### Как собирается запрос к агенту

История хранится как список реплик `{"role", "content"}`. Перед вызовом она
обрезается (не больше 8 реплик и 6000 символов) и склеивается в строку с
метками `Пользователь:` / `Ассистент:`. Двоеточия в метках ролей внутри
пользовательского текста заменяются на `∶`, чтобы нельзя было подделать
реплику ассистента.

### Жизненный цикл диалога

- `conversation_id` генерирует фронтенд и хранит в `sessionStorage`: у каждой
  вкладки свой диалог, который переживает F5;
- при закрытии вкладки браузер шлёт beacon на `/close`; сервер помечает диалог
  и удаляет его через 15 секунд;
- если страница перезагрузилась, следующий запрос с тем же id отменяет
  удаление. Beacon, пришедший после запроса новой страницы, отбрасывается по
  метке времени (`{"status":"stale"}`);
- диалог без запросов удаляется через 30 минут.

## Тесты

```bash
pip install -r requirements-dev.txt
cd backend
pytest -q
```

Тесты не поднимают реальный RAG: `main.AGENT` подменяется заглушкой.

## Docker

Из корня репозитория:

```bash
docker compose up --build
```

- зависимости ставятся при сборке образа;
- корпус `dataset/result_chunks.csv` копируется в образ;
- при сборке в образ скачивается модель `intfloat/multilingual-e5-small`
  (`/opt/models`), поэтому `docker compose up --build` работает и на чистом
  клоне без `./models`; лишние веса (`pytorch_model.bin`, `onnx`, `openvino`)
  не тянутся;
- если `./models` заполнен, Compose монтирует его в `/app/models`, и `main.py`
  предпочитает монтированную копию запасной из образа;
- фронтенд собирается там же и проксирует `/api` на бэкенд;
- значения `LLM_*` Compose берёт из `.env` рядом с `docker-compose.yml`.

Логи отключены в `main.py` (`logging.disable`), healthcheck в Compose нет.
