# Frontend — чат-бот «Юридический ассистент ПСБ»

Браузерный чат на React + TypeScript + Vite. Ответы приходят потоком SSE от
FastAPI-бэкенда из `../backend`.

```bash
npm install
npm run dev
```

Откройте http://localhost:5173 — нужен запущенный бэкенд
(`uvicorn main:app --port 8000`).

Файлы:

- `src/App.tsx` — весь интерфейс и состояние чата;
- `src/api.ts` — SSE-поток, история диалога и `conversation_id` вкладки;
- `src/styles.css` — оформление.

В Docker фронтенд собирается в `frontend/Dockerfile` и раздаётся nginx
(`nginx.conf`): `/api/` проксируется на контейнер `backend`, буферизация SSE
отключена.
