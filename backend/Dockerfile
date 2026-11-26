FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Рабочая копия модели эмбеддингов внутри образа: на чистом клоне ./models
# пуст, а compose монтирует его поверх /app/models только для чтения.
# Offline-флаги выше действуют лишь в рантайме — на время загрузки выключаем.
# Лишние веса (pytorch_model.bin, onnx, openvino) не тянем: ~500 МБ вместо 2 ГБ.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-small', local_dir='/opt/models/multilingual-e5-small', ignore_patterns=['*.bin', '*.onnx', '*.h5', '*.msgpack', 'onnx/*', 'openvino/*', '.cache/*'])"

COPY backend ./backend
COPY rag ./rag
COPY script ./script
COPY agents ./agents

# Готовый корпус чанков: из него строится индекс на старте.
COPY dataset/result_chunks.csv ./dataset/result_chunks.csv

# Модель с хоста не копируется через build context (это 2 ГБ): compose
# монтирует ./models в /app/models, а запасная копия уже лежит в /opt/models.
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
