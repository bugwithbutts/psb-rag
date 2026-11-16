#!/bin/sh
set -e

if [ "${VECTOR_STORE:-faiss}" = "pgvector" ]; then
    python -m database.bootstrap
fi

exec "$@"
