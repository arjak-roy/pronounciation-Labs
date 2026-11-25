# ============================================================================
# Production Dockerfile — Mode B Pronunciation Engine (AWS EC2 / ECS)
# ============================================================================
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies: espeak-ng, libsndfile1, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libespeak-ng1 \
    libsndfile1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
COPY scripts ./scripts

# Use lightweight CPU PyTorch wheels by default on CPU instances (saves ~2.2 GB disk space)
ARG TARGET_DEVICE=cpu
RUN pip install --upgrade pip && \
    if [ "$TARGET_DEVICE" = "gpu" ]; then \
        pip install -e ".[gpu]"; \
    else \
        pip install --extra-index-url https://download.pytorch.org/whl/cpu -e .; \
    fi

EXPOSE 8000

# Automatically bootstrap Mode B INT8 ONNX model if /app/models is empty, then launch uvicorn
CMD ["sh", "-c", "python -m scripts.bootstrap_models && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
