# =============================================================================
# AwakeForest — Application Dockerfile
# =============================================================================
# Used by: api, celery-worker-*, celery-beat, flower
# Base:    python:3.11-slim (rasterio 1.4+ ships its own GDAL on Linux)
# =============================================================================

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.4 \
    POETRY_HOME=/opt/poetry \
    POETRY_CACHE_DIR=/opt/.cache \
    PATH="/opt/poetry/bin:${PATH}"

# System deps — curl for poetry installer, build-essential for any src wheels
# libexpat1 is required for geospatial libraries (e.g., pyuwsgi, rasterio)
# libgl1/libglib2.0-0 are opencv-python's runtime shared-lib deps (pulled in
# transitively by supervision, used for its NMS utility in geoops/
# exemplar-scan merging) — opencv-python's wheel doesn't bundle these, and
# without them `import cv2` fails with "libxcb.so.1: cannot open shared
# object file" on this slim base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        libpq-dev \
        libexpat1 \
        libgl1 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

# Install Poetry via official installer
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

# Copy dependency files first so Docker caches this layer
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies into the system Python (no virtualenv in container)
ARG INSTALL_DEV=false
RUN poetry config virtualenvs.create false \
    && if [ "$INSTALL_DEV" = "true" ]; then \
         poetry install --no-interaction --no-ansi --no-root; \
       else \
         poetry install --no-interaction --no-ansi --no-root --without dev; \
       fi

# Copy application source
COPY . .

# Default entrypoint (overridden per-service in docker-compose.yml)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]