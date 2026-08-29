# syntax=docker/dockerfile:1

# ---- build ---------------------------------------------------------------
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies resolve from pyproject alone, so this layer caches across
# source edits.
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install -r pyproject.toml

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    VIRTUAL_ENV=/opt/venv uv pip install --no-deps .

# ---- runtime -------------------------------------------------------------
FROM python:3.13-slim AS runtime

RUN useradd --create-home --uid 10001 cropcraft
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=cropcraft:cropcraft templates/ ./templates/
COPY --chown=cropcraft:cropcraft static/ ./static/
COPY --chown=cropcraft:cropcraft Data/Crop_NPK.csv ./Data/Crop_NPK.csv
COPY --chown=cropcraft:cropcraft models/ ./models/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER cropcraft
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

CMD ["uvicorn", "cropcraft.main:app", "--host", "0.0.0.0", "--port", "8000"]
