# ─────────────────────────────────────────────────────────
# Backend Dockerfile — multi-stage build
#
# Stage 1 (builder): install dependencies into a venv
# Stage 2 (runtime): copy venv + source, run uvicorn
#
# Model weights (SigLIP) are downloaded at first request via
# lazy loading. For production, pre-warm by calling /health
# after startup.
# ─────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps required by faiss-cpu and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into isolated venv
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY . .

# HuggingFace cache — keep weights across container restarts
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# Disable macOS-specific OMP workaround (not needed on Linux)
ENV KMP_DUPLICATE_LIB_OK=FALSE

# Expose API port
EXPOSE 8000

# Health check — hits /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run uvicorn with 2 workers (single-worker on free Render tier)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
