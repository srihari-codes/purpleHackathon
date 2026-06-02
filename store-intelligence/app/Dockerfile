# ─────────────────────────────────────────────────────────────────────────────
# Store Intelligence — API Layer Dockerfile
#
# Build:
#   docker build -t store-intelligence-api -f app/Dockerfile .
#
# Run:
#   docker run -p 8000:8000 store-intelligence-api
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="store-intelligence"
LABEL description="Store Intelligence API — Layer 2 (Event Stream + Session Core)"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /api

# ── Python deps ───────────────────────────────────────────────────────────────
COPY app/requirements_api.txt ./requirements_api.txt
RUN pip install --no-cache-dir -r requirements_api.txt

# ── Application source ────────────────────────────────────────────────────────
COPY app/ ./app/

# ── Runtime config ────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV POS_CSV=/data/pos_transactions.csv

EXPOSE 8000

# Health check for docker compose depends_on
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
