"""
main.py — FastAPI entrypoint for the Store Intelligence API.

Endpoints:
  POST /events/ingest
  GET  /stores/{store_id}/metrics
  GET  /stores/{store_id}/funnel
  GET  /stores/{store_id}/heatmap
  GET  /stores/{store_id}/anomalies
  GET  /health

Production features:
  - Structured JSON logging with trace_id, latency_ms
  - Graceful degradation: DB unavailable → HTTP 503
  - CORS enabled for dashboard
"""

import time
import uuid
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError

from app.database import get_db, init_db
from app.models import (
    IngestRequest, IngestResult,
    StoreMetrics, FunnelResponse, HeatmapResponse,
    AnomaliesResponse, HealthResponse, ErrorResponse,
)
from app.ingestion import ingest_events
from app.metrics import get_store_metrics
from app.funnel import get_funnel
from app.heatmap import get_heatmap
from app.anomalies import get_anomalies
from app.health import get_health

# ─────────────────────────────────────────────
# Logging (structured JSON-like)
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":%(message)s}',
)
logger = logging.getLogger("store_intelligence.api")


# ─────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info('"Store Intelligence API started"')
    yield
    logger.info('"Store Intelligence API shutting down"')


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Store Intelligence API",
    description="Apex Retail — real-time store analytics from CCTV events.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Middleware: structured request logging
# ─────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    trace_id = str(uuid.uuid4())[:8]
    request.state.trace_id = trace_id
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(
            '"endpoint":"%s","trace_id":"%s","error":"%s"',
            request.url.path, trace_id, str(exc),
        )
        raise

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    store_id = request.path_params.get("store_id", "-")

    logger.info(
        '"trace_id":"%s","store_id":"%s","endpoint":"%s %s",'
        '"status_code":%d,"latency_ms":%s',
        trace_id, store_id, request.method, request.url.path,
        response.status_code, latency_ms,
    )
    response.headers["X-Trace-Id"] = trace_id
    return response


# ─────────────────────────────────────────────
# DB error handler
# ─────────────────────────────────────────────
def _db_guard(trace_id: str):
    """Returns a 503 JSONResponse for DB errors."""
    return JSONResponse(
        status_code=503,
        content={"error": "Database unavailable", "trace_id": trace_id},
    )


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.post(
    "/events/ingest",
    response_model=IngestResult,
    summary="Ingest a batch of store events (max 500)",
    status_code=200,
)
def ingest(request: Request, payload: IngestRequest, db: Session = Depends(get_db)):
    trace_id = getattr(request.state, "trace_id", "-")
    try:
        result = ingest_events(payload.events, db)
        return result
    except OperationalError:
        return _db_guard(trace_id)


@app.get(
    "/stores/{store_id}/metrics",
    response_model=StoreMetrics,
    summary="Real-time store metrics",
)
def metrics(request: Request, store_id: str, hours: int = 24, db: Session = Depends(get_db)):
    trace_id = getattr(request.state, "trace_id", "-")
    try:
        return get_store_metrics(store_id, db, hours=hours)
    except OperationalError:
        return _db_guard(trace_id)


@app.get(
    "/stores/{store_id}/funnel",
    response_model=FunnelResponse,
    summary="Conversion funnel: Entry → Zone → Billing → Purchase",
)
def funnel(request: Request, store_id: str, hours: int = 24, db: Session = Depends(get_db)):
    trace_id = getattr(request.state, "trace_id", "-")
    try:
        return get_funnel(store_id, db, hours=hours)
    except OperationalError:
        return _db_guard(trace_id)


@app.get(
    "/stores/{store_id}/heatmap",
    response_model=HeatmapResponse,
    summary="Zone visit heatmap (normalised 0-100)",
)
def heatmap(request: Request, store_id: str, hours: int = 24, db: Session = Depends(get_db)):
    trace_id = getattr(request.state, "trace_id", "-")
    try:
        return get_heatmap(store_id, db, hours=hours)
    except OperationalError:
        return _db_guard(trace_id)


@app.get(
    "/stores/{store_id}/anomalies",
    response_model=AnomaliesResponse,
    summary="Active operational anomalies",
)
def anomalies(request: Request, store_id: str, db: Session = Depends(get_db)):
    trace_id = getattr(request.state, "trace_id", "-")
    try:
        return get_anomalies(store_id, db)
    except OperationalError:
        return _db_guard(trace_id)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health + per-store feed status",
)
def health(request: Request, db: Session = Depends(get_db)):
    trace_id = getattr(request.state, "trace_id", "-")
    try:
        return get_health(db)
    except OperationalError:
        return _db_guard(trace_id)
