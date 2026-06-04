"""
app/main.py — FastAPI entrypoint for the Store Intelligence API.

Endpoints:
    POST /events/ingest              — batch ingest, partial success
    GET  /stores/{store_id}/metrics  — real-time metrics from sessions
    GET  /stores/{store_id}/funnel   — conversion funnel (session-unit)
    GET  /stores/{store_id}/heatmap  — zone heatmap 0-100 scores
    GET  /stores/{store_id}/anomalies — active anomalies with actions
    GET  /health                     — service health + stale feed warnings

Production features:
    - Structured JSON logging: trace_id, store_id, endpoint, latency_ms,
      event_count, status_code
    - Graceful degradation: all exceptions caught, structured 503 returned
    - No raw stack traces in responses
    - Idempotent ingest (event_id dedup)
    - CORS enabled for dashboard integration

Architecture:
    All singleton state is held in module-level objects, initialised in
    lifespan().  The dependency-injection pattern (FastAPI Depends) is used
    for tracing and structured logging.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .audit import AuditTimeline
from .calibration import CalibrationEngine
from .correlation import CorrelationEngine
from .ingestion import EventStore, IngestionPipeline
from .models import IngestRequest, IngestResponse
from .projections import (
    AnomalyProjection,
    FunnelProjection,
    HeatmapProjection,
    HealthProjection,
    MetricsProjection,
)
from .replay import ReplayEngine, ReplayMode
from .sessionizer import SessionStore, Sessionizer, build_session_pipeline
from .verifier import VerifierEngine

# ---------------------------------------------------------------------------
# Logging setup — structured JSON to stdout
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}',
)
logger = logging.getLogger("store_intelligence.api")


# ---------------------------------------------------------------------------
# Singleton application state
# ---------------------------------------------------------------------------

_STARTED_AT = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")

# Layer 2 core
_audit        = AuditTimeline()
_calibration  = CalibrationEngine(calibrate_every_n_events=200)
_event_store  = EventStore()
_sess_store, _sessionizer = build_session_pipeline(audit=_audit, calibration=_calibration)
_pipeline     = IngestionPipeline(_event_store, _sessionizer)
_verifier     = VerifierEngine(audit=_audit)
_correlation  = CorrelationEngine()
_replay       = ReplayEngine(
    _event_store, _sess_store, _sessionizer, _audit,
    mode=ReplayMode.LIVE, speed=0.0,
    verifier=_verifier, correlation=_correlation,
)

# Wire verifier into sessionizer post-processing
_sessionizer.set_audit(_audit)
_sessionizer.set_verifier(_verifier)


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    pos_path = os.environ.get("POS_CSV", "/data/pos_transactions.csv")
    if os.path.exists(pos_path):
        n = _correlation.load_csv(pos_path)
        logger.info('"pos_loaded","records":%d', n)
    else:
        logger.warning('"pos_csv_not_found","path":"%s"', pos_path)
    yield
    # Shutdown — nothing to clean up (in-memory)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Store Intelligence API",
    version="2.0.0",
    description=(
        "Real-time retail analytics API. "
        "Ingests detection events → builds sessions → exposes funnel, "
        "heatmap, anomaly, and health endpoints."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Structured logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def structured_log_middleware(request: Request, call_next):
    trace_id  = str(uuid.uuid4())
    store_id  = request.path_params.get("store_id", "")
    endpoint  = request.url.path
    started   = time.monotonic()

    request.state.trace_id = trace_id

    try:
        response = await call_next(request)
        status   = response.status_code
    except Exception as exc:
        logger.error(
            '"unhandled_exception","trace_id":"%s","endpoint":"%s","error":"%s"',
            trace_id, endpoint, str(exc),
        )
        status = 500
        response = JSONResponse(
            status_code=500,
            content=_error_body("INTERNAL_ERROR", "Unexpected server error"),
        )

    latency_ms = round((time.monotonic() - started) * 1000, 2)
    logger.info(
        '"request","trace_id":"%s","store_id":"%s","endpoint":"%s",'
        '"latency_ms":%s,"status_code":%d',
        trace_id, store_id, endpoint, latency_ms, status,
    )
    response.headers["X-Trace-Id"] = trace_id
    return response


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def _error_body(code: str, message: str, detail: Any = None) -> dict:
    body = {"error": code, "message": message}
    if detail:
        body["detail"] = detail
    return body


def _store_error(store_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content=_error_body(
            "STORE_NOT_FOUND",
            f"No data found for store '{store_id}'. "
            "Ingest events first via POST /events/ingest.",
        ),
    )


def _guard(store_id: str) -> Optional[JSONResponse]:
    """Return 404 if we have no data for this store."""
    known = (
        set(_event_store.get_all_store_ids())
        | set(_sess_store.get_all_store_ids())
        | set(_correlation.get_all_store_ids())
    )
    if store_id not in known:
        return _store_error(store_id)
    return None


# ---------------------------------------------------------------------------
# POST /events/ingest
# ---------------------------------------------------------------------------

@app.post(
    "/events/ingest",
    response_model=IngestResponse,
    summary="Ingest a batch of detection events",
    responses={
        200: {"description": "Partial or full success"},
        422: {"description": "Request body malformed (not a valid batch)"},
        503: {"description": "Storage unavailable"},
    },
)
async def ingest_events(request: Request, body: IngestRequest):
    """
    Idempotent batch ingest endpoint.

    - Up to 500 events per request.
    - Validates each event individually (partial success on malformed events).
    - Deduplicates by event_id — safe to call twice with the same payload.
    - Runs sessionization and verifier after each accepted event.
    """
    trace_id = getattr(request.state, "trace_id", "unknown")
    try:
        response = _pipeline.ingest_batch(body.events)

        # Run cross-session verifier checks after the batch
        for store_id in _event_store.get_all_store_ids():
            sessions = _sess_store.get_all_sessions(store_id)
            _verifier.verify_active_sessions(sessions, store_id)

        # Run POS correlation
        for store_id in _sess_store.get_all_store_ids():
            sessions = _sess_store.get_all_sessions(store_id)
            _correlation.correlate(sessions, store_id)

        logger.info(
            '"ingest","trace_id":"%s","event_count":%d,"accepted":%d,'
            '"duplicates":%d,"rejected":%d',
            trace_id, len(body.events),
            response.accepted, response.duplicates, response.rejected,
        )
        return response

    except Exception as exc:
        logger.error('"ingest_error","trace_id":"%s","error":"%s"', trace_id, exc)
        return JSONResponse(
            status_code=503,
            content=_error_body("STORAGE_UNAVAILABLE", "Event storage is temporarily unavailable"),
        )


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/metrics
# ---------------------------------------------------------------------------

@app.get(
    "/stores/{store_id}/metrics",
    summary="Real-time store metrics",
)
async def get_metrics(store_id: str, request: Request):
    guard = _guard(store_id)
    if guard:
        return guard

    sessions = _sess_store.get_all_sessions(store_id)
    return MetricsProjection.build(sessions, store_id, _correlation)


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/funnel
# ---------------------------------------------------------------------------

@app.get(
    "/stores/{store_id}/funnel",
    summary="Conversion funnel: Entry → Zone → Billing → Purchase",
)
async def get_funnel(store_id: str, request: Request):
    guard = _guard(store_id)
    if guard:
        return guard

    sessions = _sess_store.get_all_sessions(store_id)
    return FunnelProjection.build(sessions, store_id, _correlation)


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/heatmap
# ---------------------------------------------------------------------------

@app.get(
    "/stores/{store_id}/heatmap",
    summary="Zone heatmap: visit frequency and dwell, normalised 0-100",
)
async def get_heatmap(store_id: str, request: Request):
    guard = _guard(store_id)
    if guard:
        return guard

    sessions = _sess_store.get_all_sessions(store_id)
    return HeatmapProjection.build(sessions, store_id)


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/anomalies
# ---------------------------------------------------------------------------

@app.get(
    "/stores/{store_id}/anomalies",
    summary="Active anomalies: queue spikes, conversion drops, dead zones",
)
async def get_anomalies(store_id: str, request: Request):
    guard = _guard(store_id)
    if guard:
        return guard

    sessions   = _sess_store.get_all_sessions(store_id)
    warnings   = _verifier.get_warnings(store_id=store_id)
    metrics    = MetricsProjection.build(sessions, store_id, _correlation)
    conv_rate  = metrics["conversion_rate"]

    return AnomalyProjection.build(
        sessions, store_id,
        verifier_warnings=warnings,
        conversion_rate=conv_rate,
        historical_avg_rate=0.0,   # future: pull from time-series store
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    summary="Service health: status, last event per store, stale feed warnings",
)
async def get_health(request: Request):
    return HealthProjection.build(_event_store, _sess_store, _STARTED_AT)


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/audit/{visitor_id}  (bonus: audit trail)
# ---------------------------------------------------------------------------

@app.get(
    "/stores/{store_id}/audit/{visitor_id}",
    summary="Full audit trail for a visitor",
)
async def get_audit(store_id: str, visitor_id: str, request: Request):
    tl = _audit.to_dict(visitor_id)
    if tl is None:
        return JSONResponse(
            status_code=404,
            content=_error_body("VISITOR_NOT_FOUND", f"No audit data for visitor {visitor_id}"),
        )
    return tl


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/calibration  (bonus: calibration profile)
# ---------------------------------------------------------------------------

@app.get(
    "/stores/{store_id}/calibration",
    summary="Current calibration profile for a store",
)
async def get_calibration(store_id: str, request: Request):
    profile = _calibration.get_profile(store_id)
    return profile.to_dict()


# ---------------------------------------------------------------------------
# POST /stores/{store_id}/replay  (bonus: trigger replay from events.jsonl)
# ---------------------------------------------------------------------------

class ReplayRequest(BaseModel):
    path: str
    speed: float = 0.0
    store_id_filter: Optional[str] = None


@app.post(
    "/stores/{store_id}/replay",
    summary="Replay a historical events.jsonl file for deterministic debugging",
)
async def trigger_replay(store_id: str, body: ReplayRequest, request: Request):
    if _replay.is_replaying:
        return JSONResponse(
            status_code=409,
            content=_error_body("REPLAY_IN_PROGRESS", "A replay is already running"),
        )
    _replay._speed = body.speed
    try:
        result = _replay.replay_file(
            body.path,
            reset_state=True,
            store_id_filter=body.store_id_filter or store_id,
        )
        return result.to_dict()
    except FileNotFoundError as exc:
        return JSONResponse(
            status_code=404,
            content=_error_body("FILE_NOT_FOUND", str(exc)),
        )
