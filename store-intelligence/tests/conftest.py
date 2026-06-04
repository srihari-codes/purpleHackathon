"""
tests/conftest.py — Shared pytest fixtures and helpers for the full test suite.
"""

import uuid
import datetime as dt
from datetime import timezone

import pytest
from fastapi.testclient import TestClient

from app.audit import AuditTimeline
from app.calibration import CalibrationEngine
from app.correlation import CorrelationEngine
from app.ingestion import EventStore, IngestionPipeline
from app.models import EventType, InboundEvent, QueueEvent, VisitorSession
from app.replay import ReplayEngine, ReplayMode
from app.sessionizer import SessionStore, Sessionizer, build_session_pipeline
from app.verifier import VerifierEngine


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

BASE_DT = dt.datetime(2026, 3, 3, 14, 0, 0, tzinfo=timezone.utc)


def make_ts(offset_sec: int = 0) -> str:
    return (BASE_DT + dt.timedelta(seconds=offset_sec)).isoformat().replace("+00:00", "Z")


def make_raw_event(
    etype: str,
    visitor_id: str,
    *,
    event_id: str = None,
    store_id: str = "STORE_TEST",
    camera_id: str = "CAM_ENTRY_01",
    timestamp: str = None,
    zone_id: str = None,
    dwell_ms: int = 0,
    is_staff: bool = False,
    confidence: float = 0.90,
    queue_depth: int = None,
    reid_score: float = None,
    wait_duration_ms: int = None,
    offset_sec: int = 0,
) -> dict:
    if event_id is None:
        event_id = str(uuid.uuid4())
    if timestamp is None:
        timestamp = make_ts(offset_sec)

    meta: dict = {}
    if queue_depth is not None:
        meta["queue_depth"] = queue_depth
    if reid_score is not None:
        meta["reid_score"] = reid_score
    if wait_duration_ms is not None:
        meta["wait_duration_ms"] = wait_duration_ms

    return {
        "event_id": event_id,
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": etype,
        "timestamp": timestamp,
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": confidence,
        "metadata": meta,
    }


# ---------------------------------------------------------------------------
# Fresh-state fixtures (re-created per test)
# ---------------------------------------------------------------------------

@pytest.fixture()
def event_store():
    return EventStore()


@pytest.fixture()
def audit():
    return AuditTimeline()


@pytest.fixture()
def calibration():
    return CalibrationEngine(calibrate_every_n_events=5)


@pytest.fixture()
def verifier(audit):
    return VerifierEngine(audit=audit)


@pytest.fixture()
def sess_store():
    return SessionStore()


@pytest.fixture()
def sessionizer(sess_store, audit, calibration, verifier):
    s = Sessionizer(sess_store, audit=audit, calibration=calibration, verifier=verifier)
    return s


@pytest.fixture()
def pipeline(event_store, sessionizer):
    return IngestionPipeline(event_store, sessionizer)


@pytest.fixture()
def correlation():
    return CorrelationEngine()


@pytest.fixture()
def replay_engine(event_store, sess_store, sessionizer, audit, verifier, correlation):
    return ReplayEngine(
        event_store, sess_store, sessionizer, audit,
        mode=ReplayMode.REPLAY, speed=0.0,
        verifier=verifier, correlation=correlation,
    )


# ---------------------------------------------------------------------------
# App-level API client (uses the singleton app state — cleared between tests)
# ---------------------------------------------------------------------------

@pytest.fixture()
def api_client():
    from app.main import app, _event_store, _sess_store, _audit, _verifier, _correlation
    _event_store.clear()
    _sess_store.clear()
    _audit.clear()
    _verifier.clear()
    _correlation.clear()
    with TestClient(app) as c:
        yield c
    # teardown
    _event_store.clear()
    _sess_store.clear()
    _audit.clear()
    _verifier.clear()
    _correlation.clear()
