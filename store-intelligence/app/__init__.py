"""
app/ — Store Intelligence API package.

Public surface for Layer 2 (Event Stream & Session Core):
    models      — Pydantic schemas (InboundEvent, VisitorSession, ...)
    ingestion   — EventStore, IngestionPipeline
    sessionizer — SessionStore, Sessionizer, build_session_pipeline()
    audit       — AuditTimeline, AuditRecord, AuditCategory
    calibration — CalibrationEngine, StoreCalibration, CameraCalibration
    replay      — ReplayEngine, ReplayMode, ReplayResult
"""
from .models import (
    EventType,
    EventMetadata,
    InboundEvent,
    IngestRequest,
    IngestResponse,
    RejectedEvent,
    QueueEvent,
    VisitorSession,
    StoredEvent,
)
from .ingestion import EventStore, IngestionPipeline
from .sessionizer import SessionStore, Sessionizer, build_session_pipeline
from .audit import AuditTimeline, AuditRecord, AuditCategory, VisitorAuditTimeline
from .calibration import CalibrationEngine, StoreCalibration, CameraCalibration
from .replay import ReplayEngine, ReplayMode, ReplayResult

__all__ = [
    # models
    "EventType", "EventMetadata", "InboundEvent",
    "IngestRequest", "IngestResponse", "RejectedEvent",
    "QueueEvent", "VisitorSession", "StoredEvent",
    # ingestion
    "EventStore", "IngestionPipeline",
    # sessionizer
    "SessionStore", "Sessionizer", "build_session_pipeline",
    # audit
    "AuditTimeline", "AuditRecord", "AuditCategory", "VisitorAuditTimeline",
    # calibration
    "CalibrationEngine", "StoreCalibration", "CameraCalibration",
    # replay
    "ReplayEngine", "ReplayMode", "ReplayResult",
]
