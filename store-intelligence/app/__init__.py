"""
app/ — Store Intelligence API package.

Public surface for Layer 2 (Event Stream & Session Core):
    models     — Pydantic schemas (InboundEvent, VisitorSession, ...)
    ingestion  — EventStore, IngestionPipeline
    sessionizer — SessionStore, Sessionizer, build_session_pipeline()
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

__all__ = [
    "EventType", "EventMetadata", "InboundEvent",
    "IngestRequest", "IngestResponse", "RejectedEvent",
    "QueueEvent", "VisitorSession", "StoredEvent",
    "EventStore", "IngestionPipeline",
    "SessionStore", "Sessionizer", "build_session_pipeline",
]
