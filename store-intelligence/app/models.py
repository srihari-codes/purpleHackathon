"""
models.py — Pydantic schemas for request/response validation.
Mirrors the event schema in the problem statement exactly.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Any
from enum import Enum
import uuid

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"


class AnomalySeverity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


# ─────────────────────────────────────────────
# Event Schema
# ─────────────────────────────────────────────

class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: Optional[int] = None


class StoreEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: EventType
    timestamp: datetime
    zone_id: Optional[str] = None
    dwell_ms: int = 0
    is_staff: bool = False
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    metadata: Optional[EventMetadata] = None

    model_config = ConfigDict(use_enum_values=True)

    @model_validator(mode='after')
    def zone_required_for_zone_events(self):
        et = self.event_type
        zone_types = (
            EventType.ZONE_ENTER.value, EventType.ZONE_EXIT.value, EventType.ZONE_DWELL.value,
            "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL",
        )
        if et in zone_types and not self.zone_id:
            raise ValueError(f"zone_id is required for event_type={et}")
        return self


# ─────────────────────────────────────────────
# Ingest Request / Response
# ─────────────────────────────────────────────

class IngestRequest(BaseModel):
    events: List[StoreEvent] = Field(..., max_length=500)


class IngestResult(BaseModel):
    accepted: int
    rejected: int
    duplicate: int
    errors: List[dict] = []


# ─────────────────────────────────────────────
# Metrics Response
# ─────────────────────────────────────────────

class ZoneDwellMetric(BaseModel):
    zone_id: str
    avg_dwell_ms: float
    visit_count: int


class StoreMetrics(BaseModel):
    store_id: str
    window_start: datetime
    window_end: datetime
    unique_visitors: int
    conversion_rate: float
    avg_dwell_per_zone: List[ZoneDwellMetric]
    queue_depth: int
    abandonment_rate: float


# ─────────────────────────────────────────────
# Funnel Response
# ─────────────────────────────────────────────

class FunnelStage(BaseModel):
    stage: str
    count: int
    drop_off_pct: float


class FunnelResponse(BaseModel):
    store_id: str
    window_start: datetime
    window_end: datetime
    stages: List[FunnelStage]


# ─────────────────────────────────────────────
# Heatmap Response
# ─────────────────────────────────────────────

class HeatmapZone(BaseModel):
    zone_id: str
    visit_frequency: int
    avg_dwell_ms: float
    normalised_score: float  # 0–100
    data_confidence: bool  # False if <20 sessions


class HeatmapResponse(BaseModel):
    store_id: str
    zones: List[HeatmapZone]


# ─────────────────────────────────────────────
# Anomaly Response
# ─────────────────────────────────────────────

class Anomaly(BaseModel):
    anomaly_type: str
    severity: AnomalySeverity
    description: str
    suggested_action: str
    detected_at: datetime
    metadata: Optional[dict] = None


class AnomaliesResponse(BaseModel):
    store_id: str
    active_anomalies: List[Anomaly]


# ─────────────────────────────────────────────
# Health Response
# ─────────────────────────────────────────────

class StoreFeedStatus(BaseModel):
    store_id: str
    last_event_timestamp: Optional[datetime]
    lag_minutes: Optional[float]
    status: str  # OK | STALE_FEED


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    stores: List[StoreFeedStatus]


# ─────────────────────────────────────────────
# Error Response
# ─────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[Any] = None
    trace_id: Optional[str] = None
