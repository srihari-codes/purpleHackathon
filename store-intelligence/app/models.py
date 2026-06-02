"""
app/models.py — Pydantic data contracts for the Store Intelligence API.

All inbound events from the Detection Layer and all outbound session objects
are typed here.  No business logic lives in this file — only schema + validation.

Event type catalogue (mirrors pipeline/events.py EventType):
    ENTRY, EXIT, REENTRY, ZONE_ENTER, ZONE_EXIT, ZONE_DWELL,
    BILLING_QUEUE_JOIN, BILLING_QUEUE_ABANDON
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Event type catalogue
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    ENTRY                  = "ENTRY"
    EXIT                   = "EXIT"
    REENTRY                = "REENTRY"
    ZONE_ENTER             = "ZONE_ENTER"
    ZONE_EXIT              = "ZONE_EXIT"
    ZONE_DWELL             = "ZONE_DWELL"
    BILLING_QUEUE_JOIN     = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON  = "BILLING_QUEUE_ABANDON"


# ---------------------------------------------------------------------------
# Nested metadata block (mirrors detection layer schema)
# ---------------------------------------------------------------------------

class EventMetadata(BaseModel):
    """
    Metadata block emitted by the Detection Layer.

    All fields are optional — the sessionizer promotes relevant values into
    the session object rather than querying metadata at query time.
    """
    queue_depth:           Optional[int]   = None
    sku_zone:              Optional[str]   = None
    session_seq:           Optional[int]   = None
    behavior_state:        Optional[str]   = None
    reid_score:            Optional[float] = None
    reentry_count:         Optional[int]   = None
    session_duration_ms:   Optional[int]   = None
    wait_duration_ms:      Optional[int]   = None
    det_conf:              Optional[float] = None
    track_conf:            Optional[float] = None
    reid_conf:             Optional[float] = None
    zone_conf:             Optional[float] = None
    confidence_lineage:    Optional[Dict[str, float]] = None

    model_config = {"extra": "allow"}   # forward-compat: allow new fields


# ---------------------------------------------------------------------------
# Inbound event (one detection event from the pipeline)
# ---------------------------------------------------------------------------

class InboundEvent(BaseModel):
    """
    A single structured event emitted by the Detection Layer.

    Validation rules (all rejections produce a structured reason):
    - event_id   : non-empty string
    - store_id   : non-empty string
    - camera_id  : non-empty string
    - visitor_id : non-empty string
    - event_type : must be in EventType catalogue
    - timestamp  : ISO-8601, parseable, not in the future by >60 s
    - confidence : float in [0.0, 1.0]
    """

    event_id:   str        = Field(..., description="Globally unique UUID-v4 per event")
    store_id:   str        = Field(..., description="Store identifier from store_layout.json")
    camera_id:  str        = Field(..., description="Camera that produced this event")
    visitor_id: str        = Field(..., description="Re-ID token; unique per visit session")
    event_type: EventType  = Field(..., description="Event type catalogue value")
    timestamp:  str        = Field(..., description="ISO-8601 UTC timestamp")
    zone_id:    Optional[str]   = Field(None, description="Zone name; null for ENTRY/EXIT")
    dwell_ms:   int             = Field(0,    ge=0, description="Dwell duration in ms")
    is_staff:   bool            = Field(False, description="True if detected as store staff")
    confidence: float           = Field(..., ge=0.0, le=1.0, description="Final propagated confidence")
    metadata:   EventMetadata   = Field(default_factory=EventMetadata)

    # ── validators ────────────────────────────────────────────────────────

    @field_validator("event_id", "store_id", "camera_id", "visitor_id", mode="before")
    @classmethod
    def non_empty_string(cls, v: Any, info) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return s

    @field_validator("timestamp", mode="before")
    @classmethod
    def valid_iso_timestamp(cls, v: Any) -> str:
        """Parse and normalise timestamp; reject unparseable values."""
        raw = str(v).strip()
        # Support trailing Z (UTC) notation
        normalised = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalised)
        except ValueError:
            raise ValueError(f"timestamp '{raw}' is not a valid ISO-8601 datetime")
        # Enforce UTC-awareness
        if dt.tzinfo is None:
            raise ValueError("timestamp must include timezone information (use UTC / 'Z')")
        # Reject events >60 s in the future (clock skew guard)
        now_utc = datetime.now(tz=timezone.utc)
        skew = (dt.replace(tzinfo=timezone.utc if dt.tzinfo is None else dt.tzinfo) - now_utc).total_seconds()
        if skew > 60:
            raise ValueError(f"timestamp is {skew:.0f}s in the future (max 60 s clock skew allowed)")
        return raw

    def parsed_timestamp(self) -> datetime:
        """Return the timestamp as a timezone-aware UTC datetime."""
        normalised = self.timestamp.replace("Z", "+00:00")
        return datetime.fromisoformat(normalised).astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Ingest request / response
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """Batch of up to 500 events for POST /events/ingest."""
    events: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Raw event dicts from the detection pipeline",
    )


class RejectedEvent(BaseModel):
    """One validation failure within a batch."""
    event_id:  Optional[str] = None   # None if event_id itself was missing/invalid
    reason:    str


class IngestResponse(BaseModel):
    """
    Partial-success response for POST /events/ingest.

    accepted   : events written to the store (after dedup)
    duplicates : events already seen (idempotent skip)
    rejected   : events that failed validation (with reasons)
    """
    accepted:   int                  = 0
    duplicates: int                  = 0
    rejected:   int                  = 0
    errors:     List[RejectedEvent]  = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Queue event (embedded in VisitorSession)
# ---------------------------------------------------------------------------

class QueueEvent(BaseModel):
    """A billing queue join or abandon event recorded in the session."""
    event_type:   EventType          # BILLING_QUEUE_JOIN | BILLING_QUEUE_ABANDON
    timestamp:    str
    queue_depth:  Optional[int]      = None
    wait_ms:      Optional[int]      = None


# ---------------------------------------------------------------------------
# Visitor Session — primary business object
# ---------------------------------------------------------------------------

class VisitorSession(BaseModel):
    """
    Canonical visitor session produced by the Sessionizer.

    This is the unit that funnel, metrics, and heatmap endpoints operate on.
    Raw events are NOT used for these computations — sessions are.

    Re-entry semantics:
    - A REENTRY event on visitor_id V does NOT create a second session for V.
      Instead, reentry_count is incremented on the *existing* (possibly closed)
      session, or a continuation note is recorded if the session was closed.
    - purchase_candidate is True if a BILLING_QUEUE_JOIN was ever recorded.
    """

    session_id:          str          = Field(default_factory=lambda: str(uuid.uuid4()))
    visitor_id:          str
    store_id:            str
    start_time:          str          # ISO-8601 of first ENTRY event
    end_time:            Optional[str] = None   # ISO-8601 of EXIT; None if still active

    is_staff:            bool         = False

    # Zone enrichment
    zones_visited:       List[str]    = Field(default_factory=list)   # ordered list of zone_ids visited
    dwell_per_zone:      Dict[str, int] = Field(default_factory=dict) # zone_id → total dwell_ms

    # Queue enrichment
    queue_events:        List[QueueEvent] = Field(default_factory=list)

    # Re-entry and conversion
    reentry_count:       int          = 0
    purchase_candidate:  bool         = False   # True after BILLING_QUEUE_JOIN

    # Diagnostics
    event_count:         int          = 0
    avg_confidence:      float        = 1.0     # rolling average confidence for session events

    # ── helpers ───────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.end_time is None

    @property
    def duration_ms(self) -> Optional[int]:
        """Total session duration in milliseconds; None if still active."""
        if self.end_time is None:
            return None
        start = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))
        end   = datetime.fromisoformat(self.end_time.replace("Z", "+00:00"))
        return int((end - start).total_seconds() * 1000)

    def add_zone_dwell(self, zone_id: str, dwell_ms: int) -> None:
        """Accumulate dwell time for a zone."""
        self.dwell_per_zone[zone_id] = self.dwell_per_zone.get(zone_id, 0) + dwell_ms

    def record_zone_visit(self, zone_id: str) -> None:
        """Add zone to ordered visit list (no duplicates in the list, just track last)."""
        if zone_id not in self.zones_visited:
            self.zones_visited.append(zone_id)


# ---------------------------------------------------------------------------
# Stored raw event wrapper (append-only store entry)
# ---------------------------------------------------------------------------

class StoredEvent(BaseModel):
    """
    Thin wrapper that records the validated event alongside ingest metadata.
    Stored events are never mutated after creation.
    """
    event:        InboundEvent
    ingested_at:  str = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    )
