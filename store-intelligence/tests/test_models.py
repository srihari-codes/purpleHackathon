"""
tests/test_models.py — Unit tests for Pydantic data contracts (app/models.py).

25 tests covering:
  - Valid construction of every event type
  - Every field validator (uuid, non-empty, timestamp, confidence, dwell_ms)
  - VisitorSession helpers (is_active, duration_ms, add_zone_dwell, record_zone_visit)
  - IngestRequest batch size constraints
  - StoredEvent auto-populated ingested_at
"""

import uuid
import datetime as dt
from datetime import timezone

import pytest

from app.models import (
    EventMetadata,
    EventType,
    InboundEvent,
    IngestRequest,
    QueueEvent,
    StoredEvent,
    VisitorSession,
)
from tests.conftest import make_raw_event, make_ts


# ---------------------------------------------------------------------------
# 1. Valid event construction for every EventType
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("etype", [e.value for e in EventType])
def test_valid_event_all_types(etype):
    """Every event type in the catalogue must parse without error."""
    raw = make_raw_event(etype, "VIS_01", zone_id="ZONE_A" if "ZONE" in etype else None)
    evt = InboundEvent(**raw)
    assert evt.event_type == EventType(etype)


def test_valid_event_fields():
    """Basic field population from a well-formed raw dict."""
    raw = make_raw_event("ENTRY", "VIS_01", store_id="ST001", camera_id="CAM_ENTRY_01", confidence=0.85)
    evt = InboundEvent(**raw)
    assert evt.visitor_id == "VIS_01"
    assert evt.store_id == "ST001"
    assert evt.camera_id == "CAM_ENTRY_01"
    assert evt.confidence == 0.85
    assert evt.is_staff is False
    assert evt.dwell_ms == 0


# ---------------------------------------------------------------------------
# 2. event_id validators
# ---------------------------------------------------------------------------

def test_invalid_uuid_raises():
    """Non-UUID event_id must raise a validation error."""
    raw = make_raw_event("ENTRY", "VIS_01")
    raw["event_id"] = "not-a-uuid"
    with pytest.raises(ValueError, match="valid UUID"):
        InboundEvent(**raw)


def test_empty_event_id_raises():
    """Blank event_id must raise a non-empty-string error."""
    raw = make_raw_event("ENTRY", "VIS_01")
    raw["event_id"] = "   "
    with pytest.raises(ValueError):
        InboundEvent(**raw)


# ---------------------------------------------------------------------------
# 3. Non-empty string validators
# ---------------------------------------------------------------------------

def test_empty_visitor_id_raises():
    raw = make_raw_event("ENTRY", "VIS_01")
    raw["visitor_id"] = "   "
    with pytest.raises(ValueError, match="visitor_id must be a non-empty string"):
        InboundEvent(**raw)


def test_empty_store_id_raises():
    raw = make_raw_event("ENTRY", "VIS_01")
    raw["store_id"] = ""
    with pytest.raises(ValueError, match="store_id must be a non-empty string"):
        InboundEvent(**raw)


def test_empty_camera_id_raises():
    raw = make_raw_event("ENTRY", "VIS_01")
    raw["camera_id"] = "   "
    with pytest.raises(ValueError, match="camera_id must be a non-empty string"):
        InboundEvent(**raw)


# ---------------------------------------------------------------------------
# 4. Timestamp validators
# ---------------------------------------------------------------------------

def test_invalid_timestamp_format_raises():
    raw = make_raw_event("ENTRY", "VIS_01")
    raw["timestamp"] = "not-a-date"
    with pytest.raises(ValueError, match="not a valid ISO-8601 datetime"):
        InboundEvent(**raw)


def test_future_timestamp_beyond_60s_raises():
    raw = make_raw_event("ENTRY", "VIS_01")
    future = dt.datetime.now(timezone.utc) + dt.timedelta(minutes=5)
    raw["timestamp"] = future.isoformat()
    with pytest.raises(ValueError, match="max 60 s clock skew allowed"):
        InboundEvent(**raw)


def test_timezone_naive_timestamp_raises():
    raw = make_raw_event("ENTRY", "VIS_01")
    raw["timestamp"] = "2026-03-03T14:00:00"  # no timezone
    with pytest.raises(ValueError, match="timezone information"):
        InboundEvent(**raw)


def test_timestamp_with_z_suffix_accepted():
    """ISO timestamp ending in 'Z' must be accepted and round-trip correctly."""
    raw = make_raw_event("ENTRY", "VIS_01", timestamp="2026-03-03T14:00:00Z")
    evt = InboundEvent(**raw)
    assert evt.parsed_timestamp().year == 2026


def test_future_within_60s_is_accepted():
    """Timestamps up to 60 s in the future must be allowed (clock skew guard)."""
    raw = make_raw_event("ENTRY", "VIS_01")
    slightly_future = dt.datetime.now(timezone.utc) + dt.timedelta(seconds=30)
    raw["timestamp"] = slightly_future.isoformat().replace("+00:00", "Z")
    evt = InboundEvent(**raw)
    assert evt is not None


# ---------------------------------------------------------------------------
# 5. Confidence validators
# ---------------------------------------------------------------------------

def test_confidence_below_zero_raises():
    raw = make_raw_event("ENTRY", "VIS_01")
    raw["confidence"] = -0.1
    with pytest.raises(ValueError):
        InboundEvent(**raw)


def test_confidence_above_one_raises():
    raw = make_raw_event("ENTRY", "VIS_01")
    raw["confidence"] = 1.001
    with pytest.raises(ValueError):
        InboundEvent(**raw)


def test_confidence_boundary_values():
    """Exactly 0.0 and 1.0 must be accepted."""
    for val in (0.0, 1.0):
        raw = make_raw_event("ENTRY", "VIS_01", confidence=val)
        evt = InboundEvent(**raw)
        assert evt.confidence == val


# ---------------------------------------------------------------------------
# 6. Strict-mode type coercion rejection
# ---------------------------------------------------------------------------

def test_dwell_ms_as_string_rejected():
    """In strict mode, string cannot be coerced to int for dwell_ms."""
    raw = make_raw_event("ZONE_DWELL", "VIS_01", zone_id="Z1")
    raw["dwell_ms"] = "500"
    with pytest.raises(Exception):
        InboundEvent(**raw)


def test_is_staff_as_string_rejected():
    """In strict mode, 'True' string cannot be coerced to bool."""
    raw = make_raw_event("ENTRY", "VIS_01")
    raw["is_staff"] = "True"
    with pytest.raises(Exception):
        InboundEvent(**raw)


# ---------------------------------------------------------------------------
# 7. EventMetadata
# ---------------------------------------------------------------------------

def test_metadata_extra_fields_allowed():
    """EventMetadata has extra='allow' — unknown fields must not raise."""
    m = EventMetadata(queue_depth=3, unknown_field="hello")
    assert m.queue_depth == 3


def test_metadata_all_none_by_default():
    m = EventMetadata()
    assert m.queue_depth is None
    assert m.reid_score is None
    assert m.confidence_lineage is None


# ---------------------------------------------------------------------------
# 8. VisitorSession helpers
# ---------------------------------------------------------------------------

def test_visitor_session_is_active():
    s = VisitorSession(visitor_id="V1", store_id="S1", start_time=make_ts(0))
    assert s.is_active is True
    s.end_time = make_ts(60)
    assert s.is_active is False


def test_visitor_session_duration_ms():
    s = VisitorSession(visitor_id="V1", store_id="S1", start_time=make_ts(0))
    assert s.duration_ms is None  # still active
    s.end_time = make_ts(120)
    assert s.duration_ms == 120_000


def test_visitor_session_add_zone_dwell_accumulates():
    s = VisitorSession(visitor_id="V1", store_id="S1", start_time=make_ts(0))
    s.add_zone_dwell("ZONE_SKIN", 5000)
    s.add_zone_dwell("ZONE_SKIN", 3000)
    assert s.dwell_per_zone["ZONE_SKIN"] == 8000


def test_visitor_session_record_zone_visit_no_duplicates():
    s = VisitorSession(visitor_id="V1", store_id="S1", start_time=make_ts(0))
    s.record_zone_visit("ZONE_A")
    s.record_zone_visit("ZONE_A")
    s.record_zone_visit("ZONE_B")
    assert s.zones_visited == ["ZONE_A", "ZONE_B"]


# ---------------------------------------------------------------------------
# 9. IngestRequest constraints
# ---------------------------------------------------------------------------

def test_ingest_request_empty_batch_rejected():
    with pytest.raises(Exception):  # min_length=1
        from app.models import IngestRequest
        IngestRequest(events=[])


def test_ingest_request_too_large_rejected():
    from app.models import IngestRequest
    big = [make_raw_event("ENTRY", f"V{i}") for i in range(501)]
    with pytest.raises(Exception):  # max_length=500
        IngestRequest(events=big)


# ---------------------------------------------------------------------------
# 10. StoredEvent
# ---------------------------------------------------------------------------

def test_stored_event_ingested_at_auto_set():
    raw = make_raw_event("ENTRY", "VIS_01")
    evt = InboundEvent(**raw)
    stored = StoredEvent(event=evt)
    assert stored.ingested_at.endswith("Z")
    assert "2026" in stored.ingested_at or "2025" in stored.ingested_at  # sanity — it's a real time
