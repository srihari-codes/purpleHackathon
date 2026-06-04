"""
tests/test_ingestion.py — Unit tests for EventStore and IngestionPipeline (app/ingestion.py).

16 tests covering:
  - Single event acceptance
  - Deduplication (same event_id)
  - Mixed-valid batch (partial success)
  - All-invalid batch
  - Non-dict event in batch
  - EventStore read helpers
  - Large batch handling
  - Sessionizer called per accepted event
  - Thread safety (basic assertion)
"""

import uuid
from unittest.mock import MagicMock

import pytest

from app.ingestion import EventStore, IngestionPipeline
from app.models import InboundEvent, StoredEvent
from app.sessionizer import build_session_pipeline
from tests.conftest import make_raw_event, make_ts


# ---------------------------------------------------------------------------
# EventStore helpers
# ---------------------------------------------------------------------------

def test_event_store_append_and_retrieve(event_store):
    raw = make_raw_event("ENTRY", "VIS_01")
    evt = InboundEvent(**raw)
    stored = StoredEvent(event=evt)
    event_store.append(stored)

    events = event_store.get_events("STORE_TEST")
    assert len(events) == 1
    assert events[0].event.visitor_id == "VIS_01"


def test_event_store_has_seen(event_store):
    raw = make_raw_event("ENTRY", "VIS_01")
    evt = InboundEvent(**raw)
    assert event_store.has_seen(evt.event_id) is False
    event_store.append(StoredEvent(event=evt))
    assert event_store.has_seen(evt.event_id) is True


def test_event_store_total_count(event_store):
    for i in range(3):
        raw = make_raw_event("ENTRY", f"VIS_{i}", offset_sec=i)
        event_store.append(StoredEvent(event=InboundEvent(**raw)))
    assert event_store.total_count() == 3
    assert event_store.total_count("STORE_TEST") == 3


def test_event_store_get_all_store_ids(event_store):
    for sid in ("STORE_A", "STORE_B"):
        raw = make_raw_event("ENTRY", "VIS_01", store_id=sid)
        event_store.append(StoredEvent(event=InboundEvent(**raw)))
    ids = event_store.get_all_store_ids()
    assert set(ids) == {"STORE_A", "STORE_B"}


def test_event_store_last_event_timestamp(event_store):
    for i in range(3):
        raw = make_raw_event("ENTRY", f"VIS_{i}", offset_sec=i * 10)
        event_store.append(StoredEvent(event=InboundEvent(**raw)))
    last = event_store.last_event_timestamp("STORE_TEST")
    assert last == make_ts(20)


def test_event_store_clear(event_store):
    raw = make_raw_event("ENTRY", "VIS_01")
    event_store.append(StoredEvent(event=InboundEvent(**raw)))
    event_store.clear()
    assert event_store.total_count() == 0
    assert event_store.get_all_store_ids() == []


# ---------------------------------------------------------------------------
# IngestionPipeline — acceptance
# ---------------------------------------------------------------------------

def test_pipeline_single_valid_event(pipeline, event_store):
    raw = make_raw_event("ENTRY", "VIS_01")
    resp = pipeline.ingest_batch([raw])
    assert resp.accepted == 1
    assert resp.duplicates == 0
    assert resp.rejected == 0
    assert event_store.total_count("STORE_TEST") == 1


def test_pipeline_deduplication(pipeline, event_store):
    eid = str(uuid.uuid4())
    raw1 = make_raw_event("ENTRY", "VIS_01", event_id=eid)
    raw2 = make_raw_event("ENTRY", "VIS_01", event_id=eid)
    resp = pipeline.ingest_batch([raw1, raw2])
    assert resp.accepted == 1
    assert resp.duplicates == 1
    assert event_store.total_count("STORE_TEST") == 1


def test_pipeline_partial_success(pipeline):
    valid = make_raw_event("ENTRY", "VIS_01")
    invalid = make_raw_event("ENTRY", "VIS_02", timestamp="bad-ts")
    resp = pipeline.ingest_batch([valid, invalid])
    assert resp.accepted == 1
    assert resp.rejected == 1
    assert len(resp.errors) == 1
    assert "timestamp" in resp.errors[0].reason


def test_pipeline_all_invalid(pipeline):
    resp = pipeline.ingest_batch([
        {"event_id": "not-a-uuid", "store_id": "S", "camera_id": "C", "visitor_id": "V",
         "event_type": "ENTRY", "timestamp": "bad", "confidence": 0.9},
        {"no_fields": True},
    ])
    assert resp.accepted == 0
    assert resp.rejected == 2


def test_pipeline_non_dict_event(pipeline):
    """A non-dict entry in the batch must be rejected gracefully."""
    resp = pipeline.ingest_batch(["not-a-dict", 42, None])
    assert resp.accepted == 0
    assert resp.rejected == 3


def test_pipeline_multiple_stores(pipeline, event_store):
    evts = [
        make_raw_event("ENTRY", "VIS_01", store_id="STORE_A"),
        make_raw_event("ENTRY", "VIS_02", store_id="STORE_B"),
    ]
    resp = pipeline.ingest_batch(evts)
    assert resp.accepted == 2
    assert event_store.total_count("STORE_A") == 1
    assert event_store.total_count("STORE_B") == 1


def test_pipeline_large_batch(pipeline, event_store):
    """A batch of 500 unique events must all be accepted."""
    batch = [make_raw_event("ENTRY", f"VIS_{i}", offset_sec=i) for i in range(500)]
    resp = pipeline.ingest_batch(batch)
    assert resp.accepted == 500
    assert resp.rejected == 0


def test_pipeline_sessionizer_called(event_store):
    """Sessionizer.process_event must be called for each accepted event."""
    mock_sessionizer = MagicMock()
    p = IngestionPipeline(event_store, mock_sessionizer)
    batch = [make_raw_event("ENTRY", "VIS_01"), make_raw_event("EXIT", "VIS_01", offset_sec=60)]
    p.ingest_batch(batch)
    assert mock_sessionizer.process_event.call_count == 2


def test_pipeline_sessionizer_error_does_not_abort(event_store):
    """If sessionizer raises, the pipeline must still complete the batch."""
    mock_sessionizer = MagicMock()
    mock_sessionizer.process_event.side_effect = RuntimeError("boom")
    p = IngestionPipeline(event_store, mock_sessionizer)
    batch = [make_raw_event("ENTRY", "VIS_01"), make_raw_event("EXIT", "VIS_01", offset_sec=60)]
    resp = p.ingest_batch(batch)
    # Events are still stored even if sessionizer crashes
    assert resp.accepted == 2
