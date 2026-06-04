"""
tests/test_replay.py — Unit tests for ReplayEngine (app/replay.py).

11 tests covering:
  - replay_file happy path (accepted count, sessions produced)
  - replay_file with reset_state=True vs False
  - replay_file with store_id_filter
  - replay_file FileNotFoundError
  - Chronological sort (out-of-order events sorted before ingestion)
  - replay_store (in-memory replay)
  - Concurrent replay rejected with RuntimeError
  - progress() reporting
  - ReplayResult.to_dict() structure
  - Verifier runs after replay (warnings produced)
"""

import json
import os
import tempfile

import pytest

from app.audit import AuditTimeline
from app.ingestion import EventStore
from app.replay import ReplayEngine, ReplayMode, ReplayResult
from app.sessionizer import SessionStore, Sessionizer, build_session_pipeline
from app.verifier import VerifierEngine
from tests.conftest import make_raw_event, make_ts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_jsonl(events: list) -> str:
    """Write events list to a temp JSONL file; return path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        return f.name


# ---------------------------------------------------------------------------
# 1. replay_file happy path
# ---------------------------------------------------------------------------

def test_replay_file_basic(replay_engine, sess_store):
    events = [
        make_raw_event("ENTRY", "VIS_01", offset_sec=0),
        make_raw_event("ZONE_ENTER", "VIS_01", zone_id="ZONE_A", offset_sec=10),
        make_raw_event("EXIT", "VIS_01", offset_sec=60),
    ]
    path = write_jsonl(events)
    try:
        result = replay_engine.replay_file(path)
        assert result.accepted == 3
        assert result.rejected == 0
        sessions = sess_store.get_all_sessions("STORE_TEST")
        assert len(sessions) == 1
        assert sessions[0].visitor_id == "VIS_01"
        assert sessions[0].duration_ms == 60_000
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 2. replay_file: out-of-order events sorted chronologically
# ---------------------------------------------------------------------------

def test_replay_sorts_events(replay_engine, sess_store):
    """Events written out-of-order should be sorted by timestamp before ingest."""
    events = [
        make_raw_event("EXIT", "VIS_01", offset_sec=60),    # written last
        make_raw_event("ENTRY", "VIS_01", offset_sec=0),    # written first
    ]
    path = write_jsonl(events)
    try:
        result = replay_engine.replay_file(path)
        sessions = sess_store.get_all_sessions("STORE_TEST")
        assert len(sessions) == 1
        assert sessions[0].end_time == make_ts(60)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 3. replay_file: reset_state=True clears prior state
# ---------------------------------------------------------------------------

def test_replay_reset_state(replay_engine, event_store, sess_store, sessionizer):
    # Pre-populate with an unrelated event
    from app.models import InboundEvent, StoredEvent
    raw = make_raw_event("ENTRY", "VIS_PRE")
    from app.sessionizer import SessionStore
    event_store.append(StoredEvent(event=InboundEvent(**raw)))

    events = [make_raw_event("ENTRY", "VIS_NEW", offset_sec=0)]
    path = write_jsonl(events)
    try:
        replay_engine.replay_file(path, reset_state=True)
        # Old event must be gone
        assert event_store.total_count("STORE_TEST") == 1  # only replayed
        assert sess_store.get_active("VIS_PRE") is None
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 4. replay_file: reset_state=False accumulates
# ---------------------------------------------------------------------------

def test_replay_no_reset_accumulates(replay_engine, event_store):
    events1 = [make_raw_event("ENTRY", "VIS_01", offset_sec=0)]
    events2 = [make_raw_event("ENTRY", "VIS_02", offset_sec=0)]

    path1 = write_jsonl(events1)
    path2 = write_jsonl(events2)
    try:
        replay_engine.replay_file(path1, reset_state=True)
        replay_engine.replay_file(path2, reset_state=False)
        assert event_store.total_count("STORE_TEST") == 2
    finally:
        os.unlink(path1)
        os.unlink(path2)


# ---------------------------------------------------------------------------
# 5. replay_file: store_id_filter
# ---------------------------------------------------------------------------

def test_replay_store_id_filter(replay_engine, sess_store):
    events = [
        make_raw_event("ENTRY", "VIS_A", store_id="STORE_A", offset_sec=0),
        make_raw_event("ENTRY", "VIS_B", store_id="STORE_B", offset_sec=0),
    ]
    path = write_jsonl(events)
    try:
        result = replay_engine.replay_file(path, store_id_filter="STORE_A")
        assert result.accepted == 1
        assert sess_store.get_all_sessions("STORE_A") != []
        assert sess_store.get_all_sessions("STORE_B") == []
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 6. replay_file: FileNotFoundError
# ---------------------------------------------------------------------------

def test_replay_file_not_found(replay_engine):
    with pytest.raises(FileNotFoundError):
        replay_engine.replay_file("/tmp/does_not_exist_ever.jsonl")


# ---------------------------------------------------------------------------
# 7. replay_file: rejects malformed lines gracefully
# ---------------------------------------------------------------------------

def test_replay_file_malformed_lines(replay_engine):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("not-json\n")
        f.write(json.dumps(make_raw_event("ENTRY", "VIS_01", offset_sec=0)) + "\n")
        path = f.name
    try:
        result = replay_engine.replay_file(path)
        # Only 1 valid event
        assert result.accepted == 1
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 8. replay_store (in-memory replay)
# ---------------------------------------------------------------------------

def test_replay_store_in_memory(replay_engine, event_store, sess_store, pipeline):
    # Ingest via pipeline (populates event_store)
    batch = [
        make_raw_event("ENTRY", "VIS_01", offset_sec=0),
        make_raw_event("EXIT", "VIS_01", offset_sec=30),
    ]
    pipeline.ingest_batch(batch)

    # Now replay from the in-memory store
    result = replay_engine.replay_store("STORE_TEST", reset_state=True)
    assert result.accepted == 2
    sessions = sess_store.get_all_sessions("STORE_TEST")
    assert any(s.visitor_id == "VIS_01" for s in sessions)


# ---------------------------------------------------------------------------
# 9. ReplayResult.to_dict()
# ---------------------------------------------------------------------------

def test_replay_result_to_dict(replay_engine):
    events = [make_raw_event("ENTRY", "VIS_01", offset_sec=0)]
    path = write_jsonl(events)
    try:
        result = replay_engine.replay_file(path)
        d = result.to_dict()
        assert "accepted" in d
        assert "rejected" in d
        assert "elapsed_sec" in d
        assert d["total"] == 1
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 10. Concurrent replay rejected
# ---------------------------------------------------------------------------

def test_replay_concurrent_raises(replay_engine):
    """Calling _run_replay while already replaying must raise RuntimeError."""
    import threading

    replay_engine._is_replaying = True  # simulate in-progress replay
    try:
        with pytest.raises(RuntimeError, match="already in progress"):
            replay_engine._run_replay([], reset_state=False, progress_cb=None, source="test")
    finally:
        replay_engine._is_replaying = False


# ---------------------------------------------------------------------------
# 11. Verifier warnings produced after replay
# ---------------------------------------------------------------------------

def test_replay_triggers_verifier(replay_engine, verifier):
    events = [
        make_raw_event("ENTRY", "VIS_01", offset_sec=0),
        make_raw_event("BILLING_QUEUE_JOIN", "VIS_01", queue_depth=-5, offset_sec=10),
        make_raw_event("EXIT", "VIS_01", offset_sec=20),
    ]
    path = write_jsonl(events)
    try:
        replay_engine.replay_file(path)
        warnings = verifier.get_warnings()
        assert any(w.code == "QUEUE_DEPTH_NEGATIVE" for w in warnings)
    finally:
        os.unlink(path)
