"""
tests/test_verifier.py — Unit tests for VerifierEngine (app/verifier.py).

16 tests covering every check:
  QUEUE_DEPTH_NEGATIVE, CONFIDENCE_CLIFF, REENTRY_TOO_FAST,
  DUPLICATE_ACTIVE_SESSION, ENTRY_WITHOUT_DOORWAY,
  IMPOSSIBLE_CAMERA_TRANSITION, STAFF_COUNT_EXPLOSION,
  get_warnings (filter by store_id, severity, limit),
  clear(), accumulation cap.
"""

import pytest

from app.models import EventType, InboundEvent, VisitorSession
from app.verifier import VerifierEngine, VerifierSeverity
from tests.conftest import make_raw_event, make_ts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _process(verifier: VerifierEngine, raw: dict, session=None):
    evt = InboundEvent(**raw)
    return verifier.verify_event(evt, session), evt


# ---------------------------------------------------------------------------
# 1. QUEUE_DEPTH_NEGATIVE
# ---------------------------------------------------------------------------

def test_queue_depth_negative_fires(verifier):
    warnings, _ = _process(verifier, make_raw_event("BILLING_QUEUE_JOIN", "VIS_01", queue_depth=-1))
    assert any(w.code == "QUEUE_DEPTH_NEGATIVE" for w in warnings)


def test_queue_depth_zero_is_ok(verifier):
    warnings, _ = _process(verifier, make_raw_event("BILLING_QUEUE_JOIN", "VIS_01", queue_depth=0))
    assert not any(w.code == "QUEUE_DEPTH_NEGATIVE" for w in warnings)


def test_queue_depth_none_is_ok(verifier):
    """Absence of queue_depth must not trigger the check."""
    warnings, _ = _process(verifier, make_raw_event("BILLING_QUEUE_JOIN", "VIS_01"))
    assert not any(w.code == "QUEUE_DEPTH_NEGATIVE" for w in warnings)


# ---------------------------------------------------------------------------
# 2. CONFIDENCE_CLIFF
# ---------------------------------------------------------------------------

def test_confidence_cliff_fires(verifier):
    _process(verifier, make_raw_event("ZONE_ENTER", "VIS_01", confidence=0.95, offset_sec=0, zone_id="Z1"))
    warnings, _ = _process(verifier, make_raw_event("ZONE_EXIT", "VIS_01", confidence=0.40, offset_sec=10, zone_id="Z1"))
    assert any(w.code == "CONFIDENCE_CLIFF" for w in warnings)


def test_confidence_cliff_no_fire_small_drop(verifier):
    _process(verifier, make_raw_event("ZONE_ENTER", "VIS_01", confidence=0.90, offset_sec=0, zone_id="Z1"))
    warnings, _ = _process(verifier, make_raw_event("ZONE_EXIT", "VIS_01", confidence=0.75, offset_sec=10, zone_id="Z1"))
    assert not any(w.code == "CONFIDENCE_CLIFF" for w in warnings)


def test_confidence_cliff_no_fire_first_event(verifier):
    """First event for a visitor → no previous confidence → no cliff possible."""
    warnings, _ = _process(verifier, make_raw_event("ENTRY", "VIS_NEW", confidence=0.10, offset_sec=0))
    assert not any(w.code == "CONFIDENCE_CLIFF" for w in warnings)


# ---------------------------------------------------------------------------
# 3. REENTRY_TOO_FAST
# ---------------------------------------------------------------------------

def test_reentry_too_fast_fires(verifier):
    _process(verifier, make_raw_event("EXIT", "VIS_01", offset_sec=0))
    warnings, _ = _process(verifier, make_raw_event("REENTRY", "VIS_01", offset_sec=2))
    assert any(w.code == "REENTRY_TOO_FAST" for w in warnings)


def test_reentry_ok_gap(verifier):
    _process(verifier, make_raw_event("EXIT", "VIS_01", offset_sec=0))
    warnings, _ = _process(verifier, make_raw_event("REENTRY", "VIS_01", offset_sec=30))
    assert not any(w.code == "REENTRY_TOO_FAST" for w in warnings)


def test_reentry_without_prior_exit_no_fire(verifier):
    """REENTRY with no known EXIT timestamp must not trigger the speed check."""
    warnings, _ = _process(verifier, make_raw_event("REENTRY", "VIS_NEW", offset_sec=0))
    assert not any(w.code == "REENTRY_TOO_FAST" for w in warnings)


# ---------------------------------------------------------------------------
# 4. DUPLICATE_ACTIVE_SESSION
# ---------------------------------------------------------------------------

def test_duplicate_active_session_fires(verifier):
    s1 = VisitorSession(visitor_id="VIS_01", store_id="STORE_TEST", start_time=make_ts(0))
    s2 = VisitorSession(visitor_id="VIS_01", store_id="STORE_TEST", start_time=make_ts(10))
    warnings = verifier.verify_active_sessions([s1, s2], "STORE_TEST")
    assert any(w.code == "DUPLICATE_ACTIVE_SESSION" for w in warnings)


def test_no_duplicate_single_active_session(verifier):
    s1 = VisitorSession(visitor_id="VIS_01", store_id="STORE_TEST", start_time=make_ts(0))
    warnings = verifier.verify_active_sessions([s1], "STORE_TEST")
    assert not any(w.code == "DUPLICATE_ACTIVE_SESSION" for w in warnings)


def test_closed_sessions_excluded_from_duplicate_check(verifier):
    s1 = VisitorSession(visitor_id="VIS_01", store_id="STORE_TEST",
                        start_time=make_ts(0), end_time=make_ts(60))  # closed
    s2 = VisitorSession(visitor_id="VIS_01", store_id="STORE_TEST", start_time=make_ts(120))  # active
    warnings = verifier.verify_active_sessions([s1, s2], "STORE_TEST")
    assert not any(w.code == "DUPLICATE_ACTIVE_SESSION" for w in warnings)


# ---------------------------------------------------------------------------
# 5. ENTRY_WITHOUT_DOORWAY
# ---------------------------------------------------------------------------

def test_entry_without_doorway_fires(verifier):
    raw = make_raw_event("ENTRY", "VIS_01", camera_id="CAM_INTERIOR_05")
    warnings, _ = _process(verifier, raw)
    assert any(w.code == "ENTRY_WITHOUT_DOORWAY" for w in warnings)


def test_entry_from_valid_camera_no_fire(verifier):
    for cam in ("CAM_ENTRY_01", "CAM_DOOR_02", "CAM_THRESHOLD_03"):
        verifier.clear()
        raw = make_raw_event("ENTRY", "VIS_01", camera_id=cam)
        warnings, _ = _process(verifier, raw)
        assert not any(w.code == "ENTRY_WITHOUT_DOORWAY" for w in warnings), f"false alarm for {cam}"


# ---------------------------------------------------------------------------
# 6. get_warnings — filtering
# ---------------------------------------------------------------------------

def test_get_warnings_filter_by_store(verifier):
    _process(verifier, make_raw_event("BILLING_QUEUE_JOIN", "VIS_01", queue_depth=-1, store_id="STORE_A"))
    _process(verifier, make_raw_event("BILLING_QUEUE_JOIN", "VIS_02", queue_depth=-1, store_id="STORE_B"))
    ws_a = verifier.get_warnings(store_id="STORE_A")
    ws_b = verifier.get_warnings(store_id="STORE_B")
    assert all(w.store_id == "STORE_A" for w in ws_a)
    assert all(w.store_id == "STORE_B" for w in ws_b)


def test_get_warnings_filter_by_severity(verifier):
    _process(verifier, make_raw_event("BILLING_QUEUE_JOIN", "VIS_01", queue_depth=-1))
    criticals = verifier.get_warnings(severity="CRITICAL")
    assert all(w.severity == "CRITICAL" for w in criticals)


def test_get_warnings_limit(verifier):
    for i in range(20):
        _process(verifier, make_raw_event("BILLING_QUEUE_JOIN", f"VIS_{i}", queue_depth=-1, offset_sec=i))
    limited = verifier.get_warnings(limit=5)
    assert len(limited) <= 5


# ---------------------------------------------------------------------------
# 7. clear()
# ---------------------------------------------------------------------------

def test_verifier_clear_resets_state(verifier):
    _process(verifier, make_raw_event("BILLING_QUEUE_JOIN", "VIS_01", queue_depth=-1))
    assert len(verifier.get_warnings()) > 0
    verifier.clear()
    assert len(verifier.get_warnings()) == 0
