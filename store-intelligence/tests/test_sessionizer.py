"""
tests/test_sessionizer.py — Unit tests for Sessionizer and SessionStore (app/sessionizer.py).

22 tests covering:
  - Every event type handler (ENTRY, EXIT, REENTRY, ZONE_*, BILLING_*)
  - Orphaned session handling (double ENTRY without EXIT)
  - EXIT without prior ENTRY (stub session)
  - Out-of-order ZONE_ENTER before ENTRY (implicit session)
  - REENTRY: count increment, new session when no active
  - Staff flag propagation
  - Rolling average confidence
  - SessionStore CRUD
  - Two independent visitors
"""

import pytest

from app.models import EventType, InboundEvent, VisitorSession
from app.sessionizer import SessionStore, Sessionizer, build_session_pipeline
from tests.conftest import make_raw_event, make_ts


# ---------------------------------------------------------------------------
# 1. ENTRY
# ---------------------------------------------------------------------------

def test_entry_opens_session(sess_store, sessionizer):
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01")))
    sess = sess_store.get_active("VIS_01")
    assert sess is not None
    assert sess.visitor_id == "VIS_01"
    assert sess.is_active is True
    assert sess.store_id == "STORE_TEST"


def test_entry_sets_start_time(sess_store, sessionizer):
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01", offset_sec=0)))
    sess = sess_store.get_active("VIS_01")
    assert sess.start_time == make_ts(0)


def test_entry_staff_flag_propagated(sess_store, sessionizer):
    sessionizer.process_event(
        InboundEvent(**make_raw_event("ENTRY", "VIS_STAFF", is_staff=True))
    )
    sess = sess_store.get_active("VIS_STAFF")
    assert sess.is_staff is True


def test_double_entry_closes_orphan(sess_store, sessionizer):
    """Second ENTRY without EXIT must close the orphaned session first."""
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01", offset_sec=0)))
    first_sess_id = sess_store.get_active("VIS_01").session_id

    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01", offset_sec=30)))
    second_sess = sess_store.get_active("VIS_01")

    assert second_sess.session_id != first_sess_id  # new session opened
    assert second_sess.is_active is True


# ---------------------------------------------------------------------------
# 2. EXIT
# ---------------------------------------------------------------------------

def test_exit_closes_session(sess_store, sessionizer):
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01", offset_sec=0)))
    sessionizer.process_event(InboundEvent(**make_raw_event("EXIT", "VIS_01", offset_sec=60)))
    assert sess_store.get_active("VIS_01") is None
    sessions = sess_store.get_all_sessions("STORE_TEST")
    assert len(sessions) == 1
    assert sessions[0].end_time == make_ts(60)
    assert sessions[0].duration_ms == 60_000


def test_exit_without_entry_creates_stub(sess_store, sessionizer):
    """EXIT arriving before ENTRY must create a stub closed session."""
    sessionizer.process_event(InboundEvent(**make_raw_event("EXIT", "VIS_GHOST", offset_sec=0)))
    sessions = sess_store.get_all_sessions("STORE_TEST")
    assert any(s.visitor_id == "VIS_GHOST" for s in sessions)


# ---------------------------------------------------------------------------
# 3. ZONE events
# ---------------------------------------------------------------------------

def test_zone_enter_adds_to_zones_visited(sess_store, sessionizer):
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01", offset_sec=0)))
    sessionizer.process_event(
        InboundEvent(**make_raw_event("ZONE_ENTER", "VIS_01", zone_id="ZONE_SKIN", offset_sec=10))
    )
    sess = sess_store.get_active("VIS_01")
    assert "ZONE_SKIN" in sess.zones_visited


def test_zone_dwell_accumulates(sess_store, sessionizer):
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01", offset_sec=0)))
    sessionizer.process_event(
        InboundEvent(**make_raw_event("ZONE_DWELL", "VIS_01", zone_id="ZONE_SKIN", dwell_ms=10_000, offset_sec=20))
    )
    sessionizer.process_event(
        InboundEvent(**make_raw_event("ZONE_DWELL", "VIS_01", zone_id="ZONE_SKIN", dwell_ms=5_000, offset_sec=40))
    )
    sess = sess_store.get_active("VIS_01")
    assert sess.dwell_per_zone["ZONE_SKIN"] == 15_000


def test_zone_exit_adds_dwell(sess_store, sessionizer):
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01", offset_sec=0)))
    sessionizer.process_event(
        InboundEvent(**make_raw_event("ZONE_EXIT", "VIS_01", zone_id="ZONE_A", dwell_ms=8_000, offset_sec=30))
    )
    sess = sess_store.get_active("VIS_01")
    assert sess.dwell_per_zone.get("ZONE_A", 0) == 8_000


def test_zone_event_before_entry_creates_implicit_session(sess_store, sessionizer):
    """ZONE_ENTER arriving before ENTRY must implicitly open a session."""
    sessionizer.process_event(
        InboundEvent(**make_raw_event("ZONE_ENTER", "VIS_LATE", zone_id="ZONE_A", offset_sec=0))
    )
    sess = sess_store.get_active("VIS_LATE")
    assert sess is not None
    assert "ZONE_A" in sess.zones_visited


def test_multiple_zones_ordered(sess_store, sessionizer):
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01", offset_sec=0)))
    for i, zone in enumerate(["ZONE_A", "ZONE_B", "ZONE_C"], start=1):
        sessionizer.process_event(
            InboundEvent(**make_raw_event("ZONE_ENTER", "VIS_01", zone_id=zone, offset_sec=i * 10))
        )
    sess = sess_store.get_active("VIS_01")
    assert sess.zones_visited == ["ZONE_A", "ZONE_B", "ZONE_C"]


# ---------------------------------------------------------------------------
# 4. BILLING_QUEUE events
# ---------------------------------------------------------------------------

def test_billing_queue_join_sets_purchase_candidate(sess_store, sessionizer):
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01", offset_sec=0)))
    sessionizer.process_event(
        InboundEvent(**make_raw_event("BILLING_QUEUE_JOIN", "VIS_01", queue_depth=3, offset_sec=30))
    )
    sess = sess_store.get_active("VIS_01")
    assert sess.purchase_candidate is True
    assert len(sess.queue_events) == 1
    assert sess.queue_events[0].event_type == EventType.BILLING_QUEUE_JOIN
    assert sess.queue_events[0].queue_depth == 3


def test_billing_queue_abandon_recorded(sess_store, sessionizer):
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01", offset_sec=0)))
    sessionizer.process_event(
        InboundEvent(**make_raw_event("BILLING_QUEUE_JOIN", "VIS_01", queue_depth=2, offset_sec=10))
    )
    sessionizer.process_event(
        InboundEvent(**make_raw_event("BILLING_QUEUE_ABANDON", "VIS_01", wait_duration_ms=5000, offset_sec=20))
    )
    sess = sess_store.get_active("VIS_01")
    assert len(sess.queue_events) == 2
    assert sess.queue_events[1].event_type == EventType.BILLING_QUEUE_ABANDON
    assert sess.queue_events[1].wait_ms == 5000


# ---------------------------------------------------------------------------
# 5. REENTRY
# ---------------------------------------------------------------------------

def test_reentry_increments_count(sess_store, sessionizer):
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01", offset_sec=0)))
    sessionizer.process_event(InboundEvent(**make_raw_event("EXIT", "VIS_01", offset_sec=60)))
    sessionizer.process_event(
        InboundEvent(**make_raw_event("REENTRY", "VIS_01", offset_sec=120, reid_score=0.9))
    )
    sess = sess_store.get_active("VIS_01")
    assert sess is not None
    assert sess.reentry_count == 1


def test_reentry_without_exit_increments_existing_session(sess_store, sessionizer):
    """REENTRY while session is still open should update reentry_count in-place."""
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01", offset_sec=0)))
    sessionizer.process_event(
        InboundEvent(**make_raw_event("REENTRY", "VIS_01", offset_sec=30, reid_score=0.85))
    )
    sess = sess_store.get_active("VIS_01")
    assert sess.reentry_count == 1


def test_reentry_without_prior_session_creates_new(sess_store, sessionizer):
    """REENTRY with no open session must open a new continuation session."""
    sessionizer.process_event(
        InboundEvent(**make_raw_event("REENTRY", "VIS_ORPHAN", offset_sec=0, reid_score=0.8))
    )
    sess = sess_store.get_active("VIS_ORPHAN")
    assert sess is not None
    assert sess.reentry_count == 1


# ---------------------------------------------------------------------------
# 6. Rolling average confidence
# ---------------------------------------------------------------------------

def test_rolling_avg_confidence(sess_store, sessionizer):
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01", confidence=1.0, offset_sec=0)))
    sessionizer.process_event(
        InboundEvent(**make_raw_event("ZONE_ENTER", "VIS_01", zone_id="Z1", confidence=0.5, offset_sec=10))
    )
    sess = sess_store.get_active("VIS_01")
    # After 2 events: rolling avg should be between 0.5 and 1.0
    assert 0.5 <= sess.avg_confidence <= 1.0


# ---------------------------------------------------------------------------
# 7. Two independent visitors
# ---------------------------------------------------------------------------

def test_two_visitors_independent(sess_store, sessionizer):
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_A", offset_sec=0)))
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_B", offset_sec=0)))
    sessionizer.process_event(InboundEvent(**make_raw_event("EXIT", "VIS_A", offset_sec=30)))

    assert sess_store.get_active("VIS_A") is None
    assert sess_store.get_active("VIS_B") is not None
    assert len(sess_store.get_all_sessions("STORE_TEST")) == 2


# ---------------------------------------------------------------------------
# 8. SessionStore helpers
# ---------------------------------------------------------------------------

def test_session_store_get_active_count(sess_store, sessionizer):
    for vid in ("VIS_A", "VIS_B", "VIS_C"):
        sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", vid, offset_sec=0)))
    assert sess_store.get_active_count("STORE_TEST") == 3
    assert sess_store.get_active_count() == 3


def test_session_store_get_all_store_ids(sess_store, sessionizer):
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01", store_id="STORE_A")))
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_02", store_id="STORE_B")))
    ids = sess_store.get_all_store_ids()
    assert "STORE_A" in ids
    assert "STORE_B" in ids


def test_session_store_clear(sess_store, sessionizer):
    sessionizer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01")))
    sess_store.clear()
    assert sess_store.get_active("VIS_01") is None
    assert sess_store.get_all_sessions("STORE_TEST") == []
