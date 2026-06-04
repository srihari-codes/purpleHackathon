"""
tests/test_audit.py — Unit tests for AuditTimeline (app/audit.py).

12 tests covering:
  - record_session_open / close
  - record_reentry
  - record_zone (ZONE_ENTER, ZONE_EXIT, ZONE_DWELL)
  - record_queue (join + abandon)
  - record_detection
  - get_timeline / to_dict
  - explain() string output
  - records_by_category
  - clear()
"""

import pytest

from app.audit import AuditCategory, AuditRecord, AuditTimeline, VisitorAuditTimeline
from tests.conftest import make_ts


# ---------------------------------------------------------------------------
# 1. record_session_open
# ---------------------------------------------------------------------------

def test_record_session_open(audit):
    audit.record_session_open(
        visitor_id="VIS_01",
        session_id="SESS_AAA",
        store_id="STORE_TEST",
        timestamp=make_ts(0),
        confidence=0.9,
        is_staff=False,
    )
    tl = audit.get_timeline("VIS_01")
    assert tl is not None
    sessions = tl.records_by_category(AuditCategory.SESSION)
    assert len(sessions) == 1
    assert "VIS_01" in sessions[0].reason
    assert sessions[0].session_id == "SESS_AAA"


# ---------------------------------------------------------------------------
# 2. record_session_close
# ---------------------------------------------------------------------------

def test_record_session_close(audit):
    audit.record_session_close(
        visitor_id="VIS_01",
        session_id="SESS_AAA",
        timestamp=make_ts(60),
        confidence=0.88,
        duration_ms=60_000,
    )
    tl = audit.get_timeline("VIS_01")
    exits = tl.records_by_category(AuditCategory.EXIT)
    assert len(exits) == 1
    assert "60.0s" in exits[0].reason


# ---------------------------------------------------------------------------
# 3. record_reentry
# ---------------------------------------------------------------------------

def test_record_reentry(audit):
    audit.record_reentry(
        visitor_id="VIS_01",
        session_id="SESS_BBB",
        timestamp=make_ts(120),
        confidence=0.85,
        reentry_count=1,
        reid_score=0.92,
        camera_id="CAM_ENTRY_01",
    )
    tl = audit.get_timeline("VIS_01")
    reentries = tl.records_by_category(AuditCategory.REENTRY)
    assert len(reentries) == 1
    assert "0.92" in reentries[0].reason
    assert reentries[0].metadata["reentry_count"] == 1


# ---------------------------------------------------------------------------
# 4. record_zone
# ---------------------------------------------------------------------------

def test_record_zone_enter(audit):
    audit.record_zone("VIS_01", "SESS_01", "ZONE_ENTER", make_ts(10), 0.9, "ZONE_SKIN")
    tl = audit.get_timeline("VIS_01")
    zones = tl.records_by_category(AuditCategory.ZONE)
    assert any("Entered zone ZONE_SKIN" in r.reason for r in zones)


def test_record_zone_exit(audit):
    audit.record_zone("VIS_01", "SESS_01", "ZONE_EXIT", make_ts(30), 0.88, "ZONE_SKIN", dwell_ms=20_000)
    tl = audit.get_timeline("VIS_01")
    zones = tl.records_by_category(AuditCategory.ZONE)
    assert any("20.0s" in r.reason for r in zones)


def test_record_zone_dwell(audit):
    audit.record_zone("VIS_01", "SESS_01", "ZONE_DWELL", make_ts(40), 0.87, "ZONE_SKIN", dwell_ms=15_000)
    tl = audit.get_timeline("VIS_01")
    zones = tl.records_by_category(AuditCategory.ZONE)
    assert any("15.0s" in r.reason for r in zones)


# ---------------------------------------------------------------------------
# 5. record_queue
# ---------------------------------------------------------------------------

def test_record_queue_join(audit):
    audit.record_queue("VIS_01", "SESS_01", "BILLING_QUEUE_JOIN", make_ts(50), 0.9, queue_depth=3)
    tl = audit.get_timeline("VIS_01")
    queues = tl.records_by_category(AuditCategory.QUEUE)
    assert len(queues) == 1
    assert "depth=3" in queues[0].reason
    assert queues[0].metadata["queue_depth"] == 3


def test_record_queue_abandon(audit):
    audit.record_queue("VIS_01", "SESS_01", "BILLING_QUEUE_ABANDON", make_ts(60), 0.85, wait_ms=8000)
    tl = audit.get_timeline("VIS_01")
    queues = tl.records_by_category(AuditCategory.QUEUE)
    assert any("8.0s" in r.reason for r in queues)


# ---------------------------------------------------------------------------
# 6. record_detection
# ---------------------------------------------------------------------------

def test_record_detection(audit):
    audit.record_detection("VIS_01", "ENTRY", make_ts(0), 0.91, "CAM_ENTRY_01")
    tl = audit.get_timeline("VIS_01")
    detections = tl.records_by_category(AuditCategory.DETECTION)
    assert len(detections) == 1
    assert "CAM_ENTRY_01" in detections[0].reason


# ---------------------------------------------------------------------------
# 7. explain() and to_dict()
# ---------------------------------------------------------------------------

def test_explain_output(audit):
    audit.record_session_open("VIS_01", "SESS_01", "STORE_TEST", make_ts(0), 0.9, False)
    text = audit.explain("VIS_01")
    assert "VIS_01" in text
    assert "Audit Trail" in text


def test_to_dict_structure(audit):
    audit.record_session_open("VIS_01", "SESS_01", "STORE_TEST", make_ts(0), 0.9, False)
    d = audit.to_dict("VIS_01")
    assert d is not None
    assert d["visitor_id"] == "VIS_01"
    assert d["record_count"] == 1
    assert isinstance(d["records"], list)


def test_to_dict_unknown_visitor_returns_none(audit):
    assert audit.to_dict("UNKNOWN_VIS") is None


# ---------------------------------------------------------------------------
# 8. get_all_visitor_ids
# ---------------------------------------------------------------------------

def test_get_all_visitor_ids(audit):
    for vid in ("VIS_A", "VIS_B", "VIS_C"):
        audit.record_detection(vid, "ENTRY", make_ts(0), 0.9, "CAM_ENTRY_01")
    ids = audit.get_all_visitor_ids()
    assert set(ids) == {"VIS_A", "VIS_B", "VIS_C"}


# ---------------------------------------------------------------------------
# 9. clear()
# ---------------------------------------------------------------------------

def test_audit_clear(audit):
    audit.record_session_open("VIS_01", "SESS_01", "STORE_TEST", make_ts(0), 0.9, False)
    audit.clear()
    assert audit.get_timeline("VIS_01") is None
    assert audit.get_all_visitor_ids() == []
