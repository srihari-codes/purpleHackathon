"""
tests/test_projections.py — Unit tests for all 5 projections (app/projections.py).

23 tests covering:
  - MetricsProjection: empty, unique visitor dedup, conversion rate, zone dwell,
    queue stats, abandonment rate, staff exclusion, active session count
  - FunnelProjection: all stages, pct/drop-off math, reentry dedup, empty input
  - HeatmapProjection: data_confidence flag, zone scores 0-100, empty sessions
  - AnomalyProjection: queue spike WARN/CRITICAL, conversion drop, dead zone
  - HealthProjection: OK status, DEGRADED (stale feed), warning text
"""

import datetime as dt
from datetime import timedelta, timezone
from unittest.mock import MagicMock

import pytest

from app.models import EventType, QueueEvent, VisitorSession
from app.projections import (
    AnomalyProjection,
    FunnelProjection,
    HeatmapProjection,
    HealthProjection,
    MetricsProjection,
)
from tests.conftest import make_ts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_customer_session(
    visitor_id: str,
    *,
    store_id: str = "STORE_TEST",
    start_offset: int = 0,
    end_offset: int = None,
    zones: list = None,
    dwell: dict = None,
    purchase_candidate: bool = False,
    is_staff: bool = False,
    queue_events: list = None,
    reentry_count: int = 0,
) -> VisitorSession:
    s = VisitorSession(
        visitor_id=visitor_id,
        store_id=store_id,
        start_time=make_ts(start_offset),
        end_time=make_ts(end_offset) if end_offset is not None else None,
        is_staff=is_staff,
        zones_visited=zones or [],
        dwell_per_zone=dwell or {},
        purchase_candidate=purchase_candidate,
        reentry_count=reentry_count,
        queue_events=queue_events or [],
    )
    return s


# ---------------------------------------------------------------------------
# MetricsProjection
# ---------------------------------------------------------------------------

def test_metrics_empty_sessions():
    result = MetricsProjection.build([], "STORE_TEST")
    assert result["unique_visitors"] == 0
    assert result["conversion_rate"] == 0.0
    assert result["total_sessions"] == 0
    assert result["abandonment_rate"] == 0.0


def test_metrics_unique_visitor_dedup():
    """Two sessions for the same visitor_id → unique_visitors == 1."""
    s1 = make_customer_session("VIS_01")
    s2 = make_customer_session("VIS_01")  # re-entry session
    result = MetricsProjection.build([s1, s2], "STORE_TEST")
    assert result["unique_visitors"] == 1


def test_metrics_conversion_rate_zero():
    s = make_customer_session("VIS_01", purchase_candidate=False)
    result = MetricsProjection.build([s], "STORE_TEST")
    assert result["conversion_rate"] == 0.0
    assert result["converted_count"] == 0


def test_metrics_conversion_rate_full():
    sessions = [make_customer_session(f"VIS_{i}", purchase_candidate=True) for i in range(3)]
    result = MetricsProjection.build(sessions, "STORE_TEST")
    assert result["conversion_rate"] == 1.0
    assert result["converted_count"] == 3


def test_metrics_conversion_rate_partial():
    sessions = [
        make_customer_session("VIS_01", purchase_candidate=True),
        make_customer_session("VIS_02", purchase_candidate=False),
        make_customer_session("VIS_03", purchase_candidate=False),
    ]
    result = MetricsProjection.build(sessions, "STORE_TEST")
    assert result["conversion_rate"] == pytest.approx(1 / 3, rel=1e-3)


def test_metrics_avg_dwell_per_zone():
    s1 = make_customer_session("VIS_01", dwell={"ZONE_A": 6000, "ZONE_B": 3000})
    s2 = make_customer_session("VIS_02", dwell={"ZONE_A": 2000})
    result = MetricsProjection.build([s1, s2], "STORE_TEST")
    assert result["avg_dwell_per_zone"]["ZONE_A"] == 4000  # (6000+2000)/2
    assert result["avg_dwell_per_zone"]["ZONE_B"] == 3000  # only s1


def test_metrics_staff_excluded():
    staff = make_customer_session("STAFF_01", is_staff=True, purchase_candidate=True)
    customer = make_customer_session("VIS_01", is_staff=False, purchase_candidate=False)
    result = MetricsProjection.build([staff, customer], "STORE_TEST")
    assert result["unique_visitors"] == 1  # staff not counted
    assert result["total_sessions"] == 1


def test_metrics_queue_depth():
    qe = QueueEvent(event_type=EventType.BILLING_QUEUE_JOIN, timestamp=make_ts(10), queue_depth=7)
    s = make_customer_session("VIS_01", queue_events=[qe])
    result = MetricsProjection.build([s], "STORE_TEST")
    assert result["queue_depth"] == 7
    assert result["current_queue_depth"] == 7


def test_metrics_abandonment_rate_one():
    qe_join = QueueEvent(event_type=EventType.BILLING_QUEUE_JOIN, timestamp=make_ts(0))
    qe_abandon = QueueEvent(event_type=EventType.BILLING_QUEUE_ABANDON, timestamp=make_ts(10))
    s = make_customer_session("VIS_01", queue_events=[qe_join, qe_abandon])
    result = MetricsProjection.build([s], "STORE_TEST")
    assert result["abandonment_rate"] == 1.0


def test_metrics_active_sessions_count():
    s_active = make_customer_session("VIS_01")        # no end_time → active
    s_closed = make_customer_session("VIS_02", end_offset=60)
    result = MetricsProjection.build([s_active, s_closed], "STORE_TEST")
    assert result["active_sessions"] == 1


# ---------------------------------------------------------------------------
# FunnelProjection
# ---------------------------------------------------------------------------

def test_funnel_empty():
    result = FunnelProjection.build([], "STORE_TEST")
    assert all(stage["count"] == 0 for stage in result["funnel"])


def test_funnel_all_stages():
    s = make_customer_session("VIS_01", zones=["ZONE_A"], purchase_candidate=True)
    result = FunnelProjection.build([s], "STORE_TEST")
    funnel = result["funnel"]
    assert funnel[0]["stage"] == "ENTRY"
    assert funnel[0]["count"] == 1
    assert funnel[1]["stage"] == "ZONE_VISIT"
    assert funnel[1]["count"] == 1
    assert funnel[2]["stage"] == "BILLING_QUEUE"
    assert funnel[2]["count"] == 1
    assert funnel[3]["stage"] == "PURCHASE"
    assert funnel[3]["count"] == 1


def test_funnel_pct_of_top():
    sessions = [
        make_customer_session("VIS_01", zones=["ZONE_A"], purchase_candidate=True),
        make_customer_session("VIS_02"),  # no zone visit, no purchase
    ]
    result = FunnelProjection.build(sessions, "STORE_TEST")
    funnel = result["funnel"]
    assert funnel[0]["pct_of_top"] == 100.0
    assert funnel[1]["pct_of_top"] == 50.0  # 1 out of 2 visited a zone


def test_funnel_reentry_dedup():
    """Two sessions for the same visitor must count as 1 funnel entry."""
    s1 = make_customer_session("VIS_01", zones=["ZONE_A"])
    s2 = make_customer_session("VIS_01", purchase_candidate=True)  # second session
    result = FunnelProjection.build([s1, s2], "STORE_TEST")
    assert result["funnel"][0]["count"] == 1  # still 1 unique visitor


def test_funnel_staff_excluded():
    staff = make_customer_session("STAFF_01", is_staff=True, purchase_candidate=True)
    result = FunnelProjection.build([staff], "STORE_TEST")
    assert result["funnel"][0]["count"] == 0


# ---------------------------------------------------------------------------
# HeatmapProjection
# ---------------------------------------------------------------------------

def test_heatmap_low_confidence_flag():
    sessions = [make_customer_session(f"VIS_{i}", zones=["Z1"]) for i in range(5)]
    result = HeatmapProjection.build(sessions, "STORE_TEST")
    assert result["data_confidence"] is False  # < 20 sessions


def test_heatmap_high_confidence_flag():
    sessions = [make_customer_session(f"VIS_{i}", zones=["Z1"]) for i in range(20)]
    result = HeatmapProjection.build(sessions, "STORE_TEST")
    assert result["data_confidence"] is True  # >= 20 sessions


def test_heatmap_empty_sessions():
    result = HeatmapProjection.build([], "STORE_TEST")
    assert result["zones"] == []
    assert result["session_count"] == 0


def test_heatmap_zone_scores_normalised():
    """The zone with the most visits should have freq_score == 100."""
    sessions = [make_customer_session(f"VIS_{i}", zones=["HOT_ZONE"]) for i in range(5)]
    sessions += [make_customer_session(f"VIS_C{i}", zones=["COLD_ZONE"]) for i in range(1)]
    result = HeatmapProjection.build(sessions, "STORE_TEST")
    hot = next(z for z in result["zones"] if z["zone_id"] == "HOT_ZONE")
    cold = next(z for z in result["zones"] if z["zone_id"] == "COLD_ZONE")
    assert hot["freq_score"] == 100
    assert cold["freq_score"] < 100


def test_heatmap_sorted_by_combined_score():
    sessions = [make_customer_session("VIS_01", zones=["HOT", "COLD"], dwell={"HOT": 50000, "COLD": 1000})]
    result = HeatmapProjection.build(sessions, "STORE_TEST")
    scores = [z["combined_score"] for z in result["zones"]]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# AnomalyProjection
# ---------------------------------------------------------------------------

def test_anomaly_queue_spike_warn():
    qe = QueueEvent(event_type=EventType.BILLING_QUEUE_JOIN, timestamp=make_ts(0), queue_depth=6)
    s = make_customer_session("VIS_01", queue_events=[qe])
    result = AnomalyProjection.build([s], "STORE_TEST", conversion_rate=0.5)
    types = [a["type"] for a in result["anomalies"]]
    assert "BILLING_QUEUE_SPIKE" in types


def test_anomaly_queue_spike_critical():
    qe = QueueEvent(event_type=EventType.BILLING_QUEUE_JOIN, timestamp=make_ts(0), queue_depth=10)
    s = make_customer_session("VIS_01", queue_events=[qe])
    result = AnomalyProjection.build([s], "STORE_TEST", conversion_rate=0.5)
    spike = next(a for a in result["anomalies"] if a["type"] == "BILLING_QUEUE_SPIKE")
    assert spike["severity"] == "CRITICAL"


def test_anomaly_no_spike_below_threshold():
    qe = QueueEvent(event_type=EventType.BILLING_QUEUE_JOIN, timestamp=make_ts(0), queue_depth=3)
    s = make_customer_session("VIS_01", queue_events=[qe])
    result = AnomalyProjection.build([s], "STORE_TEST")
    types = [a["type"] for a in result["anomalies"]]
    assert "BILLING_QUEUE_SPIKE" not in types


def test_anomaly_conversion_drop():
    result = AnomalyProjection.build(
        [], "STORE_TEST", conversion_rate=0.05, historical_avg_rate=0.30
    )
    types = [a["type"] for a in result["anomalies"]]
    assert "CONVERSION_DROP" in types


def test_anomaly_dead_zone():
    """A zone last seen > 30 min ago must surface a DEAD_ZONE anomaly."""
    old_time = dt.datetime.now(timezone.utc) - dt.timedelta(minutes=45)
    old_ts = old_time.isoformat().replace("+00:00", "Z")
    s = VisitorSession(
        visitor_id="VIS_01",
        store_id="STORE_TEST",
        start_time=old_ts,
        end_time=old_ts,
        zones_visited=["DEAD_ZONE"],
    )
    result = AnomalyProjection.build([s], "STORE_TEST")
    types = [a["type"] for a in result["anomalies"]]
    assert "DEAD_ZONE" in types


# ---------------------------------------------------------------------------
# HealthProjection
# ---------------------------------------------------------------------------

def test_health_ok_status(event_store, sess_store):
    from app.ingestion import EventStore
    from app.sessionizer import SessionStore

    # Fresh stores with a recent event — write directly
    from app.models import InboundEvent, StoredEvent
    from tests.conftest import make_raw_event
    raw = make_raw_event("ENTRY", "VIS_01")
    raw["timestamp"] = dt.datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    evt = InboundEvent(**raw)
    event_store.append(StoredEvent(event=evt))

    result = HealthProjection.build(event_store, sess_store, "2026-01-01T00:00:00Z")
    assert result["status"] == "OK"
    assert result["store_count"] == 1


def test_health_degraded_stale_feed(event_store, sess_store):
    from app.models import InboundEvent, StoredEvent
    from tests.conftest import make_raw_event
    # Ingest an event with a 20-minute-old timestamp
    stale = (dt.datetime.now(timezone.utc) - dt.timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
    raw = make_raw_event("ENTRY", "VIS_STALE")
    raw["timestamp"] = stale
    evt = InboundEvent(**raw)
    event_store.append(StoredEvent(event=evt))

    result = HealthProjection.build(event_store, sess_store, "2026-01-01T00:00:00Z")
    assert result["status"] == "DEGRADED"
    assert len(result["warnings"]) > 0
