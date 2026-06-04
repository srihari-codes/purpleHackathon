# PROMPT:
#   Generate a comprehensive pytest test suite for a retail store analytics FastAPI application
#   that processes CCTV detection events. The system: (1) ingests batches of structured events
#   (ENTRY, EXIT, ZONE_ENTER, ZONE_EXIT, ZONE_DWELL, BILLING_QUEUE_JOIN, BILLING_QUEUE_ABANDON,
#   REENTRY) via POST /events/ingest, (2) sessionises visitors — deduplicating re-entries so the
#   same physical person is never double-counted as a unique visitor, (3) correlates visitor
#   billing-zone dwell with POS transactions to compute conversion rate, (4) exposes
#   GET /stores/{id}/metrics, /funnel, /heatmap, /anomalies, and /health endpoints.
#   Tests must cover: event validation (malformed timestamps, empty visitor_id, future timestamps,
#   confidence out-of-range), idempotent ingest (same event_id twice → accepted=1/duplicates=1),
#   partial success on mixed valid/invalid batch, session lifecycle (ENTRY→ZONE_ENTER→ZONE_DWELL→
#   BILLING_QUEUE_JOIN→EXIT), out-of-order session creation (ZONE_ENTER before ENTRY),
#   re-entry handling (ENTRY→EXIT→REENTRY), calibration engine rolling window, verifier engine
#   violations (QUEUE_DEPTH_NEGATIVE, CONFIDENCE_CLIFF, REENTRY_TOO_FAST,
#   DUPLICATE_ACTIVE_SESSION), replay engine reproducibility, POS correlation (5-minute window),
#   funnel projection, full API integration flow, and edge cases: empty store → 404, all-staff
#   clip → 0 unique visitors, zero purchases → 0.0 conversion rate, re-entry funnel dedup,
#   abandonment rate, verifier during sessionisation, replay + verifier consistency, and
#   Purplle-specific production CSV format with non-standard column headers.
#
# CHANGES MADE:
#   1. Hardened the timestamp validation test to assert the exact error message substring
#      ("not a valid ISO-8601 datetime") rather than a generic ValueError, catching regressions
#      in the Pydantic validator message format.
#   2. Added `test_brigade_bangalore_csv_format` — the AI generated only a generic CSV test;
#      replaced it with the actual Purplle production column schema (order_id, coupon_code,
#      offer_name, GMV, NMV, total_amount, etc.) to validate the production parser path.
#   3. Added `test_visitor_centric_conversion_and_funnel_reentry` — the AI's funnel test only
#      checked single-session visitors; this test verifies that a visitor who re-enters and joins
#      the billing queue in their second session is counted as 1 unique converted visitor, not 2.
#   4. Added `test_abandonment_rate_multiple_events` — the AI omitted BILLING_QUEUE_ABANDON
#      as a standalone edge case; added explicit test with join + abandon in same session and
#      assertion that abandonment_rate == 1.0.
#   5. Added `test_event_verifier_triggered_during_sessionization` — the AI generated verifier
#      unit tests but did not test that verifier warnings are produced during the full API ingest
#      flow; added end-to-end check via POST /events/ingest with a negative queue_depth event.
#   6. Added `test_replay_engine_verifier_and_correlation_consistency` — the AI missed the
#      POST /stores/{id}/replay endpoint integration test; added to verify that the replay engine
#      correctly surfaces verifier warnings and that /metrics responds correctly after a replay.
#   7. Removed AI-generated happy-path-only heatmap test that asserted `zones` was non-empty
#      for a single session — with < 20 sessions data_confidence is False and the zone list
#      may be empty if no ZONE_ENTER events were ingested; replaced with the data_confidence
#      flag assertion in `test_api_full_flow`.

import os
import json
import uuid
import tempfile
import datetime as dt
from datetime import timezone
import pytest
from fastapi.testclient import TestClient

from app.models import EventType, InboundEvent, VisitorSession, QueueEvent
from app.ingestion import EventStore, IngestionPipeline
from app.sessionizer import SessionStore, Sessionizer, build_session_pipeline
from app.audit import AuditTimeline, AuditCategory
from app.calibration import CalibrationEngine, StoreCalibration
from app.replay import ReplayEngine, ReplayMode
from app.verifier import VerifierEngine, VerifierSeverity, VerifierWarning
from app.correlation import CorrelationEngine, POSTransaction
from app.projections import (
    MetricsProjection,
    FunnelProjection,
    HeatmapProjection,
    AnomalyProjection,
    HealthProjection
)
from app.main import app, _event_store, _sess_store, _sessionizer, _audit, _calibration, _verifier, _correlation, _replay

client = TestClient(app)

# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def make_ts(offset_sec=0):
    base = dt.datetime(2026, 3, 3, 14, 0, 0, tzinfo=timezone.utc)
    return (base + dt.timedelta(seconds=offset_sec)).isoformat().replace("+00:00", "Z")

def make_raw_event(
    etype: str,
    visitor_id: str,
    event_id: str = None,
    store_id: str = "STORE_BLR_002",
    camera_id: str = "CAM_ENTRY_01",
    timestamp: str = None,
    zone_id: str = None,
    dwell_ms: int = 0,
    is_staff: bool = False,
    confidence: float = 0.9,
    queue_depth: int = None,
    reid_score: float = None,
    wait_duration_ms: int = None,
    offset_sec: int = 0,
):
    if not event_id:
        event_id = str(uuid.uuid4())
    if not timestamp:
        timestamp = make_ts(offset_sec)
    
    meta = {}
    if queue_depth is not None:
        meta["queue_depth"] = queue_depth
    if reid_score is not None:
        meta["reid_score"] = reid_score
    if wait_duration_ms is not None:
        meta["wait_duration_ms"] = wait_duration_ms

    return {
        "event_id": event_id,
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": etype,
        "timestamp": timestamp,
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": confidence,
        "metadata": meta
    }


# ---------------------------------------------------------------------------
# 1. Validation Tests
# ---------------------------------------------------------------------------

def test_event_validation_rules():
    # Valid event validation
    valid_raw = make_raw_event("ENTRY", "VIS_01")
    event = InboundEvent(**valid_raw)
    assert event.visitor_id == "VIS_01"
    assert event.parsed_timestamp().year == 2026

    # Malformed timestamp
    invalid_ts = valid_raw.copy()
    invalid_ts["timestamp"] = "invalid-date-format"
    with pytest.raises(ValueError, match="not a valid ISO-8601 datetime"):
        InboundEvent(**invalid_ts)

    # Empty string validation
    empty_vid = valid_raw.copy()
    empty_vid["visitor_id"] = "   "
    with pytest.raises(ValueError, match="visitor_id must be a non-empty string"):
        InboundEvent(**empty_vid)

    # Future timestamp (skew check)
    future_raw = valid_raw.copy()
    future_raw["timestamp"] = (dt.datetime.now(timezone.utc) + dt.timedelta(minutes=5)).isoformat()
    with pytest.raises(ValueError, match="max 60 s clock skew allowed"):
        InboundEvent(**future_raw)

    # Invalid confidence range
    invalid_conf = valid_raw.copy()
    invalid_conf["confidence"] = 1.5
    with pytest.raises(ValueError):
        InboundEvent(**invalid_conf)

# ---------------------------------------------------------------------------
# 2. Ingestion & Deduplication Tests
# ---------------------------------------------------------------------------

def test_pipeline_deduplication():
    es = EventStore()
    s_store, s_izer = build_session_pipeline()
    pipeline = IngestionPipeline(es, s_izer)

    event_id = str(uuid.uuid4())
    evt1 = make_raw_event("ENTRY", "VIS_01", event_id=event_id)
    evt2 = make_raw_event("ENTRY", "VIS_01", event_id=event_id)  # duplicate

    resp = pipeline.ingest_batch([evt1, evt2])
    assert resp.accepted == 1
    assert resp.duplicates == 1
    assert resp.rejected == 0

    assert len(es.get_events("STORE_BLR_002")) == 1

def test_partial_success_handling():
    es = EventStore()
    s_store, s_izer = build_session_pipeline()
    pipeline = IngestionPipeline(es, s_izer)

    evt_valid = make_raw_event("ENTRY", "VIS_01")
    evt_invalid = make_raw_event("ENTRY", "VIS_02", timestamp="bad-ts")

    resp = pipeline.ingest_batch([evt_valid, evt_invalid])
    assert resp.accepted == 1
    assert resp.duplicates == 0
    assert resp.rejected == 1
    assert len(resp.errors) == 1
    assert "timestamp" in resp.errors[0].reason

# ---------------------------------------------------------------------------
# 3. Sessionization Logic Tests
# ---------------------------------------------------------------------------

def test_session_lifecycle_happy_path():
    s_store, s_izer = build_session_pipeline()
    
    # 1. ENTRY
    s_izer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01", offset_sec=0)))
    sess = s_store.get_active("VIS_01")
    assert sess is not None
    assert sess.visitor_id == "VIS_01"
    assert sess.is_active is True

    # 2. ZONE_ENTER
    s_izer.process_event(InboundEvent(**make_raw_event("ZONE_ENTER", "VIS_01", zone_id="ZONE_SKIN", offset_sec=10)))
    assert "ZONE_SKIN" in sess.zones_visited

    # 3. ZONE_DWELL
    s_izer.process_event(InboundEvent(**make_raw_event("ZONE_DWELL", "VIS_01", zone_id="ZONE_SKIN", dwell_ms=15000, offset_sec=25)))
    assert sess.dwell_per_zone["ZONE_SKIN"] == 15000

    # 4. BILLING_QUEUE_JOIN
    s_izer.process_event(InboundEvent(**make_raw_event("BILLING_QUEUE_JOIN", "VIS_01", zone_id="BILLING", queue_depth=2, offset_sec=30)))
    assert sess.purchase_candidate is True
    assert len(sess.queue_events) == 1
    assert sess.queue_events[0].queue_depth == 2

    # 5. EXIT
    s_izer.process_event(InboundEvent(**make_raw_event("EXIT", "VIS_01", offset_sec=60)))
    assert sess.is_active is False
    assert sess.duration_ms == 60000

def test_session_out_of_order_creation():
    s_store, s_izer = build_session_pipeline()

    # Zone enter arrives before ENTRY (missed ENTRY)
    s_izer.process_event(InboundEvent(**make_raw_event("ZONE_ENTER", "VIS_01", zone_id="ZONE_SKIN", offset_sec=10)))
    sess = s_store.get_active("VIS_01")
    assert sess is not None  # implicitly created
    assert sess.start_time == make_ts(10)

def test_session_reentry():
    s_store, s_izer = build_session_pipeline()

    # Session 1: ENTRY -> EXIT
    s_izer.process_event(InboundEvent(**make_raw_event("ENTRY", "VIS_01", offset_sec=0)))
    s_izer.process_event(InboundEvent(**make_raw_event("EXIT", "VIS_01", offset_sec=100)))

    # REENTRY
    s_izer.process_event(InboundEvent(**make_raw_event("REENTRY", "VIS_01", offset_sec=200, reid_score=0.85)))
    
    sess = s_store.get_active("VIS_01")
    assert sess is not None
    assert sess.reentry_count == 1

# ---------------------------------------------------------------------------
# 4. Calibration Engine Tests
# ---------------------------------------------------------------------------

def test_calibration_rolling_window():
    cal = CalibrationEngine(calibrate_every_n_events=5)
    # Feed some observations
    for i in range(10):
        cal.observe_event("STORE_A", "CAM_01", confidence=0.8)
        cal.observe_reid("STORE_A", reid_score=0.7)
        cal.observe_staff_score("STORE_A", score=0.6)
        cal.observe_queue_dwell("STORE_A", dwell_sec=4.0)
        cal.observe_zone_dwell("STORE_A", dwell_ms=20000.0)

    # Perform calibration
    changes = cal.calibrate("STORE_A")
    assert "STORE_A" in cal.all_profiles()
    
    profile = cal.get_profile("STORE_A")
    assert profile.reid_confidence_threshold > 0.0
    assert profile.queue_join_dwell_sec > 0.0

# ---------------------------------------------------------------------------
# 5. Verifier Engine Tests
# ---------------------------------------------------------------------------

def test_verifier_violations():
    audit = AuditTimeline()
    verifier = VerifierEngine(audit=audit)

    # 1. QUEUE_DEPTH_NEGATIVE
    evt1 = InboundEvent(**make_raw_event("BILLING_QUEUE_JOIN", "VIS_01", queue_depth=-1))
    warnings = verifier.verify_event(evt1, None)
    assert any(w.code == "QUEUE_DEPTH_NEGATIVE" for w in warnings)

    # 2. CONFIDENCE_CLIFF
    evt2 = InboundEvent(**make_raw_event("ZONE_ENTER", "VIS_02", confidence=0.9, offset_sec=0))
    verifier.verify_event(evt2, None)
    evt3 = InboundEvent(**make_raw_event("ZONE_EXIT", "VIS_02", confidence=0.3, offset_sec=10))
    warnings = verifier.verify_event(evt3, None)
    assert any(w.code == "CONFIDENCE_CLIFF" for w in warnings)

    # 3. REENTRY_TOO_FAST
    evt4 = InboundEvent(**make_raw_event("EXIT", "VIS_03", offset_sec=0))
    verifier.verify_event(evt4, None)
    evt5 = InboundEvent(**make_raw_event("REENTRY", "VIS_03", offset_sec=2))
    warnings = verifier.verify_event(evt5, None)
    assert any(w.code == "REENTRY_TOO_FAST" for w in warnings)

    # 4. DUPLICATE_ACTIVE_SESSION
    s1 = VisitorSession(visitor_id="VIS_04", store_id="STORE_A", start_time=make_ts(0))
    s2 = VisitorSession(visitor_id="VIS_04", store_id="STORE_A", start_time=make_ts(10))
    warnings = verifier.verify_active_sessions([s1, s2], "STORE_A")
    assert any(w.code == "DUPLICATE_ACTIVE_SESSION" for w in warnings)

# ---------------------------------------------------------------------------
# 6. Replay Engine Tests
# ---------------------------------------------------------------------------

def test_replay_engine_reproducibility():
    es = EventStore()
    s_store, s_izer = build_session_pipeline()
    audit = AuditTimeline()
    replay = ReplayEngine(es, s_store, s_izer, audit, mode=ReplayMode.REPLAY)

    events = [
        make_raw_event("ENTRY", "VIS_01", offset_sec=0),
        make_raw_event("ZONE_ENTER", "VIS_01", zone_id="SKIN", offset_sec=20),
        make_raw_event("EXIT", "VIS_01", offset_sec=40)
    ]

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tf:
        for e in events:
            tf.write(json.dumps(e) + "\n")
        temp_path = tf.name

    try:
        res = replay.replay_file(temp_path)
        assert res.accepted == 3
        sessions = s_store.get_all_sessions("STORE_BLR_002")
        assert len(sessions) == 1
        assert sessions[0].visitor_id == "VIS_01"
        assert sessions[0].duration_ms == 40000
    finally:
        os.unlink(temp_path)

# ---------------------------------------------------------------------------
# 7. POS Correlation & Projections
# ---------------------------------------------------------------------------

def test_pos_correlation():
    corr = CorrelationEngine()
    
    # Session with billing zone
    s = VisitorSession(visitor_id="VIS_01", store_id="STORE_A", start_time=make_ts(0))
    s.record_zone_visit("ZONE_BILLING_QUEUE")
    s.queue_events.append(QueueEvent(
        event_type=EventType.BILLING_QUEUE_JOIN,
        timestamp=make_ts(50)
    ))

    # POS transaction 3 minutes after billing queue join (within 5 minutes window)
    txn = POSTransaction(store_id="STORE_A", transaction_id="TX_100", timestamp=make_ts(230))
    corr.add_transaction(txn)

    corr.correlate([s], "STORE_A")
    assert corr.is_converted(s.session_id) is True

    # Check conversion rate
    rate = corr.conversion_rate([s], "STORE_A")
    assert rate == 1.0

def test_funnel_projection():
    s1 = VisitorSession(visitor_id="VIS_01", store_id="STORE_A", start_time=make_ts(0))
    s1.record_zone_visit("ZONE_SKIN")
    s1.purchase_candidate = True

    proj = FunnelProjection.build([s1], "STORE_A")
    assert proj["funnel"][0]["count"] == 1  # ENTRY
    assert proj["funnel"][1]["count"] == 1  # ZONE_VISIT
    assert proj["funnel"][2]["count"] == 1  # BILLING_QUEUE

# ---------------------------------------------------------------------------
# 8. API Endpoint Tests (Integration)
# ---------------------------------------------------------------------------

def test_api_full_flow():
    # Clear main state
    _event_store.clear()
    _sess_store.clear()
    _audit.clear()
    _verifier.clear()
    _correlation.clear()

    # 1. Ingest events
    batch = [
        make_raw_event("ENTRY", "VIS_API_01", offset_sec=0, store_id="STORE_API_TEST"),
        make_raw_event("ZONE_ENTER", "VIS_API_01", zone_id="ZONE_A", offset_sec=10, store_id="STORE_API_TEST"),
        make_raw_event("BILLING_QUEUE_JOIN", "VIS_API_01", zone_id="BILLING", queue_depth=2, offset_sec=20, store_id="STORE_API_TEST"),
        make_raw_event("EXIT", "VIS_API_01", offset_sec=60, store_id="STORE_API_TEST")
    ]
    response = client.post("/events/ingest", json={"events": batch})
    assert response.status_code == 200
    assert response.json()["accepted"] == 4

    # 2. GET metrics
    resp_metrics = client.get("/stores/STORE_API_TEST/metrics")
    assert resp_metrics.status_code == 200
    metrics = resp_metrics.json()
    assert metrics["unique_visitors"] == 1
    assert metrics["total_sessions"] == 1

    # 3. GET funnel
    resp_funnel = client.get("/stores/STORE_API_TEST/funnel")
    assert resp_funnel.status_code == 200
    funnel = resp_funnel.json()
    assert funnel["funnel"][0]["count"] == 1

    # 4. GET heatmap
    resp_heatmap = client.get("/stores/STORE_API_TEST/heatmap")
    assert resp_heatmap.status_code == 200
    heatmap = resp_heatmap.json()
    assert heatmap["data_confidence"] is False  # < 20 sessions

    # 5. GET anomalies
    resp_anom = client.get("/stores/STORE_API_TEST/anomalies")
    assert resp_anom.status_code == 200

    # 6. GET health
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] in ("OK", "DEGRADED")


# ---------------------------------------------------------------------------
# 9. Edge Cases (Empty, All Staff, Stale feed, Zero Purchases)
# ---------------------------------------------------------------------------

def test_empty_store():
    # If store does not exist, return 404
    resp = client.get("/stores/NON_EXISTENT_STORE/metrics")
    assert resp.status_code == 404

def test_all_staff():
    _event_store.clear()
    _sess_store.clear()

    # Ingest staff-only events
    batch = [
        make_raw_event("ENTRY", "VIS_STAFF_01", is_staff=True, store_id="STORE_STAFF_ONLY"),
        make_raw_event("EXIT", "VIS_STAFF_01", is_staff=True, store_id="STORE_STAFF_ONLY")
    ]
    client.post("/events/ingest", json={"events": batch})
    
    resp_metrics = client.get("/stores/STORE_STAFF_ONLY/metrics")
    assert resp_metrics.status_code == 200
    # Staff must be filtered out of business metrics
    assert resp_metrics.json()["unique_visitors"] == 0

def test_zero_purchases_conversion_rate():
    _event_store.clear()
    _sess_store.clear()

    batch = [
        make_raw_event("ENTRY", "VIS_NOPUR_01", store_id="STORE_ZERO_PUR"),
        make_raw_event("EXIT", "VIS_NOPUR_01", store_id="STORE_ZERO_PUR")
    ]
    client.post("/events/ingest", json={"events": batch})

    resp_metrics = client.get("/stores/STORE_ZERO_PUR/metrics")
    assert resp_metrics.json()["conversion_rate"] == 0.0

def test_brigade_bangalore_csv_format():
    corr = CorrelationEngine()
    
    # Create a temporary CSV with production columns
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tf:
        tf.write(
            "order_id,coupon_code,offer_name,discount_code,invoice_number,invoice_type,"
            "order_date,order_time,return_id,store_id,store_name,city,customer_name,"
            "customer_number,sku,product_id,ean,product_name,brand_name,dep_name,"
            "sub_category,brand_type,tax,hsn_code,salesperson_id,employee_code,"
            "salesperson_name,qty,GMV,NMV,coupon_amount,item_promotion,amt_without_gwp,"
            "total_amount,pb_eb_sale,week_assigned,tax_m,taxable_amt,tax_amt\n"
        )
        # Row 1: order 104363838, date 10-04-2026, time 16:55:36, store ST1008, total_amount 274.36
        tf.write(
            "104363838,,Buy 2 Get 1 on PB,,ML0426KAP0001358,sales,10-04-2026,16:55:36,,"
            "ST1008,Brigade_Bangalore,Bangalore,Guest ,9346413680,PPLBDD8904362534994NM2,"
            "402813,8.90436E+12,DERMDOC Body Wash,DERMDOC,bath-and-body,Body Wash,PB,18,"
            "33049990,1178,CL2063,kasthuri v,1,400,274.36,0,125.64,274.36,274.36,274.36,,1.18,232.51,41.85\n"
        )
        temp_path = tf.name

    try:
        loaded = corr.load_csv(temp_path)
        assert loaded == 1
        
        # Verify transaction loaded correctly under store_id ST1008
        assert corr.transaction_count("ST1008") == 1
        txns = corr._transactions["ST1008"]
        assert txns[0].transaction_id == "104363838"
        assert txns[0].basket_value_inr == 274.36
        assert txns[0].timestamp == "2026-04-10T16:55:36Z"
    finally:
        os.unlink(temp_path)


def test_visitor_centric_conversion_and_funnel_reentry():
    _event_store.clear()
    _sess_store.clear()
    _verifier.clear()
    _correlation._transactions.clear()

    # Visitor enters first time, exits. No purchase.
    # Visitor re-enters second time, exits with purchase candidate status (joined queue).
    batch = [
        make_raw_event("ENTRY", "VIS_MULTI_01", store_id="STORE_REENTRY", offset_sec=0),
        make_raw_event("EXIT", "VIS_MULTI_01", store_id="STORE_REENTRY", offset_sec=10),
        make_raw_event("REENTRY", "VIS_MULTI_01", store_id="STORE_REENTRY", offset_sec=20, reid_score=0.9),
        make_raw_event("BILLING_QUEUE_JOIN", "VIS_MULTI_01", store_id="STORE_REENTRY", offset_sec=30, queue_depth=2),
        make_raw_event("EXIT", "VIS_MULTI_01", store_id="STORE_REENTRY", offset_sec=40)
    ]
    client.post("/events/ingest", json={"events": batch})

    # Since they joined queue in the second session, they are a purchase candidate.
    # Therefore, visitor-centric conversion rate should be 1.0 (1 converted / 1 unique visitor).
    resp_metrics = client.get("/stores/STORE_REENTRY/metrics")
    assert resp_metrics.status_code == 200
    data = resp_metrics.json()
    assert data["unique_visitors"] == 1
    assert data["conversion_rate"] == 1.0

    # Let's also check the funnel:
    # ENTRY stage: 1
    # BILLING_QUEUE stage: 1
    # PURCHASE stage: 1 (since purchase_candidate maps to purchase when correlation_engine is None or default in local mock)
    resp_funnel = client.get("/stores/STORE_REENTRY/funnel")
    assert resp_funnel.status_code == 200
    funnel_stages = resp_funnel.json()["funnel"]
    assert funnel_stages[0]["count"] == 1  # ENTRY
    assert funnel_stages[2]["count"] == 1  # BILLING_QUEUE
    assert funnel_stages[3]["count"] == 1  # PURCHASE


def test_abandonment_rate_multiple_events():
    _event_store.clear()
    _sess_store.clear()
    _verifier.clear()

    # Visitor enters, joins queue, and then abandons queue.
    batch = [
        make_raw_event("ENTRY", "VIS_ABANDON_01", store_id="STORE_ABANDON", offset_sec=0),
        make_raw_event("BILLING_QUEUE_JOIN", "VIS_ABANDON_01", store_id="STORE_ABANDON", offset_sec=10, queue_depth=2),
        make_raw_event("BILLING_QUEUE_ABANDON", "VIS_ABANDON_01", store_id="STORE_ABANDON", offset_sec=20, wait_duration_ms=5000),
        make_raw_event("EXIT", "VIS_ABANDON_01", store_id="STORE_ABANDON", offset_sec=30)
    ]
    client.post("/events/ingest", json={"events": batch})

    # The visitor joined AND abandoned.
    # Abandonment rate = abandon_sessions / join_sessions.
    # Since they abandoned, it should be 1.0.
    resp_metrics = client.get("/stores/STORE_ABANDON/metrics")
    assert resp_metrics.status_code == 200
    data = resp_metrics.json()
    assert data["abandonment_rate"] == 1.0


def test_event_verifier_triggered_during_sessionization():
    _event_store.clear()
    _sess_store.clear()
    _verifier.clear()

    # Ingesting a negative queue depth should trigger QUEUE_DEPTH_NEGATIVE warning.
    batch = [
        make_raw_event("ENTRY", "VIS_ERR_01", store_id="STORE_ERR", offset_sec=0),
        make_raw_event("BILLING_QUEUE_JOIN", "VIS_ERR_01", store_id="STORE_ERR", offset_sec=10, queue_depth=-5),
        make_raw_event("EXIT", "VIS_ERR_01", store_id="STORE_ERR", offset_sec=20)
    ]
    client.post("/events/ingest", json={"events": batch})

    # Check warnings
    warnings = _verifier.get_warnings(store_id="STORE_ERR")
    assert len(warnings) > 0
    assert any(w.code == "QUEUE_DEPTH_NEGATIVE" for w in warnings)


def test_replay_engine_verifier_and_correlation_consistency():
    _event_store.clear()
    _sess_store.clear()
    _verifier.clear()

    # Replay events from a file. First write events to a temporary file.
    events = [
        make_raw_event("ENTRY", "VIS_REP_01", store_id="STORE_REP", offset_sec=0),
        make_raw_event("BILLING_QUEUE_JOIN", "VIS_REP_01", store_id="STORE_REP", offset_sec=10, queue_depth=-3),
        make_raw_event("EXIT", "VIS_REP_01", store_id="STORE_REP", offset_sec=20)
    ]
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tf:
        for ev in events:
            tf.write(json.dumps(ev) + "\n")
        temp_path = tf.name

    try:
        # Trigger replay via endpoint
        resp = client.post("/stores/STORE_REP/replay", json={"path": temp_path})
        assert resp.status_code == 200
        
        # Verify that the event-level verifier ran and warnings exist
        warnings = _verifier.get_warnings(store_id="STORE_REP")
        assert len(warnings) > 0
        assert any(w.code == "QUEUE_DEPTH_NEGATIVE" for w in warnings)
        
        # Verify that session is processed correctly
        resp_metrics = client.get("/stores/STORE_REP/metrics")
        assert resp_metrics.status_code == 200
        assert resp_metrics.json()["unique_visitors"] == 1
    finally:
        os.unlink(temp_path)

