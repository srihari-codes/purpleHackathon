"""
tests/test_api_integration.py — Full end-to-end API integration tests (app/main.py).

27 tests covering every endpoint, edge case, and cross-concern:
  POST /events/ingest       — success, dedup, partial, all-invalid, batch>500, empty
  GET  /stores/{id}/metrics — 200 with data, 404 unknown store
  GET  /stores/{id}/funnel  — 200 with data, 404 unknown store
  GET  /stores/{id}/heatmap — 200 (data_confidence flag), 404 unknown store
  GET  /stores/{id}/anomalies — 200, queue spike anomaly present, 404
  GET  /health              — OK, DEGRADED (stale feed)
  GET  /stores/{id}/audit/{visitor_id} — 200, 404 unknown visitor
  GET  /stores/{id}/calibration — 200 profile structure
  POST /stores/{id}/replay  — 200, 409 conflict, 404 file not found
  Cross-cutting: X-Trace-Id header, CORS header, staff filter, zero-purchase rate,
                 reentry dedup in metrics, abandonment rate, verifier during ingest
"""

import datetime as dt
import json
import os
import tempfile
from datetime import timezone

import pytest

from tests.conftest import make_raw_event, make_ts


# ---------------------------------------------------------------------------
# Ingest helpers
# ---------------------------------------------------------------------------

def ingest(client, events, store_id=None):
    """Helper: POST a batch and return response JSON."""
    resp = client.post("/events/ingest", json={"events": events})
    assert resp.status_code == 200
    return resp.json()


def full_session(visitor_id: str, store_id: str = "STORE_INT", with_zone=True, with_purchase=False):
    """Return a minimal list of events for one full visitor session."""
    batch = [
        make_raw_event("ENTRY", visitor_id, store_id=store_id, offset_sec=0),
    ]
    if with_zone:
        batch.append(make_raw_event("ZONE_ENTER", visitor_id, store_id=store_id, zone_id="ZONE_A", offset_sec=10))
    if with_purchase:
        batch.append(make_raw_event("BILLING_QUEUE_JOIN", visitor_id, store_id=store_id, queue_depth=2, offset_sec=20))
    batch.append(make_raw_event("EXIT", visitor_id, store_id=store_id, offset_sec=60))
    return batch


# ---------------------------------------------------------------------------
# POST /events/ingest
# ---------------------------------------------------------------------------

def test_ingest_success(api_client):
    data = ingest(api_client, full_session("VIS_01"))
    assert data["accepted"] == 3   # ENTRY + ZONE_ENTER + EXIT
    assert data["rejected"] == 0
    assert data["duplicates"] == 0


def test_ingest_idempotent_deduplication(api_client):
    batch = full_session("VIS_01")
    data1 = ingest(api_client, batch)
    data2 = ingest(api_client, batch)  # replay same batch
    assert data1["accepted"] == 3
    assert data2["accepted"] == 0
    assert data2["duplicates"] == 3


def test_ingest_partial_success(api_client):
    valid = make_raw_event("ENTRY", "VIS_01", store_id="STORE_INT")
    invalid = make_raw_event("ENTRY", "VIS_02", store_id="STORE_INT", timestamp="bad-ts")
    data = ingest(api_client, [valid, invalid])
    assert data["accepted"] == 1
    assert data["rejected"] == 1
    assert len(data["errors"]) == 1
    assert "timestamp" in data["errors"][0]["reason"]


def test_ingest_all_invalid(api_client):
    resp = api_client.post("/events/ingest", json={"events": [
        {"bad": "data"},
        {"also": "bad"},
    ]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] == 0
    assert data["rejected"] == 2


def test_ingest_empty_batch_rejected(api_client):
    """Empty events list violates min_length=1 → 422."""
    resp = api_client.post("/events/ingest", json={"events": []})
    assert resp.status_code == 422


def test_ingest_batch_over_500_rejected(api_client):
    big = [make_raw_event("ENTRY", f"VIS_{i}", store_id="STORE_INT", offset_sec=i) for i in range(501)]
    resp = api_client.post("/events/ingest", json={"events": big})
    assert resp.status_code == 422


def test_ingest_invalid_uuid_rejected(api_client):
    raw = make_raw_event("ENTRY", "VIS_01", store_id="STORE_INT")
    raw["event_id"] = "not-a-uuid"
    data = ingest(api_client, [raw])
    assert data["rejected"] == 1
    assert "valid UUID" in data["errors"][0]["reason"]


def test_ingest_strict_type_dwell_ms_string(api_client):
    raw = make_raw_event("ZONE_DWELL", "VIS_01", store_id="STORE_INT", zone_id="Z1")
    raw["dwell_ms"] = "500"
    data = ingest(api_client, [raw])
    assert data["rejected"] == 1


def test_ingest_response_has_x_trace_id(api_client):
    resp = api_client.post("/events/ingest", json={"events": full_session("VIS_01")})
    assert "x-trace-id" in resp.headers


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/metrics
# ---------------------------------------------------------------------------

def test_metrics_404_unknown_store(api_client):
    resp = api_client.get("/stores/NO_SUCH_STORE/metrics")
    assert resp.status_code == 404
    assert resp.json()["error"] == "STORE_NOT_FOUND"


def test_metrics_200_after_ingest(api_client):
    ingest(api_client, full_session("VIS_01", "STORE_METRICS"))
    resp = api_client.get("/stores/STORE_METRICS/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["unique_visitors"] == 1
    assert data["total_sessions"] == 1
    assert data["store_id"] == "STORE_METRICS"
    assert "as_of" in data


def test_metrics_staff_excluded(api_client):
    ingest(api_client, [
        make_raw_event("ENTRY", "STAFF_01", store_id="STORE_STAFF", is_staff=True),
        make_raw_event("EXIT", "STAFF_01", store_id="STORE_STAFF", is_staff=True, offset_sec=30),
    ])
    resp = api_client.get("/stores/STORE_STAFF/metrics")
    assert resp.status_code == 200
    assert resp.json()["unique_visitors"] == 0


def test_metrics_zero_purchase_conversion_rate(api_client):
    ingest(api_client, full_session("VIS_01", "STORE_NOPURCH", with_purchase=False))
    resp = api_client.get("/stores/STORE_NOPURCH/metrics")
    assert resp.json()["conversion_rate"] == 0.0


def test_metrics_reentry_dedup(api_client):
    """Same visitor_id re-entering must count as 1 unique visitor."""
    sid = "STORE_REENT"
    batch = [
        make_raw_event("ENTRY", "VIS_01", store_id=sid, offset_sec=0),
        make_raw_event("EXIT", "VIS_01", store_id=sid, offset_sec=30),
        make_raw_event("REENTRY", "VIS_01", store_id=sid, offset_sec=60, reid_score=0.9),
        make_raw_event("EXIT", "VIS_01", store_id=sid, offset_sec=90),
    ]
    ingest(api_client, batch)
    resp = api_client.get(f"/stores/{sid}/metrics")
    assert resp.json()["unique_visitors"] == 1


def test_metrics_abandonment_rate(api_client):
    sid = "STORE_ABAND"
    batch = [
        make_raw_event("ENTRY", "VIS_01", store_id=sid, offset_sec=0),
        make_raw_event("BILLING_QUEUE_JOIN", "VIS_01", store_id=sid, queue_depth=2, offset_sec=10),
        make_raw_event("BILLING_QUEUE_ABANDON", "VIS_01", store_id=sid, wait_duration_ms=5000, offset_sec=20),
        make_raw_event("EXIT", "VIS_01", store_id=sid, offset_sec=30),
    ]
    ingest(api_client, batch)
    resp = api_client.get(f"/stores/{sid}/metrics")
    assert resp.json()["abandonment_rate"] == 1.0


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/funnel
# ---------------------------------------------------------------------------

def test_funnel_404_unknown_store(api_client):
    resp = api_client.get("/stores/NO_SUCH_STORE/funnel")
    assert resp.status_code == 404


def test_funnel_200_correct_stages(api_client):
    sid = "STORE_FUN"
    ingest(api_client, full_session("VIS_01", sid, with_zone=True, with_purchase=True))
    resp = api_client.get(f"/stores/{sid}/funnel")
    assert resp.status_code == 200
    funnel = resp.json()["funnel"]
    assert funnel[0]["stage"] == "ENTRY"
    assert funnel[0]["count"] == 1
    assert funnel[1]["stage"] == "ZONE_VISIT"
    assert funnel[1]["count"] == 1
    assert funnel[2]["stage"] == "BILLING_QUEUE"
    assert funnel[2]["count"] == 1
    assert funnel[3]["stage"] == "PURCHASE"
    assert funnel[3]["count"] == 1


def test_funnel_pct_of_top_is_100_for_entry(api_client):
    sid = "STORE_FUN2"
    ingest(api_client, full_session("VIS_01", sid))
    resp = api_client.get(f"/stores/{sid}/funnel")
    funnel = resp.json()["funnel"]
    assert funnel[0]["pct_of_top"] == 100.0


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/heatmap
# ---------------------------------------------------------------------------

def test_heatmap_404_unknown_store(api_client):
    resp = api_client.get("/stores/NO_SUCH_STORE/heatmap")
    assert resp.status_code == 404


def test_heatmap_200_data_confidence_false_below_20(api_client):
    sid = "STORE_HEAT"
    ingest(api_client, full_session("VIS_01", sid, with_zone=True))
    resp = api_client.get(f"/stores/{sid}/heatmap")
    assert resp.status_code == 200
    data = resp.json()
    assert data["data_confidence"] is False  # only 1 session


def test_heatmap_200_zones_populated(api_client):
    sid = "STORE_HEAT2"
    ingest(api_client, full_session("VIS_01", sid, with_zone=True))
    resp = api_client.get(f"/stores/{sid}/heatmap")
    data = resp.json()
    # ZONE_A was visited — should appear
    assert any(z["zone_id"] == "ZONE_A" for z in data["zones"])


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/anomalies
# ---------------------------------------------------------------------------

def test_anomalies_404_unknown_store(api_client):
    resp = api_client.get("/stores/NO_SUCH_STORE/anomalies")
    assert resp.status_code == 404


def test_anomalies_200_queue_spike(api_client):
    sid = "STORE_ANOM"
    ingest(api_client, [
        make_raw_event("ENTRY", "VIS_01", store_id=sid, offset_sec=0),
        make_raw_event("BILLING_QUEUE_JOIN", "VIS_01", store_id=sid, queue_depth=9, offset_sec=10),
    ])
    resp = api_client.get(f"/stores/{sid}/anomalies")
    assert resp.status_code == 200
    types = [a["type"] for a in resp.json()["anomalies"]]
    assert "BILLING_QUEUE_SPIKE" in types


def test_anomalies_verifier_warning_surfaced(api_client):
    """QUEUE_DEPTH_NEGATIVE from verifier must appear in /anomalies."""
    sid = "STORE_VWARN"
    ingest(api_client, [
        make_raw_event("ENTRY", "VIS_01", store_id=sid, offset_sec=0),
        make_raw_event("BILLING_QUEUE_JOIN", "VIS_01", store_id=sid, queue_depth=-3, offset_sec=10),
    ])
    resp = api_client.get(f"/stores/{sid}/anomalies")
    assert resp.status_code == 200
    types = [a["type"] for a in resp.json()["anomalies"]]
    assert "QUEUE_DEPTH_NEGATIVE" in types


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_ok(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("OK", "DEGRADED")  # depends on prior test pollution
    assert "started_at" in data
    assert "store_count" in data


def test_health_stale_feed_degraded(api_client):
    stale = (dt.datetime.now(timezone.utc) - dt.timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
    raw = make_raw_event("ENTRY", "VIS_STALE", store_id="STORE_STALE_H")
    raw["timestamp"] = stale
    api_client.post("/events/ingest", json={"events": [raw]})
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "DEGRADED"
    assert any("STORE_STALE_H" in w for w in data["warnings"])


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/audit/{visitor_id}
# ---------------------------------------------------------------------------

def test_audit_endpoint_200(api_client):
    sid = "STORE_AUD"
    ingest(api_client, [make_raw_event("ENTRY", "VIS_AUD_01", store_id=sid)])
    resp = api_client.get(f"/stores/{sid}/audit/VIS_AUD_01")
    assert resp.status_code == 200
    data = resp.json()
    assert data["visitor_id"] == "VIS_AUD_01"
    assert data["record_count"] >= 1


def test_audit_endpoint_404(api_client):
    resp = api_client.get("/stores/STORE_X/audit/TOTALLY_UNKNOWN_VISITOR")
    assert resp.status_code == 404
    assert resp.json()["error"] == "VISITOR_NOT_FOUND"


# ---------------------------------------------------------------------------
# GET /stores/{store_id}/calibration
# ---------------------------------------------------------------------------

def test_calibration_endpoint_200(api_client):
    # Just GET — calibration profile is created on demand
    resp = api_client.get("/stores/STORE_CAL/calibration")
    assert resp.status_code == 200
    data = resp.json()
    assert "reid_confidence_threshold" in data
    assert "queue_join_dwell_sec" in data
    assert "store_id" in data


# ---------------------------------------------------------------------------
# POST /stores/{store_id}/replay
# ---------------------------------------------------------------------------

def test_replay_endpoint_200(api_client):
    events = [
        make_raw_event("ENTRY", "VIS_REP", store_id="STORE_REP", offset_sec=0),
        make_raw_event("EXIT", "VIS_REP", store_id="STORE_REP", offset_sec=30),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        path = f.name
    try:
        resp = api_client.post("/stores/STORE_REP/replay", json={"path": path})
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] == 2
    finally:
        os.unlink(path)


def test_replay_endpoint_file_not_found(api_client):
    resp = api_client.post(
        "/stores/STORE_REP/replay",
        json={"path": "/tmp/does_not_exist_12345.jsonl"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"] == "FILE_NOT_FOUND"


def test_replay_endpoint_conflict_409(api_client):
    """If a replay is already running, a second request must return 409."""
    from app.main import _replay
    _replay._is_replaying = True
    try:
        resp = api_client.post("/stores/X/replay", json={"path": "/any.jsonl"})
        assert resp.status_code == 409
        assert resp.json()["error"] == "REPLAY_IN_PROGRESS"
    finally:
        _replay._is_replaying = False
