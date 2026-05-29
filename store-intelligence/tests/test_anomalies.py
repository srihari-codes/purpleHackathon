import uuid
from datetime import datetime, timezone, timedelta
import pytest
from app.database import EventRecord, POSTransaction

STORE = "STORE_BLR_002"
NOW = datetime.now(timezone.utc)

def make_event(et, vid, zone_id=None, is_staff=False, m=5, queue_depth=None):
    meta = {"session_seq": 1}
    if queue_depth is not None: meta["queue_depth"] = queue_depth
    return {"event_id": str(uuid.uuid4()), "store_id": STORE, "camera_id": "CAM_BILLING_01",
            "visitor_id": vid, "event_type": et, "timestamp": (NOW-timedelta(minutes=m)).isoformat(),
            "zone_id": zone_id, "dwell_ms": 0, "is_staff": is_staff, "confidence": 0.9, "metadata": meta}
def ingest(client, events):
    r = client.post("/events/ingest", json={"events": events}); assert r.status_code == 200

class TestAnomalyDetection:
    def test_no_anomalies_on_empty_store(self, test_client):
        c, _ = test_client
        r = c.get(f"/stores/{STORE}/anomalies"); assert r.status_code == 200
        assert r.json()["active_anomalies"] == []
    def test_billing_queue_spike_warn(self, test_client):
        c, _ = test_client
        ingest(c, [make_event("BILLING_QUEUE_JOIN","VIS_q01",zone_id="BILLING_QUEUE",queue_depth=5,m=1)])
        anomalies = c.get(f"/stores/{STORE}/anomalies").json()["active_anomalies"]
        qa = [a for a in anomalies if a["anomaly_type"] == "BILLING_QUEUE_SPIKE"]
        assert len(qa) == 1; assert qa[0]["severity"] == "WARN"
    def test_billing_queue_spike_critical(self, test_client):
        c, _ = test_client
        ingest(c, [make_event("BILLING_QUEUE_JOIN","VIS_q02",zone_id="BILLING_QUEUE",queue_depth=10,m=1)])
        anomalies = c.get(f"/stores/{STORE}/anomalies").json()["active_anomalies"]
        qa = [a for a in anomalies if a["anomaly_type"] == "BILLING_QUEUE_SPIKE"]
        assert len(qa) == 1; assert qa[0]["severity"] == "CRITICAL"
    def test_no_queue_spike_below_threshold(self, test_client):
        c, _ = test_client
        ingest(c, [make_event("BILLING_QUEUE_JOIN","VIS_q03",zone_id="BILLING_QUEUE",queue_depth=3,m=1)])
        anomalies = c.get(f"/stores/{STORE}/anomalies").json()["active_anomalies"]
        assert not any(a["anomaly_type"] == "BILLING_QUEUE_SPIKE" for a in anomalies)
    def test_dead_zone_detected_after_30_min(self, test_client):
        c, _ = test_client
        ingest(c, [make_event("ZONE_ENTER","VIS_dz01",zone_id="SKINCARE",m=31)])
        anomalies = c.get(f"/stores/{STORE}/anomalies").json()["active_anomalies"]
        dead = [a for a in anomalies if a["anomaly_type"] == "DEAD_ZONE"]
        assert len(dead) >= 1; assert "SKINCARE" in dead[0]["metadata"]["dead_zones"]
    def test_no_dead_zone_within_30_min(self, test_client):
        c, _ = test_client
        ingest(c, [make_event("ZONE_ENTER","VIS_dz02",zone_id="SKINCARE",m=5)])
        anomalies = c.get(f"/stores/{STORE}/anomalies").json()["active_anomalies"]
        dead = [a for a in anomalies if a["anomaly_type"] == "DEAD_ZONE"]
        if dead: assert "SKINCARE" not in dead[0]["metadata"].get("dead_zones", [])
    def test_anomaly_has_suggested_action(self, test_client):
        c, _ = test_client
        ingest(c, [make_event("BILLING_QUEUE_JOIN","VIS_sa01",zone_id="BILLING_QUEUE",queue_depth=6,m=1)])
        for a in c.get(f"/stores/{STORE}/anomalies").json()["active_anomalies"]:
            assert a.get("suggested_action")

class TestHealth:
    def test_health_ok_with_no_stores(self, test_client):
        c, _ = test_client
        r = c.get("/health"); assert r.status_code == 200
        b = r.json(); assert b["status"] in ("ok", "degraded"); assert "version" in b
    def test_health_ok_with_recent_events(self, test_client):
        c, _ = test_client
        ingest(c, [make_event("ENTRY","VIS_h01",m=2)])
        stores = c.get("/health").json()["stores"]
        s = next((s for s in stores if s["store_id"] == STORE), None)
        assert s is not None; assert s["status"] == "OK"; assert s["lag_minutes"] <= 5
    def test_stale_feed_flagged_for_old_events(self, test_client):
        c, db = test_client
        old = EventRecord(event_id=str(uuid.uuid4()), store_id=STORE, camera_id="CAM_ENTRY_01",
                          visitor_id="VIS_old01", event_type="ENTRY",
                          timestamp=NOW - timedelta(minutes=15), dwell_ms=0, is_staff=False, confidence=0.9)
        db.add(old); db.commit()
        stores = c.get("/health").json()["stores"]
        s = next((s for s in stores if s["store_id"] == STORE), None)
        assert s is not None; assert s["status"] == "STALE_FEED"
    def test_health_overall_degraded_when_stale(self, test_client):
        c, db = test_client
        old = EventRecord(event_id=str(uuid.uuid4()), store_id=STORE, camera_id="CAM_ENTRY_01",
                          visitor_id="VIS_old02", event_type="ENTRY",
                          timestamp=NOW - timedelta(minutes=20), dwell_ms=0, is_staff=False, confidence=0.9)
        db.add(old); db.commit()
        assert c.get("/health").json()["status"] == "degraded"
