import uuid
from datetime import datetime, timezone, timedelta
import pytest
from app.database import POSTransaction

STORE = "STORE_BLR_002"
NOW = datetime.now(timezone.utc)

def ts(m=0): return (NOW - timedelta(minutes=m)).isoformat()
def make_event(et, vid, zone_id=None, is_staff=False, dwell_ms=0, m=5):
    return {"event_id": str(uuid.uuid4()), "store_id": STORE, "camera_id": "CAM_FLOOR_01",
            "visitor_id": vid, "event_type": et, "timestamp": ts(m), "zone_id": zone_id,
            "dwell_ms": dwell_ms, "is_staff": is_staff, "confidence": 0.9, "metadata": {"session_seq": 1}}
def ingest(client, events):
    r = client.post("/events/ingest", json={"events": events}); assert r.status_code == 200; return r.json()
def add_pos(db, store_id=STORE, m=1):
    t = POSTransaction(transaction_id=str(uuid.uuid4()), store_id=store_id,
                       timestamp=NOW - timedelta(minutes=m), basket_value_inr=1000.0)
    db.add(t); db.commit()

class TestMetrics:
    def test_empty_store_returns_zeros(self, test_client):
        c, _ = test_client
        r = c.get(f"/stores/{STORE}/metrics"); assert r.status_code == 200
        b = r.json(); assert b["unique_visitors"] == 0; assert b["conversion_rate"] == 0.0
    def test_unique_visitors_excludes_staff(self, test_client):
        c, _ = test_client
        ingest(c, [make_event("ENTRY","VIS_a1",m=10), make_event("ENTRY","VIS_a2",m=8),
                   make_event("ENTRY","VIS_s1",is_staff=True,m=7)])
        assert c.get(f"/stores/{STORE}/metrics").json()["unique_visitors"] == 2
    def test_all_staff_clip_no_crash(self, test_client):
        c, _ = test_client
        ingest(c, [make_event("ENTRY",f"VIS_s{i}",is_staff=True) for i in range(5)])
        b = c.get(f"/stores/{STORE}/metrics").json()
        assert b["unique_visitors"] == 0; assert b["conversion_rate"] == 0.0
    def test_conversion_rate_with_billing_correlation(self, test_client):
        c, db = test_client
        ingest(c, [make_event("ENTRY","VIS_cv01",m=15), make_event("ZONE_ENTER","VIS_cv01",zone_id="BILLING_COUNTER",m=3)])
        add_pos(db)
        assert c.get(f"/stores/{STORE}/metrics").json()["conversion_rate"] > 0.0
    def test_zero_purchase_store_no_null(self, test_client):
        c, _ = test_client
        ingest(c, [make_event("ENTRY",f"VIS_x{i}") for i in range(5)])
        b = c.get(f"/stores/{STORE}/metrics").json()
        assert b["conversion_rate"] == 0.0; assert b["unique_visitors"] == 5
    def test_reentry_not_double_counted(self, test_client):
        c, _ = test_client
        ingest(c, [make_event("ENTRY","VIS_re1",m=60), make_event("EXIT","VIS_re1",m=50), make_event("REENTRY","VIS_re1",m=30)])
        assert c.get(f"/stores/{STORE}/metrics").json()["unique_visitors"] == 1
    def test_avg_dwell_per_zone_computed(self, test_client):
        c, _ = test_client
        ingest(c, [make_event("ZONE_DWELL","VIS_d1",zone_id="SKINCARE",dwell_ms=45000,m=10)])
        zones = {z["zone_id"]: z for z in c.get(f"/stores/{STORE}/metrics").json().get("avg_dwell_per_zone",[])}
        assert "SKINCARE" in zones; assert zones["SKINCARE"]["avg_dwell_ms"] == pytest.approx(45000.0, rel=0.1)
    def test_metrics_unknown_store_returns_zeros(self, test_client):
        c, _ = test_client
        assert c.get("/stores/STORE_UNKNOWN_999/metrics").json()["unique_visitors"] == 0

class TestFunnel:
    def test_funnel_stages_present(self, test_client):
        c, _ = test_client
        stages = {s["stage"] for s in c.get(f"/stores/{STORE}/funnel").json()["stages"]}
        assert {"Entry", "Zone Visit", "Billing Queue", "Purchase"} == stages
    def test_empty_store_funnel_all_zeros(self, test_client):
        c, _ = test_client
        for stage in c.get(f"/stores/{STORE}/funnel").json()["stages"]: assert stage["count"] == 0
    def test_funnel_drop_off_calculation(self, test_client):
        c, _ = test_client
        for i in range(10): ingest(c, [make_event("ENTRY",f"VIS_f{i:02d}",m=60)])
        for i in range(6): ingest(c, [make_event("ZONE_ENTER",f"VIS_f{i:02d}",zone_id="SKINCARE",m=50)])
        for i in range(3): ingest(c, [make_event("ZONE_ENTER",f"VIS_f{i:02d}",zone_id="BILLING_COUNTER",m=40)])
        stages = {s["stage"]: s for s in c.get(f"/stores/{STORE}/funnel").json()["stages"]}
        assert stages["Entry"]["count"] == 10; assert stages["Zone Visit"]["count"] == 6
        assert stages["Zone Visit"]["drop_off_pct"] == pytest.approx(40.0, rel=0.1)

class TestHeatmap:
    def test_heatmap_max_zone_is_100(self, test_client):
        c, _ = test_client
        for i in range(5): ingest(c, [make_event("ZONE_ENTER",f"VIS_h{i}",zone_id="SKINCARE")])
        for i in range(2): ingest(c, [make_event("ZONE_ENTER",f"VIS_hb{i}",zone_id="HAIRCARE")])
        zones = {z["zone_id"]: z for z in c.get(f"/stores/{STORE}/heatmap").json().get("zones",[])}
        assert zones["SKINCARE"]["normalised_score"] == pytest.approx(100.0, rel=0.01)
        assert zones["HAIRCARE"]["normalised_score"] < 100.0
    def test_heatmap_low_confidence_flag(self, test_client):
        c, _ = test_client
        for i in range(5): ingest(c, [make_event("ZONE_ENTER",f"VIS_lc{i}",zone_id="FRAGRANCE")])
        for z in c.get(f"/stores/{STORE}/heatmap").json().get("zones",[]): assert z["data_confidence"] == False
    def test_heatmap_empty_store(self, test_client):
        c, _ = test_client
        r = c.get(f"/stores/{STORE}/heatmap"); assert r.status_code == 200; assert r.json()["zones"] == []
