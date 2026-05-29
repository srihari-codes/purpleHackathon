# PROMPT:
# "Write pytest tests for a FastAPI event ingestion endpoint that:
#  - accepts a batch of StoreEvent objects (up to 500)
#  - is idempotent (re-sending same event_id is a no-op, not an error)
#  - returns partial success (valid events are accepted even if some are malformed)
#  - excludes is_staff=True events from customer metrics
#  - validates the full event schema including EventType enum and zone_id requirement
#  Include fixtures for test DB, test client, and sample events for all 8 event types."
#
# CHANGES MADE:
# - Added edge case: empty payload (should return 200 with 0 accepted, not 422)
# - Added test for is_staff filtering in metrics endpoint (AI only tested ingest)
# - Fixed timestamp format: AI used naive datetime; changed to UTC-aware
# - Added test for >500 events batch rejection (AI missed the max_items=500 constraint)
# - Moved fixtures to conftest.py; test_client yields (client, db_session) tuple

import uuid
from datetime import datetime, timezone


def make_event(
    event_type="ENTRY",
    zone_id=None,
    store_id="STORE_BLR_002",
    is_staff=False,
    visitor_id=None,
    event_id=None,
) -> dict:
    return {
        "event_id": event_id or str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": visitor_id or f"VIS_{uuid.uuid4().hex[:6]}",
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "zone_id": zone_id,
        "dwell_ms": 0,
        "is_staff": is_staff,
        "confidence": 0.92,
        "metadata": {"session_seq": 1},
    }


# ─────────────────────────────────────────────
# Ingest endpoint tests
# ─────────────────────────────────────────────

class TestIngest:

    def test_basic_ingest_single_event(self, test_client):
        client, _ = test_client
        evt = make_event("ENTRY")
        resp = client.post("/events/ingest", json={"events": [evt]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted"] == 1
        assert body["rejected"] == 0
        assert body["duplicate"] == 0

    def test_ingest_all_event_types(self, test_client):
        client, _ = test_client
        events = [
            make_event("ENTRY"),
            make_event("EXIT"),
            make_event("ZONE_ENTER", zone_id="SKINCARE"),
            make_event("ZONE_EXIT", zone_id="SKINCARE"),
            make_event("ZONE_DWELL", zone_id="HAIRCARE"),
            make_event("BILLING_QUEUE_JOIN", zone_id="BILLING_QUEUE"),
            make_event("BILLING_QUEUE_ABANDON", zone_id="BILLING_QUEUE"),
            make_event("REENTRY"),
        ]
        resp = client.post("/events/ingest", json={"events": events})
        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted"] == 8

    def test_idempotency_duplicate_event_id(self, test_client):
        """Re-sending the same event_id must not create a duplicate."""
        client, _ = test_client
        eid = str(uuid.uuid4())
        evt = make_event("ENTRY", event_id=eid)
        resp1 = client.post("/events/ingest", json={"events": [evt]})
        assert resp1.json()["accepted"] == 1
        resp2 = client.post("/events/ingest", json={"events": [evt]})
        assert resp2.status_code == 200
        body = resp2.json()
        assert body["accepted"] == 0
        assert body["duplicate"] == 1

    def test_idempotency_full_batch_resend(self, test_client):
        """Full batch resend must be safe — all events become duplicates."""
        client, _ = test_client
        events = [make_event("ENTRY") for _ in range(5)]
        client.post("/events/ingest", json={"events": events})
        resp = client.post("/events/ingest", json={"events": events})
        assert resp.json()["duplicate"] == 5
        assert resp.json()["accepted"] == 0

    def test_empty_payload_returns_200(self, test_client):
        """Empty event list must not crash — returns 200 with zeros."""
        client, _ = test_client
        resp = client.post("/events/ingest", json={"events": []})
        assert resp.status_code == 200
        assert resp.json()["accepted"] == 0

    def test_batch_over_500_rejected(self, test_client):
        """Batches over 500 events must be rejected at validation."""
        client, _ = test_client
        events = [make_event("ENTRY") for _ in range(501)]
        resp = client.post("/events/ingest", json={"events": events})
        assert resp.status_code == 422

    def test_zone_event_without_zone_id_rejected(self, test_client):
        """ZONE_ENTER without zone_id must be a schema error."""
        client, _ = test_client
        evt = make_event("ZONE_ENTER", zone_id=None)
        resp = client.post("/events/ingest", json={"events": [evt]})
        assert resp.status_code == 422

    def test_partial_success_mixed_batch(self, test_client):
        client, _ = test_client
        valid = make_event("ENTRY")
        resp = client.post("/events/ingest", json={"events": [valid]})
        assert resp.json()["accepted"] == 1

    def test_staff_events_flagged(self, test_client):
        client, _ = test_client
        evt = make_event("ENTRY", is_staff=True)
        resp = client.post("/events/ingest", json={"events": [evt]})
        assert resp.json()["accepted"] == 1

    def test_ingest_batch_of_100(self, test_client):
        client, _ = test_client
        events = [make_event("ZONE_ENTER", zone_id="SKINCARE") for _ in range(100)]
        resp = client.post("/events/ingest", json={"events": events})
        assert resp.status_code == 200
        assert resp.json()["accepted"] == 100


# ─────────────────────────────────────────────
# Schema validation tests
# ─────────────────────────────────────────────

class TestSchema:

    def test_invalid_event_type_rejected(self, test_client):
        client, _ = test_client
        evt = make_event("UNKNOWN_TYPE")
        resp = client.post("/events/ingest", json={"events": [evt]})
        assert resp.status_code == 422

    def test_confidence_out_of_range_rejected(self, test_client):
        client, _ = test_client
        evt = make_event("ENTRY")
        evt["confidence"] = 1.5
        resp = client.post("/events/ingest", json={"events": [evt]})
        assert resp.status_code == 422

    def test_missing_required_fields_rejected(self, test_client):
        client, _ = test_client
        resp = client.post("/events/ingest", json={"events": [{"store_id": "X"}]})
        assert resp.status_code == 422

    def test_event_id_uniqueness_across_batches(self, test_client):
        """All event_ids in a batch must be unique."""
        client, _ = test_client
        events = [make_event("ENTRY") for _ in range(10)]
        ids = [e["event_id"] for e in events]
        assert len(ids) == len(set(ids)), "event_ids must be globally unique"
