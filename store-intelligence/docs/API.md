# API Endpoints Reference Guide

The Store Intelligence API runs by default on `http://localhost:8000`. This guide details all available endpoints, their query/body formats, and example `curl` commands.

---

## 1. System Health Check

### `GET /health`
Checks the API service health, startup time, and monitors incoming video feeds for lag/stale signals.

- **Status Code**: `200 OK`
- **Output Schema**:
  ```json
  {
    "status": "OK" | "DEGRADED",
    "started_at": "ISO-8601 UTC timestamp",
    "checked_at": "ISO-8601 UTC timestamp",
    "stores": [
      {
        "store_id": "string",
        "last_event_at": "ISO-8601 UTC timestamp",
        "lag_sec": 0.0,
        "stale_feed": true | false,
        "session_count": 0,
        "active_sessions": 0,
        "status": "HEALTHY" | "STALE_FEED"
      }
    ],
    "store_count": 0
  }
  ```

#### Example Command
```bash
curl -s http://localhost:8000/health | jq .
```

---

## 2. Event Ingestion Pipeline

### `POST /events/ingest`
Batch-ingests a list of raw CCTV detection events. It supports partial success, deduplicates events by `event_id` (idempotency key), and automatically invokes the sessionizer state machine and verifier check loops.

- **Body Schema (`IngestRequest`)**:
  ```json
  {
    "events": [
      {
        "event_id": "string (UUID-v4)",
        "store_id": "string",
        "camera_id": "string",
        "visitor_id": "string",
        "event_type": "ENTRY" | "EXIT" | "REENTRY" | "ZONE_ENTER" | "ZONE_EXIT" | "ZONE_DWELL" | "BILLING_QUEUE_JOIN" | "BILLING_QUEUE_ABANDON",
        "timestamp": "ISO-8601 UTC timestamp (Z)",
        "zone_id": "string | null",
        "dwell_ms": 0,
        "is_staff": false,
        "confidence": 1.0,
        "metadata": {
          "queue_depth": null,
          "sku_zone": null,
          "session_seq": null
        }
      }
    ]
  }
  ```
- **Response Schema (`IngestResponse`)**:
  ```json
  {
    "accepted": 0,
    "duplicates": 0,
    "rejected": 0,
    "errors": [
      {
        "event_id": "string",
        "reason": "string"
      }
    ]
  }
  ```

#### Example Command
```bash
curl -s -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {
        "event_id": "f8a0efb3-a1bf-4f24-9b93-8b7f0e0c0342",
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_ENTRY_03",
        "visitor_id": "VISITOR_9999",
        "event_type": "ENTRY",
        "timestamp": "2026-06-04T09:00:00.000Z",
        "confidence": 0.95
      }
    ]
  }' | jq .
```

---

## 3. Store Analytics & Projections

### `GET /stores/{store_id}/metrics`
Returns aggregate customer metrics, conversion rates (excluding staff), average dwell time per zone, current billing queue depth, and active session ratios.

#### Example Command
```bash
curl -s http://localhost:8000/stores/STORE_BLR_002/metrics | jq .
```

- **Output Example**:
  ```json
  {
    "store_id": "STORE_BLR_002",
    "unique_visitors": 1970,
    "conversion_rate": 0.0873,
    "converted_count": 172,
    "avg_dwell_per_zone": {
      "ZONE_FOH": 26253,
      "ZONE_BILLING_QUEUE": 54542
    },
    "current_queue_depth": 2,
    "abandonment_rate": 0.0698,
    "total_sessions": 2698,
    "active_sessions": 1893,
    "as_of": "2026-06-04T09:29:38Z"
  }
  ```

---

### `GET /stores/{store_id}/funnel`
Returns the session-based conversion funnel stages: `ENTRY` $\rightarrow$ `ZONE_VISIT` $\rightarrow$ `BILLING_QUEUE` $\rightarrow$ `PURCHASE`.

#### Example Command
```bash
curl -s http://localhost:8000/stores/STORE_BLR_002/funnel | jq .
```

- **Output Example**:
  ```json
  {
    "store_id": "STORE_BLR_002",
    "funnel": [
      {
        "stage": "ENTRY",
        "count": 1970,
        "pct_of_top": 100.0,
        "drop_off_pct": 0.0
      },
      {
        "stage": "ZONE_VISIT",
        "count": 1961,
        "pct_of_top": 99.5,
        "drop_off_pct": 0.5
      },
      {
        "stage": "BILLING_QUEUE",
        "count": 172,
        "pct_of_top": 8.7,
        "drop_off_pct": 91.2
      },
      {
        "stage": "PURCHASE",
        "count": 172,
        "pct_of_top": 8.7,
        "drop_off_pct": 0.0
      }
    ],
    "as_of": "2026-06-04T09:29:38Z"
  }
  ```

---

### `GET /stores/{store_id}/heatmap`
Exposes zone frequency and dwell duration scores normalized to a 0–100 scale for heatmap visualization.

#### Example Command
```bash
curl -s http://localhost:8000/stores/STORE_BLR_002/heatmap | jq .
```

- **Output Example**:
  ```json
  {
    "store_id": "STORE_BLR_002",
    "zones": [
      {
        "zone_id": "ZONE_MAKEUP",
        "visit_count": 842,
        "avg_dwell_ms": 24472,
        "freq_score": 89,
        "dwell_score": 100,
        "combined_score": 94
      }
    ],
    "data_confidence": true,
    "session_count": 2698,
    "as_of": "2026-06-04T09:29:38Z"
  }
  ```

---

### `GET /stores/{store_id}/anomalies`
Identifies active operational anomalies (dead zones, sudden queue spikes, staff inflation ratios, conversion rate drops) and outputs suggested actions.

#### Example Command
```bash
curl -s http://localhost:8000/stores/STORE_BLR_002/anomalies | jq .
```

- **Output Example**:
  ```json
  {
    "store_id": "STORE_BLR_002",
    "anomalies": [
      {
        "type": "DEAD_ZONE",
        "severity": "INFO",
        "message": "Zone ZONE_FOH has had no visits in >30 min",
        "visitor_ids": [],
        "timestamp": "2026-06-04T09:29:38Z",
        "suggested_action": "Check zone ZONE_FOH signage and product placement.",
        "metadata": {
          "zone_id": "ZONE_FOH",
          "minutes_idle": 77429.3
        }
      }
    ],
    "as_of": "2026-06-04T09:29:38Z"
  }
  ```

---

## 4. Diagnostics & Maintenance Endpoints (Bonus)

### `GET /stores/{store_id}/audit/{visitor_id}`
Returns a full timeline history showing every camera sighting, line crossing, and zone transition associated with a specific customer.

#### Example Command
```bash
curl -s http://localhost:8000/stores/STORE_BLR_002/audit/visitor_21 | jq .
```

---

### `GET /stores/{store_id}/calibration`
Returns the rolling sensor quality calibrations (like average tracking confidence levels and threshold statuses).

#### Example Command
```bash
curl -s http://localhost:8000/stores/STORE_BLR_002/calibration | jq .
```

---

### `POST /stores/{store_id}/replay`
Deterministically replays raw events recorded in a `.jsonl` file to reconstruct analytics. State is cleared before the replay runs.

- **Body Schema**:
  ```json
  {
    "path": "string (absolute file path)",
    "speed": 0.0,
    "store_id_filter": "string | null"
  }
  ```

#### Example Command
```bash
curl -s -X POST http://localhost:8000/stores/STORE_BLR_002/replay \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/data/events.jsonl",
    "speed": 0.0
  }' | jq .
```
- **Response**:
  ```json
  {
    "source": "/data/events.jsonl",
    "total": 36294,
    "accepted": 36294,
    "duplicates": 0,
    "rejected": 0,
    "elapsed_sec": 36.85
  }
  ```
