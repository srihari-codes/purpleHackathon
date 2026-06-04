# Purplle Store Intelligence

> **Raw CCTV footage → Live retail analytics API + real-time dashboard. Zero configuration required.**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org) [![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com) [![YOLO11m](https://img.shields.io/badge/Model-YOLO11m-red)](https://ultralytics.com) [![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://docker.com)

---

## What This Is

A complete, fully dynamic retail intelligence platform built from scratch: drop in CCTV clips, get a live analytics dashboard with real-time annotated video, queue depth tracking, entry/exit counts, zone heatmaps, conversion funnels, and a production-grade REST API.

The system is **fully dynamic**: there are no static files to configure, no hardcoded store IDs, and no service restarts required. Adding a new store, registering sequentially assigned camera slots, drawing region shapes, and setting up line-crossing boundaries are all handled dynamically through the web UI and immediately hot-reloaded by the background containers.

**Stack:**
| Layer | Technology |
|---|---|
| Detection | YOLO11m + ByteTrack |
| Re-ID | 8-signal consensus voter (spatial, appearance, trajectory, zone, temporal, handoff, ghost-track, shadow-track) |
| API | FastAPI + Pydantic strict validation |
| Dashboard | WebSocket single-page app with live annotated frames |
| Containerisation | Docker Compose — GPU auto-detected, CPU fallback |

---

## Quickstart (5 commands)

```bash
git clone <your-repository-url>
cd store-intelligence
cp /path/to/footage/*.mp4 data/clips/   # CAM 1.mp4 … CAM 5.mp4
./run.sh                                 # starts all services
curl http://localhost:8000/health        # verify API is live
```

> **NVIDIA GPU?** Detected automatically. No flags needed. Falls back to CPU cleanly.

---

## The Full Workflow

### Step 1 — Launch

```bash
./run.sh
```

Three containers start:

| Service | URL | Purpose |
|---|---|---|
| `store_detection_pipeline` | `http://localhost:8080` | Live dashboard + onboarding wizard |
| `store_zone_calibrator` | `http://localhost:8081` | Zone calibration studio |
| `store_intelligence_api` | `http://localhost:8000` | REST analytics API |

---

### Step 2 — Onboard via the Wizard (`http://localhost:8080`)

The dashboard opens in **Wizard Mode** when no store is configured. Walk through 4 steps:

**① Store Setup**
- Enter Store Name and Store Code → auto-generates a `STORE_ID`

**② Add Cameras**
- Click `+ Entrance/Exit`, `+ Billing Counter`, `+ Sales Floor`, or `+ Godown/Staff`
- Each camera gets a typed ID: `CAM_ENTRY_01`, `CAM_BILLING_01`, `CAM_FLOOR_01`, etc.
- Upload the matching `.mp4` clip — first frame previewed instantly

**③ Configure Zones** (`http://localhost:8081`)
- Draw polygons directly on first-frame screenshots
- Define: entry/exit lines, billing queue area, product zones, staff-only areas
- Save → calibration JSON written to `data/calibration/{STORE_ID}.json`

**④ Start Analysis**
- Click **Start Analysis** → detection pipeline launches in-container
- Dashboard transitions to **Live View** automatically

---

### Step 3 — Live Dashboard

Once analysis starts:

- **Annotated video feeds** — bounding boxes, track IDs, zone labels, entry/exit arrows streamed via WebSocket
- **Real-time metrics** — active visitors, entries today, exits today, queue depth
- **Live event log** — every detection event scrolls in real time
- **WebSocket status badge** — `🟢 Connected` / `🔴 Disconnected`

---

## API Reference

Base URL: `http://localhost:8000` · Swagger UI: `http://localhost:8000/docs`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/events/ingest` | Ingest up to 500 events. Idempotent by `event_id`. Partial success. |
| `GET` | `/stores/{id}/metrics` | Unique visitors, conversion rate, avg dwell/zone, queue depth, abandonment rate |
| `GET` | `/stores/{id}/funnel` | Entry → Zone Visit → Billing Queue → Purchase with drop-off % |
| `GET` | `/stores/{id}/heatmap` | Zone frequency + avg dwell, normalised 0–100. `data_confidence` flag |
| `GET` | `/stores/{id}/anomalies` | Active anomalies: queue spike, conversion drop, dead zones. Severity + suggested action |
| `GET` | `/health` | Service status, last event timestamp per store, `STALE_FEED` warnings |
| `POST` | `/stores/{id}/replay` | Deterministic replay of `events.jsonl` — resets + re-ingests |
| `GET` | `/stores/{id}/audit/{visitor_id}` | Full audit trail for a single visitor session |

### Quick cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Store metrics
curl http://localhost:8000/stores/STORE_BLR_002/metrics

# Ingest an event
curl -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "events": [{
      "event_id": "550e8400-e29b-41d4-a716-446655440000",
      "store_id": "STORE_BLR_002",
      "camera_id": "CAM_ENTRY_01",
      "visitor_id": "VIS_abc123",
      "event_type": "ENTRY",
      "timestamp": "2026-04-10T14:00:00Z",
      "zone_id": null,
      "dwell_ms": 0,
      "is_staff": false,
      "confidence": 0.91,
      "metadata": {"queue_depth": null, "sku_zone": null, "session_seq": 1}
    }]
  }'

# Replay events after API restart
curl -X POST http://localhost:8000/stores/STORE_BLR_002/replay \
  -H "Content-Type: application/json" \
  -d '{"path": "/data/events.jsonl"}'
```

---

## Event Schema

All pipeline output conforms to this schema:

```json
{
  "event_id":   "uuid-v4",
  "store_id":   "STORE_BLR_002",
  "camera_id":  "CAM_ENTRY_01",
  "visitor_id": "VIS_c8a2f1",
  "event_type": "ZONE_DWELL",
  "timestamp":  "2026-03-03T14:22:10Z",
  "zone_id":    "SKINCARE",
  "dwell_ms":   8400,
  "is_staff":   false,
  "confidence": 0.91,
  "metadata": {
    "queue_depth":         null,
    "sku_zone":            "MOISTURISER",
    "session_seq":         5,
    "behavior_state":      "DWELLING",
    "reid_score":          0.87,
    "reentry_count":       0,
    "session_duration_ms": 142000,
    "wait_duration_ms":    null,
    "det_conf":            0.94,
    "track_conf":          0.98,
    "reid_conf":           0.87,
    "zone_conf":           0.85
  }
}
```

**Event types:** `ENTRY` · `EXIT` · `REENTRY` · `ZONE_ENTER` · `ZONE_EXIT` · `ZONE_DWELL` · `BILLING_QUEUE_JOIN` · `BILLING_QUEUE_ABANDON`

---

## Manual Pipeline (Batch Mode)

Skip the wizard and drive the pipeline directly:

```bash
# Run detection against clips, output to events.jsonl
docker exec store_detection_pipeline python detect.py \
  --store_id   STORE_BLR_002 \
  --clips_dir  /data/clips \
  --camera_map '{"CAM_ENTRY_01":"/data/clips/CAM 3.mp4","CAM_BILLING_01":"/data/clips/CAM 5.mp4"}' \
  --camera_roles '{"CAM_ENTRY_01":"entry","CAM_BILLING_01":"billing"}' \
  --output     /data/events.jsonl \
  --speed      0

# Replay into API
curl -X POST http://localhost:8000/stores/STORE_BLR_002/replay \
  -d '{"path":"/data/events.jsonl"}'
```

---

## Running Tests

```bash
# All 20 tests (no external services needed — fully in-memory)
docker exec store_intelligence_api python3 -m pytest /api/app/../tests/ -v

# With coverage
docker exec store_intelligence_api python3 -m pytest /api/app/../tests/ --cov=app --cov-report=term
```

Expected: **20 passed**, ~83% coverage.

---

## Architecture Overview

```
CCTV Clips (CAM 1–5)
        │
        ▼
  Onboarding Wizard   ←── http://localhost:8080
  (store setup +
   clip upload +
   zone calibration)
        │
        ▼
 Detection Pipeline
 ┌─────────────────────────────────────────┐
 │  YOLO11m → ByteTrack → 8-signal Re-ID  │
 │  StaffTracker │ ZoneDwellTracker        │
 │  EntryExitDetector │ QueueTracker       │
 │         │                  │            │
 │  events.jsonl      WebSocket frames     │
 └─────────────────────────────────────────┘
        │                     │
        ▼                     ▼
  Intelligence API      Live Dashboard
  :8000                 :8080
  /metrics              Annotated video
  /funnel               Queue depth
  /heatmap              Event log
  /anomalies
  /health
```

---

## Project Structure

```
store-intelligence/
├── pipeline/
│   ├── detect.py            # Main detection + tracking orchestrator
│   ├── tracker.py           # 8-signal consensus Re-ID engine
│   ├── events.py            # Event schema + emitter
│   ├── billing_queue.py     # Queue depth tracking (BILLING_QUEUE_JOIN/ABANDON)
│   ├── entry_exit.py        # Line-crossing entry/exit detector
│   ├── staff.py             # HSV + zone-rule staff classifier
│   ├── zones.py             # Zone polygon definitions
│   ├── zone_mapper.py       # Hot-reloadable calibration JSON loader
│   ├── gui_server.py        # WebSocket server + wizard API host
│   ├── wizard_backend.py    # Wizard session management endpoints
│   ├── wizard.html          # Onboarding wizard + live dashboard SPA
│   ├── calibrate_zones.py   # Zone calibration studio backend
│   ├── _calib_ui.html       # Zone drawing UI
│   └── run.sh               # Pipeline entrypoint
├── app/
│   ├── main.py              # FastAPI entrypoint + middleware
│   ├── models.py            # Pydantic event schema (strict validation)
│   ├── ingestion.py         # Ingest, dedup, partial success
│   ├── sessionizer.py       # Raw events → VisitorSession objects
│   ├── projections.py       # /metrics /funnel /heatmap /anomalies
│   ├── correlation.py       # POS ↔ visitor session correlation
│   ├── verifier.py          # Integrity checks + anomaly signals
│   ├── replay.py            # Deterministic event replay engine
│   └── audit.py             # Per-visitor audit timeline
├── tests/
│   └── test_layer2.py       # 20 tests, ~83% coverage (with AI prompt header)
├── docs/
│   ├── DESIGN.md            # Full architecture + AI-assisted decisions
│   └── CHOICES.md           # 5 key decisions with full reasoning
├── data/
│   ├── clips/               # CCTV footage (gitignored)
│   ├── calibration/         # Per-store zone JSON (committed)
│   └── pos_transactions.csv # POS data for conversion correlation
├── docker-compose.yml
├── run.sh                   # GPU/CPU auto-detect + compose up
└── README.md
```

---

## Docs

- [`docs/DESIGN.md`](docs/DESIGN.md) — Full architecture, every module, AI-assisted decisions
- [`docs/CHOICES.md`](docs/CHOICES.md) — Detection model, Re-ID, schema, API storage, all reasoning
- `http://localhost:8000/docs` — Live Swagger UI (auto-generated)
