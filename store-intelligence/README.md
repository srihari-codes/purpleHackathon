# Store Intelligence Platform

**Purplle Tech Challenge 2026 — Round 2**

A complete pipeline from raw CCTV footage to a live store analytics API for Apex Retail.

```
Raw CCTV → Detection (YOLOv8+ByteTrack+OSNet) → Events → FastAPI → Live Dashboard
```

---

## Quick Start (5 commands)

```bash
# 1. Clone and enter directory
git clone <your-repo-url> store-intelligence && cd store-intelligence

# 2. Start the API
docker compose up -d

# 3. Verify it's running
curl http://localhost:8000/health

# 4. Run tests
pip install -r requirements.txt
pytest tests/ --cov=app --cov-report=term-missing

# 5. (Optional) Run the live dashboard
python dashboard/app.py --store-id STORE_BLR_002
```

---

## Running the Detection Pipeline Against Clips

### Step 1 — Install pipeline dependencies
```bash
pip install ultralytics opencv-python-headless torch torchvision
# For Re-ID (optional but recommended):
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

### Step 2 — Prepare your clips
Name clips using this convention (enables auto-parsing):
```
data/clips/{STORE_ID}__{CAMERA_ID}__{YYYY-MM-DDTHH-MM-SSZ}.mp4

# Examples:
data/clips/STORE_BLR_002__CAM_ENTRY_01__2026-03-03T14-00-00Z.mp4
data/clips/STORE_BLR_002__CAM_FLOOR_01__2026-03-03T14-00-00Z.mp4
data/clips/STORE_BLR_002__CAM_BILLING_01__2026-03-03T14-00-00Z.mp4
```

### Step 3 — Process all clips (one command)
```bash
bash pipeline/run.sh data/clips data/store_layout.json http://localhost:8000
```

Events are:
- POSTed to the API in real time (batches of 200)
- Written to `data/events/{clip_name}.jsonl` for audit

### Step 4 — Single clip (manual)
```bash
python -m pipeline.detect \
  --clip data/clips/STORE_BLR_002__CAM_ENTRY_01__2026-03-03T14-00-00Z.mp4 \
  --store-id STORE_BLR_002 \
  --camera-id CAM_ENTRY_01 \
  --camera-type entry_exit \
  --layout data/store_layout.json \
  --clip-start "2026-03-03T14:00:00Z" \
  --output-jsonl data/events/entry.jsonl \
  --api-url http://localhost:8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/events/ingest` | Ingest batch of events (max 500, idempotent) |
| GET | `/stores/{id}/metrics` | Unique visitors, conversion rate, dwell, queue |
| GET | `/stores/{id}/funnel` | Entry→Zone→Billing→Purchase funnel |
| GET | `/stores/{id}/heatmap` | Zone visit frequency heatmap (0-100) |
| GET | `/stores/{id}/anomalies` | Queue spike, conversion drop, dead zone |
| GET | `/health` | Service status + per-store feed lag |

### Example Queries
```bash
# Ingest events
curl -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d '{"events": [...]}'

# Get store metrics
curl http://localhost:8000/stores/STORE_BLR_002/metrics

# Get funnel
curl http://localhost:8000/stores/STORE_BLR_002/funnel

# Get active anomalies
curl http://localhost:8000/stores/STORE_BLR_002/anomalies

# Health check
curl http://localhost:8000/health
```

---

## Live Dashboard (Part E)

The terminal dashboard shows real-time metrics, anomalies, zone heatmap, and funnel,
updating every 2 seconds from the live API.

```bash
python dashboard/app.py --store-id STORE_BLR_002 --api-url http://localhost:8000
```

Local URL: `http://localhost:8000` (terminal dashboard, no browser needed)

---

## Architecture

See [`docs/DESIGN.md`](docs/DESIGN.md) for full architecture and AI-Assisted Decisions.

```
store-intelligence/
├── pipeline/          # Detection + tracking + event emission
│   ├── detect.py      # Main CV pipeline (YOLOv8 + ByteTrack)
│   ├── tracker.py     # OSNet Re-ID + visitor session management
│   ├── emit.py        # Event schema builder + API batcher
│   └── run.sh         # One-command: process all clips
├── app/               # FastAPI intelligence API
│   ├── main.py        # Entrypoint + middleware
│   ├── models.py      # Pydantic schemas
│   ├── ingestion.py   # Idempotent ingest
│   ├── metrics.py     # Real-time metrics
│   ├── funnel.py      # Conversion funnel
│   ├── heatmap.py     # Zone heatmap
│   ├── anomalies.py   # Anomaly detection
│   ├── health.py      # Health endpoint
│   └── database.py    # SQLAlchemy SQLite
├── dashboard/
│   └── app.py         # Rich terminal live dashboard
├── tests/
│   ├── test_pipeline.py   # Ingest + schema tests
│   ├── test_metrics.py    # Metrics + funnel + heatmap tests
│   └── test_anomalies.py  # Anomaly + health tests
├── data/
│   ├── clips/         # Drop CCTV clips here
│   ├── store_layout.json
│   └── pos_transactions.csv
├── docs/
│   ├── DESIGN.md      # Architecture + AI decisions
│   └── CHOICES.md     # 3 key design decisions
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v --cov=app --cov-report=term-missing
```

Coverage target: >70% (app/ modules)

---

## Migrating to PostgreSQL

Change one environment variable in `docker-compose.yml`:

```yaml
environment:
  - DATABASE_URL=postgresql://user:password@postgres:5432/store_intelligence
```

Then add a postgres service. No application code changes required.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Detection model | YOLOv8n | CPU-viable for offline batch, built-in ByteTrack |
| Re-ID | OSNet (x0.25) + bbox fallback | Zero-shot cross-camera matching |
| visitor_id scope | Stable across day | Prevents re-entry inflation |
| Database | SQLite (→ PostgreSQL) | Zero-ops for hackathon, portable schema |
| Conversion attribution | 5-min billing zone window | Standard POS-less retail attribution |

Full reasoning in [`docs/CHOICES.md`](docs/CHOICES.md).
