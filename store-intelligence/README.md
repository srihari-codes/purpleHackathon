# Purplle Store Intelligence — Quickstart Guide

A complete retail analytics platform: raw CCTV footage → live store metrics API.

- **Detection**: YOLO11m + ByteTrack + 8-signal Re-ID consensus
- **API**: FastAPI with real-time sessions, funnel, heatmap, and anomaly detection
- **Dashboard**: Live WebSocket web UI at `http://localhost:8080`
- **Docs**: [`docs/DESIGN.md`](docs/DESIGN.md) · [`docs/CHOICES.md`](docs/CHOICES.md)

---

## 5-Command Quickstart

```bash
git clone <your-repository-url>
cd purpleHackathon/store-intelligence
cp /path/to/your/clips/*.mp4 data/clips/       # place CAM 1.mp4 … CAM 5.mp4
./run.sh                                         # starts pipeline + API + calibration UI
curl http://localhost:8000/health               # verify API is running
```

That's it. The detection pipeline processes all clips, writes events to `data/events.jsonl`, and the API is immediately queryable.

---

## Prerequisites

Ensure you have the following installed on your host system:
1. **Docker** and **Docker Compose**
2. **Git**
3. (Optional but recommended) **NVIDIA Container Toolkit** for full GPU/CUDA acceleration. The startup script will automatically fall back to CPU mode if NVIDIA GPU drivers are not present.

---

## Prerequisites

Ensure you have the following installed on your host system:
1. **Docker** and **Docker Compose**
2. **Git**
3. (Optional but recommended) **NVIDIA Container Toolkit** for full GPU/CUDA acceleration. The startup script will automatically fall back to CPU mode if NVIDIA GPU drivers are not present.

---

## Setup & Running Instructions

### Step 1: Clone the Repository
Clone the repository and navigate to the project directory:
```bash
git clone <your-repository-url>
cd purpleHackathon
```

---

### Step 2: Position the CCTV Footage
The detection pipeline requires the 5 CCTV video feeds (`CAM 1.mp4`, `CAM 2.mp4`, `CAM 3.mp4`, `CAM 4.mp4`, and `CAM 5.mp4`).

Choose **one** of the two setup methods below:

#### Option A: Project Local Clips Folder (Recommended & Easiest)
Simply place your 5 `CAM *.mp4` files inside the project's local clips directory:
`store-intelligence/data/clips/`

#### Option B: Use any Custom Folder
You can keep the files anywhere on your machine (e.g., inside your `Downloads/CCTV Footage` folder) and pass the path using the `CLIPS_DIR` environment variable when starting.

---

### Step 3: Start the Pipeline

First, navigate into the project's root folder:
```bash
cd store-intelligence
```

Depending on the setup option you chose in **Step 2**, run the corresponding command:

#### If you set up Option A (Default folder):
Simply execute the run script:
```bash
./run.sh
```

#### If you set up Option B (Custom folder):
Provide the absolute path to your folder via `CLIPS_DIR`:
```bash
CLIPS_DIR="/path/to/your/CCTV Footage" ./run.sh
```

---

## Overall Docker Design

* **Automatic Hardware Detection:** The `./run.sh` script automatically checks for GPU availability. If CUDA is detected, it configures complete hardware-accelerated tracking. If not, it falls back to a clean CPU pipeline automatically.
* **Events File Persistence:** Output events are written to `store-intelligence/data/events.jsonl`. This file is persisted on your host and is ignored in git so that your local testing outputs do not conflict with the code repository.
* **Development Code Mounting:** The local `pipeline/` directory is mounted directly into the container in real-time, allowing you to edit python modules on your host and see changes instantly without rebuilding.

---

## Running the Services Separately

### 1. View the Main GUI Dashboard
Once the container starts, open your browser and navigate to:
```text
http://localhost:8080
```
This launches the real-time event analytics dashboard. Metrics (active visitors, queue depth, event count) update live as the detection pipeline processes frames.

### 2. Run the Interactive Calibration UI
If you need to calibrate entry/exit crossing lines or polygon boundaries, run:
```bash
# If Option A
docker compose up calibrate

# If Option B
CLIPS_DIR="/path/to/your/CCTV Footage" docker compose up calibrate
```
Then navigate to `http://localhost:8081` in your browser. Any custom polygons you draw will be saved directly back to `pipeline/zones_override.json` on your host machine.

---

## How Detection Output Feeds Into the API

The detection pipeline and the Intelligence API are connected in two ways:

### Automatic (default)
`./run.sh` starts both the `pipeline` and `api` containers. The pipeline writes events to `/data/events.jsonl` (host path: `store-intelligence/data/events.jsonl`). The pipeline also ingests events into the API automatically via `POST /events/ingest` as they are emitted.

### Manual replay (for debugging or re-ingestion)
If the API container restarted and lost in-memory state, replay events from the JSONL file without re-running detection:

```bash
# Replay all events from the saved JSONL into the API
curl -X POST http://localhost:8000/stores/STORE_BLR_002/replay \
  -H "Content-Type: application/json" \
  -d '{"path": "/data/events.jsonl"}'
```

### Manual batch ingest
You can also send events directly from the JSONL file in batches:

```bash
# Read events.jsonl and POST in batches of 100
python3 -c "
import json, requests, itertools
with open('data/events.jsonl') as f:
    events = [json.loads(l) for l in f if l.strip()]
for i in range(0, len(events), 100):
    batch = events[i:i+100]
    r = requests.post('http://localhost:8000/events/ingest', json={'events': batch})
    print(f'Batch {i//100+1}: accepted={r.json()[\"accepted\"]}')
"
```

---

## API Reference

The Intelligence API runs on **`http://localhost:8000`**.

| Endpoint | Description |
|---|---|
| `POST /events/ingest` | Ingest batch of events (up to 500). Idempotent. |
| `GET /stores/{id}/metrics` | Unique visitors, conversion rate, dwell, queue depth, abandonment rate |
| `GET /stores/{id}/funnel` | Entry → Zone Visit → Billing Queue → Purchase with drop-off % |
| `GET /stores/{id}/heatmap` | Zone visit frequency + avg dwell, normalised 0–100 |
| `GET /stores/{id}/anomalies` | Active anomalies: queue spike, conversion drop, dead zones |
| `GET /health` | Service status + STALE_FEED warnings per store |
| `GET /stores/{id}/audit/{visitor_id}` | Full audit trail for a single visitor |
| `POST /stores/{id}/replay` | Replay a historical events.jsonl for debugging |

### Quick cURL examples

```bash
# Check service health
curl http://localhost:8000/health

# Get store metrics
curl http://localhost:8000/stores/STORE_BLR_002/metrics

# Get conversion funnel
curl http://localhost:8000/stores/STORE_BLR_002/funnel

# Get zone heatmap
curl http://localhost:8000/stores/STORE_BLR_002/heatmap

# Get active anomalies
curl http://localhost:8000/stores/STORE_BLR_002/anomalies

# Ingest a test event
curl -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "events": [{
      "event_id": "test-uuid-001",
      "store_id": "STORE_BLR_002",
      "camera_id": "CAM_ENTRY_03",
      "visitor_id": "VIS_test01",
      "event_type": "ENTRY",
      "timestamp": "2026-04-10T14:00:00Z",
      "zone_id": null,
      "dwell_ms": 0,
      "is_staff": false,
      "confidence": 0.91,
      "metadata": {"queue_depth": null, "sku_zone": null, "session_seq": 1}
    }]
  }'
```

---

## Running Tests

```bash
cd store-intelligence
python3 -m pytest tests/ -v                          # run all 20 tests
python3 -m pytest tests/ --cov=app --cov-report=term # coverage report (expect ~83%)
```

All tests pass with no external services required (fully in-memory).

