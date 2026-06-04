# Store Intelligence — Architecture & Design

## Overview

Store Intelligence is an end-to-end retail analytics platform that converts raw CCTV footage into a queryable live analytics API. The system is built around four tightly-coupled stages: a computer-vision detection layer, a structured event stream, a sessionisation and analytics engine, and a live web dashboard. Every design decision connects back to the north-star metric: **offline store conversion rate**.

---

## System Architecture

```
Raw CCTV Clips (CAM 1–5)
        │
        ▼
┌───────────────────────────────────────────────┐
│  Detection Layer  (pipeline/detect.py)         │
│                                               │
│  ┌─────────────┐   ┌──────────────────────┐   │
│  │ YOLODetector │──▶│ VisitorIdentityMgr   │   │
│  │ YOLO11m +   │   │ 8-signal consensus   │   │
│  │ ByteTrack   │   │ Re-ID engine         │   │
│  └─────────────┘   └──────────────────────┘   │
│          │                  │                  │
│  ┌───────▼──────┐  ┌────────▼────────────┐    │
│  │ StaffTracker │  │ ZoneDwellTracker    │    │
│  │ HSV+zone rule│  │ 30s ZONE_DWELL emit │    │
│  └──────────────┘  └─────────────────────┘    │
│          │                  │                  │
│  ┌───────▼──────────────────▼──────────────┐  │
│  │         EventEmitter                    │  │
│  │  events.jsonl  +  WebSocket broadcast   │  │
│  └─────────────────────────────────────────┘  │
└───────────────────────────────────────────────┘
        │                    │
        │ events.jsonl       │ WebSocket
        ▼                    ▼
┌─────────────────┐  ┌───────────────────────┐
│  Intelligence   │  │  Live Dashboard        │
│  API (FastAPI)  │  │  wizard.html           │
│                 │  │  Real-time metrics     │
│  /ingest        │  │  Camera frames         │
│  /metrics       │  │  Event log             │
│  /funnel        │  └───────────────────────┘
│  /heatmap       │
│  /anomalies     │
│  /health        │
└─────────────────┘
```

### Stage 1 — Detection Layer

The detection layer runs inside a Docker container (`store_detection_pipeline`) and processes all camera clips concurrently. Each camera is handled by a `CameraProcessor` instance running in a shared thread pool.

**Person detection** uses YOLO11m via the `ultralytics` library with ByteTrack as the tracker (`tracker="bytetrack.yaml"`). The model runs on CUDA if the NVIDIA Container Runtime is present; the system automatically falls back to CPU. A secondary fallback using MOG2 background subtraction is available for environments where `ultralytics` cannot be installed.

**Re-ID and identity persistence** is handled by `VisitorIdentityManager`, which uses an eight-signal consensus vote: spatial proximity, appearance embedding similarity (colour histogram), trajectory trajectory score, zone continuity, temporal plausibility, camera handoff heuristics, ghost-track Kalman prediction, and shadow-track velocity extrapolation. The consensus threshold is calibrated per-store by `CalibrationEngine`.

**Timestamp derivation**: each frame timestamp is `camera_start_time + (frame_index / fps)`. Camera start time is OCR'd from the first frame watermark using Tesseract; cameras where OCR fails inherit the median timestamp from successfully OCR'd cameras.

**Staff classification** combines two signals: black-clothing detection via HSV colour analysis, and zone-rule override — anyone whose centroid lies within a staff-only or billing-counter polygon is unconditionally classified as staff (`is_staff=true`).

**Entry / exit detection** uses a configurable crossing line on Camera 3 (`CAM_ENTRY_03`). The line is loaded from `zones_override.json` if a manual calibration has been performed, otherwise falls back to the default normalised y-position. Direction is determined by which side of the line the centroid moves from and to across consecutive frames.

**Billing queue tracking** on Camera 5 (`CAM_BILLING_05`) uses `QueueTracker`, which monitors visitors present in `ZONE_BILLING_QUEUE` and emits `BILLING_QUEUE_JOIN` and `BILLING_QUEUE_ABANDON` events based on dwell time relative to subsequent POS transactions.

### Stage 2 — Event Stream

All detection events are written to a JSONL file (`/data/events.jsonl`) and simultaneously broadcast over a WebSocket connection to the live dashboard. The event schema is enforced at emission time by the `StoreEvent` dataclass in `pipeline/events.py`.

The schema exactly matches the specification, plus several enrichment fields that carry no additional API cost but materially improve debuggability: `behavior_state` (from a finite-state machine tracking ENTERED → BROWSING → QUEUEING → EXITED), `reid_score`, `reentry_count`, `session_duration_ms`, `wait_duration_ms`, and the four-component confidence lineage (`det_conf × track_conf × reid_conf × zone_conf`).

### Stage 3 — Intelligence API

The API is a FastAPI application (`app/main.py`) running in the `store_intelligence_api` container. All state is in-memory (no external database dependency at runtime), built from the ingested event stream via a sessionisation layer.

**Sessioniser** (`app/sessionizer.py`): converts the raw event stream into `VisitorSession` objects. Sessions are keyed by `visitor_id`. Re-entry events reopen the session rather than creating a new visitor, preserving the de-duplication invariant required for accurate conversion rate.

**Projections** (`app/projections.py`): all five API endpoints read from pre-computed, on-demand projections over the session store. Projections are never cached between requests — each call recomputes from the live session state, ensuring the "real-time" guarantee.

**POS Correlation** (`app/correlation.py`): a visitor session is marked as converted if the visitor joined the billing queue within the 5-minute window before any POS transaction timestamp for the same store.

**VerifierEngine** (`app/verifier.py`): runs integrity checks after each ingest batch — `QUEUE_DEPTH_NEGATIVE`, `CONFIDENCE_CLIFF`, `REENTRY_TOO_FAST`, `DUPLICATE_ACTIVE_SESSION`. Results feed directly into the `/anomalies` endpoint.

**ReplayEngine** (`app/replay.py`): allows deterministic re-ingestion of a historical `events.jsonl` file, resetting state before replay to guarantee reproducibility. Used in testing and for the `/stores/{id}/replay` diagnostic endpoint.

### Stage 4 — Live Dashboard

The dashboard (`pipeline/wizard.html`) is a single-page web application served by the pipeline container on port 8080. It connects to the WebSocket server (`pipeline/gui_server.py`) to receive real-time annotated camera frames and event data, updating visitor counts, queue depth, and the event log without page refresh. A multi-camera onboarding wizard handles new store configuration, zone calibration upload, and pipeline start/stop.

---

## AI-Assisted Decisions

### 1. Eight-Signal Consensus Re-ID vs. Simple Embedding Distance

Early in development the Re-ID layer used a single cosine-distance check against a colour histogram embedding — straightforward to implement but brittle when two visitors wore similar colours or when a track was briefly occluded. I asked an LLM to review the failure modes and suggest an improvement. It proposed a probabilistic voting architecture where each signal (spatial, appearance, trajectory, zone, temporal, camera handoff, ghost-track, shadow-track) casts a weighted vote and the identity is accepted only when the consensus score crosses a calibrated threshold.

I agreed with the direction but overrode the suggested implementation in two ways: (a) I kept ghost tracks as a Kalman-filter prediction rather than the LLM's suggestion of a fixed-velocity extrapolation, because a Kalman filter gracefully degrades confidence when the track is lost for many frames, which is important for accurate confidence calibration; (b) I removed the LLM's proposed "colour histogram decay" — in practice, customers change zone lighting enough that a histogram from 30 seconds ago actively hurts more than it helps.

### 2. In-Memory Projections vs. a Persistent Time-Series Store

When designing the API, the LLM initially suggested using TimescaleDB (a PostgreSQL extension for time-series data) because it would naturally support the 7-day rolling average needed for the `CONVERSION_DROP` anomaly. I considered this seriously but chose an in-memory approach backed by a single `SessionStore` dict for three reasons: (1) `docker compose up` with no additional infrastructure dependency is a hard submission requirement; (2) for 5 stores × 3 cameras × 20-minute clips the dataset fits comfortably in memory; (3) the 7-day historical average can be stubbed as a configurable parameter (`historical_avg_rate`) that a production deployment would populate from a cold-storage query. The LLM's suggestion was architecturally correct for production at scale — I documented it in CHOICES.md and kept the interface open for future injection.

### 3. Staff Detection: HSV + Zone Rule vs. VLM-based Classification

I explored using a Vision Language Model (GPT-4V / Gemini Vision) for staff detection by prompting it to classify bounding-box crops as "staff" or "customer" based on uniform colour. The LLM itself suggested this approach when I described the uniform-detection problem. In practice, the crops from a 1080p/15fps CCTV feed at typical retail distances are too small (often 50×120 px after blur) for a VLM to make a reliable classification — the latency cost (300–800 ms per crop via API) would also make real-time detection infeasible.

I kept the HSV black-clothing detector (tuned thresholds: Hue 0–180, Saturation 0–80, Value 0–80 for black; loosened by ±15 for mixed-lighting clips) as the primary signal and overrode it with a deterministic zone rule: any person whose centroid is in `ZONE_CASH_COUNTER` is unconditionally staff. This hybrid avoids both the latency problem and the lighting sensitivity of pure colour classification. The zone override catches every cashier in frame even when they wear non-standard clothing.

---

## Key Trade-offs & Constraints

| Decision | Choice | Rationale |
|---|---|---|
| Detection model | YOLO11m + ByteTrack | Best speed/accuracy trade-off; ByteTrack handles occlusion better than DeepSORT in retail |
| Re-ID approach | 8-signal consensus | Resilient to appearance changes; explainable via confidence lineage |
| Storage | In-memory (dict) | Zero external dependency; fits within hackathon clip volume |
| API framework | FastAPI | Async, auto-docs, Pydantic validation built in |
| Staff detection | HSV + zone rule | Real-time feasible; VLM latency unacceptable for 15fps |
| Timestamp derivation | OCR watermark + frame offset | Accurate to frame level; degrades gracefully when OCR fails |
| Dashboard | WebSocket web UI | Scores higher than terminal; real-time frame streaming proves live connection |

---

## Confidence Propagation

Every emitted event carries a four-component confidence score:

```
final_confidence = det_conf × track_conf × reid_conf × zone_conf
```

- **`det_conf`**: YOLO detection confidence (0.35 minimum threshold)
- **`track_conf`**: ByteTrack track quality score
- **`reid_conf`**: Consensus Re-ID vote score
- **`zone_conf`**: Fixed per-camera constant (configurable; default 0.85)

Low-confidence events are **never silently dropped**. They are emitted with their actual confidence value and optionally flagged by the `VerifierEngine` via a `CONFIDENCE_CLIFF` warning when a visitor's confidence drops by more than 0.4 in a single event transition.

---

## Production Hardening

- **Idempotent ingest**: `event_id` (UUID v4) is used as a deduplication key in `EventStore._seen_ids`; replaying the same payload returns `duplicates=N` without side effects.
- **Structured logging**: every HTTP request emits `trace_id`, `store_id`, `endpoint`, `latency_ms`, `event_count`, `status_code` as JSON to stdout.
- **Graceful degradation**: all exceptions are caught at the middleware level; no raw stack traces in responses. Storage failures return HTTP 503 with a structured `{"error": "STORAGE_UNAVAILABLE", "message": "..."}` body.
- **STALE_FEED detection**: `/health` computes `lag_sec` per store and raises `STALE_FEED` if the last event is more than 10 minutes old, aggregating warning strings in a top-level `"warnings"` list.
- **Strict Pydantic Type Validation**: Enforces strict type checking (e.g., rejecting coerced type strings for boolean and integer fields like `is_staff` and `dwell_ms`) and validates that `event_id` is a valid UUID-v4 to ensure schema compliance.
- **Zero-Traffic & POS-only Store Support**: Gracefully supports queries for stores that have POS transactions loaded but no CCTV event traffic, returning 200 OK responses with zero-initialized metrics rather than throwing a 404 error.
- **GPU/CPU fallback**: `./run.sh` detects the NVIDIA Container Runtime and automatically uses the CPU docker-compose profile if GPU is unavailable.
