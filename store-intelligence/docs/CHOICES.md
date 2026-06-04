# Store Intelligence — Architectural Choices

This document covers three key decisions made during development, including the options considered, what AI tools suggested, and the rationale for what was ultimately chosen.

---

## Decision 1 — Detection Model Selection

### The Problem

The core of the pipeline is person detection and tracking across five simultaneous CCTV feeds at 15 fps and 1080p. The model must handle: grouped entries (2–4 people in the same frame), partial occlusion by shelving units, significant lighting variation (natural, fluorescent, mixed), and face-blurred footage that eliminates face-based identity cues.

### Options Considered

| Option | Pros | Cons |
|---|---|---|
| **YOLOv8n (nano)** | Fastest inference; ~6 ms/frame on GPU | Lower recall on small/occluded persons; misses grouped entries |
| **YOLOv8m (medium)** | Good balance of speed and accuracy | Slightly slower than nano |
| **YOLO11m** | State-of-the-art person-class mAP; better at crowded scenes; native ByteTrack integration in `ultralytics` | Slightly heavier than YOLOv8m |
| **RT-DETR** | Transformer-based; excellent recall on partially occluded objects | 3–4× slower inference; requires more VRAM |
| **MediaPipe Pose** | Skeleton-based, good at grouped scenes | Not a person detector; no tracking IDs |
| **MOG2 background subtraction** | No GPU, zero model download | Very noisy; fails in static camera with people standing still |

### What AI Suggested

When I described the retail CCTV context to an LLM, it recommended starting with YOLOv8n for speed and using StrongSORT for tracking because StrongSORT has better appearance features than ByteTrack. It also suggested using an OSNet Re-ID model from `torchreid` for cross-camera identity matching.

### What I Chose and Why

**Detection: YOLO11m** — The upgrade from YOLOv8m to YOLO11m is effectively free in terms of code complexity (same `ultralytics` API), and benchmarks show meaningfully better person-class precision in crowded scenes, which is exactly the billing-queue and group-entry edge case. The nano variant was ruled out after observing missed detections on small background persons in test frames.

**Tracking: ByteTrack** — I disagreed with the LLM's StrongSORT suggestion for this use case. ByteTrack uses only IoU and motion prediction (no appearance model), making it faster and more deterministic at 15 fps. The appearance modelling StrongSORT adds is useful for surveillance at lower frame rates; at 15 fps with good IoU overlap between frames, ByteTrack's identity persistence is adequate, and I handle appearance-based cross-camera Re-ID at a higher level in `VisitorIdentityManager`.

**Re-ID: custom 8-signal consensus (not OSNet)** — The LLM suggested OSNet from `torchreid`, which would add a ~70 MB model download and a CUDA dependency for a feature that ByteTrack already partially covers within a single camera. I implemented a lightweight 8-signal consensus voter instead (spatial, colour histogram, trajectory, zone, temporal, camera handoff, ghost-track, shadow-track). This runs in pure Python/NumPy with no additional model, is explainable via the confidence lineage, and is easier to calibrate per-store.

**Fallback: MOG2** — Retained as an automatic fallback when `ultralytics` is unavailable. This covers CI environments and machines without a compatible CUDA stack. The fallback produces lower-quality detections but keeps the pipeline functional.

---

## Decision 2 — Event Schema Design

### The Problem

The event schema is the contract between the detection layer and the Intelligence API. It must support all analytics queries (conversion rate, dwell, funnel, queue depth, anomalies) and be forward-compatible with new features without a breaking change.

### Options Considered

**Option A — Minimal spec-only schema**: Emit exactly the fields listed in the problem statement (`event_id`, `store_id`, `camera_id`, `visitor_id`, `event_type`, `timestamp`, `zone_id`, `dwell_ms`, `is_staff`, `confidence`, `metadata.queue_depth`, `metadata.sku_zone`, `metadata.session_seq`). Simple and easy to validate.

**Option B — Enriched schema with optional extra fields**: Add enrichment fields to the `metadata` block that carry no breaking changes to consumers but materially improve debuggability and future analytics. Fields default to `null` when not applicable.

### What AI Suggested

The LLM proposed Option A with the addition of a `behavior_state` enum (ENTERED, BROWSING, DWELLING, QUEUEING, EXITED) derived from a finite-state machine. It also proposed splitting the schema into two separate event types: "raw detection events" from the pipeline and "session events" computed by the API, with a Kafka topic between them. It noted that embedding confidence components directly into the event would "pollute the schema."

### What I Chose and Why

**Option B** — The enriched schema. I kept all required fields from the spec and added:

- `behavior_state` — agreed with the LLM's FSM suggestion; it makes API-level anomaly detection much cleaner
- `reid_score` — the composite Re-ID confidence at the moment of identity association; essential for diagnosing phantom visitors
- `reentry_count` — tracks how many times this visitor_id has cycled through ENTRY→EXIT
- `session_duration_ms` — time since the visitor's first_seen timestamp; useful for session-level analytics without a join
- `wait_duration_ms` — billing queue abandon wait time; enables future queue abandonment SLA queries
- `det_conf`, `track_conf`, `reid_conf`, `zone_conf` — the four confidence pipeline components explicitly stored; this lets operators spot systematic detection drops per camera without re-running inference

I **disagreed with the LLM on the Kafka split**. For a 5-store, 3-camera hackathon submission, introducing a message broker would add operational complexity (another Docker service, offset management, serialisation format decisions) for no functional benefit. The JSONL → REST ingest pattern achieves the same decoupling and is already used in production pipelines at much larger scale than this challenge requires.

The "polluted schema" concern is a style preference. Since `metadata` is already a free-form JSON object, the extra fields are invisible to consumers that don't read them, and they add zero overhead on the wire.

---

## Decision 3 — API Architecture: In-Memory Projections vs. Persistent Database

### The Problem

The API must serve real-time metrics — not cached from yesterday. It needs to correlate events with POS transactions and compute conversion rates, funnel stages, heatmaps, and anomalies in response to each request. The question is where and how that state is stored.

### Options Considered

| Option | Description | Pros | Cons |
|---|---|---|---|
| **SQLite** | Write events to a local SQLite DB; query with SQL at request time | Persistent across restarts; SQL is expressive | Requires schema migrations; WAL-mode needed for concurrent reads/writes |
| **PostgreSQL + TimescaleDB** | Time-series optimised; native 7-day rolling window queries | Production-grade; scales to 40 stores | Needs a fourth Docker service; adds significant operational weight |
| **Redis** | In-memory, fast; sorted sets for time-series | Sub-ms reads; TTL-based staleness | No built-in complex analytics; manual aggregation |
| **In-memory Python dicts** | All state in `SessionStore` / `EventStore` dicts held in the API process | Zero external dependency; instant reads; testable in isolation | Lost on restart; single-process only |

### What AI Suggested

The LLM recommended **TimescaleDB** because it would natively answer the "conversion rate vs 7-day rolling average" query in the `/anomalies` endpoint with a single SQL window function. It also pointed out that in-memory storage would lose all data if the API container restarted.

The LLM also suggested using a **Redis cache** in front of whatever database I chose, to keep response latency under 50 ms under load.

### What I Chose and Why

**In-memory Python dicts**, for the following reasons:

1. **Acceptance gate requirement**: `docker compose up` must start everything with no manual steps. Adding TimescaleDB or Redis adds a third and potentially fourth container, each with first-boot initialisation time and port conflicts on developer machines.

2. **Data volume**: 5 stores × 3 cameras × 20-minute clips produces approximately 50,000–200,000 events. This is trivially small for Python dict lookups. A `VisitorSession` object with a full event history is roughly 2–3 KB; 10,000 sessions is ~30 MB.

3. **The 7-day rolling average** is architecturally honest: the system doesn't have 7 days of data — it has 20-minute clips. I modelled `historical_avg_rate` as an injectable parameter (`historical_avg_rate=0.0` in the current build) with a clear comment that a production deployment would populate it from a cold-storage query. This is more honest than fabricating a "7-day average" from 20 minutes of data.

4. **Test isolation**: in-memory state can be cleared between test cases with a single `.clear()` call, making the test suite fast and deterministic. With TimescaleDB or Redis this would require truncating tables between each test, or spinning up a test container.

I **agreed with the LLM's concern about restart durability** and addressed it pragmatically: the pipeline emits all events to `events.jsonl`, and the `ReplayEngine` (`app/replay.py`) can re-ingest that file via `POST /stores/{id}/replay` to restore API state after a restart. This achieves durability without external infrastructure.

The Redis caching suggestion was valid but out of scope for this challenge: at 5 stores and sub-second projection computation, caching would reduce latency from ~5 ms to ~1 ms — immeasurable to a human user.

---

## Storage Engine Note

As documented above, SQLite was the other strong candidate for a persistent option if zero-restart-durability were a hard requirement. The `CorrelationEngine` already parses POS CSV files into Python objects on startup; SQLite would replace the in-memory dict with a disk-backed B-tree, requiring minimal code change. This is documented in code comments as the recommended upgrade path for production deployment.
