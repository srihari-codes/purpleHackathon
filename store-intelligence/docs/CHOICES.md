# Store Intelligence — Architectural Choices

> Every decision documented here was made under real constraints: a single CCTV feed that runs at 15fps, faces blurred for privacy, a billing queue that actually builds and disperses, and a hard requirement that `docker compose up` is the only setup step. The reasoning below is exactly how I thought through each one.

---

## Choice 1 — Detection Model: YOLO11m + ByteTrack (not OSNet, not RT-DETR, not StrongSORT)

### The Problem

Five simultaneous 1080p/15fps CCTV feeds. Faces blurred. People partially occluded by shelving. Groups of 2–4 entering together through a narrow door. A billing queue that deepens and disperses over 20 minutes. The model has to handle all of this in real time, or at least at max-speed playback, while producing reliable per-person track IDs that survive occlusions.

### Options I Evaluated

| Option | Why I Looked At It | Why I Rejected It |
|---|---|---|
| **YOLOv8n (nano)** | Fastest — ~6ms/frame GPU | Recall collapses on small/distant persons; missed grouped entries in test frames |
| **YOLOv8m** | Solid baseline | YOLO11m is a free upgrade — same API, better mAP on crowded scenes |
| **YOLO11m** | State-of-art person-class precision; native ByteTrack in `ultralytics` | ✅ **Chosen** |
| **RT-DETR** | Transformer backbone; excellent occluded-object recall | 3–4× slower; needs more VRAM; would have required separate tracker integration |
| **MediaPipe Pose** | Skeleton-based, good at crowds | Not a detector — no bounding boxes, no track IDs |
| **MOG2 background subtraction** | Zero dependencies, zero model size | Fails completely when people stand still; no identity |

### What AI Suggested

When I described the retail CCTV context, the LLM recommended starting with YOLOv8n for speed and upgrading only if accuracy was insufficient. It also strongly suggested StrongSORT for tracking because StrongSORT has an appearance module that improves re-association after occlusion. For Re-ID, it suggested an OSNet model from `torchreid`.

### What I Chose and Why

**YOLO11m:** The jump from YOLOv8m to YOLO11m is free from an integration perspective — same `ultralytics` API call, same `tracker="bytetrack.yaml"`. The precision gain on crowded person detection is real and directly relevant to the group-entry edge case. I rejected the LLM's nano suggestion — I didn't want to discover the accuracy problem after building the whole pipeline.

**ByteTrack over StrongSORT:** I disagree with the LLM here. StrongSORT's appearance model adds value at low frame rates (2–5fps) where IoU overlap between frames is unreliable. At 15fps, IoU overlap is almost always sufficient for within-camera track continuity. ByteTrack is simpler, faster, and more deterministic. The appearance-based Re-ID problem — the genuinely hard one — happens *across* camera handoffs and *after* occlusions longer than a few seconds. I handle that at a higher level in `VisitorIdentityManager`, not in the tracker.

**Custom 8-signal consensus Re-ID over OSNet:** The LLM's OSNet suggestion would have added ~70MB of model download, a `torchreid` CUDA dependency, and an embedding inference step per track per frame. For the cross-camera Re-ID problem at 15fps with 5 cameras, the marginal accuracy gain doesn't justify the infrastructure weight. My 8-signal consensus voter runs in pure Python/NumPy at <1ms per association and is fully explainable via the confidence lineage stored in each event.

**MOG2 retained as fallback:** On machines without CUDA or `ultralytics`, the pipeline falls back to background subtraction. The detections are noisier but the system stays functional.

---

## Choice 2 — Re-ID Architecture: Consensus Voting over Single Embedding

### The Problem

The hardest technical problem in this system is identity persistence. ByteTrack assigns track IDs per-frame within a single camera view. When a person steps behind a shelf for 8 seconds, the track ID is lost. When they reappear, ByteTrack issues a new track ID. Without a Re-ID layer, this person generates multiple `ENTRY` events and inflates unique visitor counts.

### Options I Evaluated

**Option A — Single cosine distance on colour histogram:** Fast to build. Works well when people wear distinctly different colours. Collapses immediately when two people in similar clothing are in the same zone, or when lighting changes between zones.

**Option B — Deep embedding (OSNet/torchreid):** Produces genuinely discriminative 512-dim feature vectors. Real accuracy advantage. But adds 70MB+ model, GPU dependency for real-time inference, and significant latency per crop.

**Option C — Multi-signal consensus voting:** N independent signals, each with a calibrated weight, vote on identity associations. More signals = more resilient to any one signal failing. Fully explainable. Runs in Python/NumPy.

### What AI Suggested

The LLM initially suggested Option B (OSNet) because it produces the most discriminative representations. It also proposed a "colour histogram temporal decay" — weighting recent histograms more heavily than older ones.

### What I Chose and Why

**Option C — 8-signal consensus.** The key insight is that any single signal is brittle. Appearance fails under lighting change. Trajectory fails when a person reverses direction. Zone continuity fails when two people cross zones simultaneously. But when all eight signals agree, the association is extremely reliable.

The eight signals and why each one is there:
1. **Spatial proximity** — The person most likely to be the same track is the one that appeared closest to the predicted position.
2. **Appearance (HSV histogram)** — Colour is stable within a camera's lighting zone.
3. **Trajectory (Kalman)** — Even without a detection, we can predict where the track should be. If the new detection is near the prediction, it's likely the same person.
4. **Zone continuity** — Jumping from `ZONE_SKINCARE` to `ZONE_ACCESSORIES` across the store in one second is physically impossible. Zone transitions gate the association.
5. **Temporal plausibility** — A person who disappeared 45 seconds ago is far less likely to be the same person as one who disappeared 3 seconds ago.
6. **Camera handoff** — A track that disappeared near the edge of Camera 1's FOV should be associated with a new detection near the corresponding edge of Camera 2's FOV.
7. **Ghost track (Kalman prediction)** — When a track is temporarily lost (e.g., person walks behind a tall display), a Kalman filter continues predicting the track's position. New detections near the ghost are strong candidates for re-association.
8. **Shadow track (velocity extrapolation)** — For fast-moving persons (staff walking briskly), pure velocity extrapolation is more accurate than Kalman for short gaps.

I rejected the LLM's histogram decay suggestion because in practice, lighting changes between zones make old histograms actively harmful — a 30-second-old embedding from the entrance zone is misleading when the person is now under fluorescent lighting in the fragrance zone.

---

## Choice 3 — Event Schema: Enriched vs. Minimal Spec

### The Problem

The event schema is the contract between the detection layer and everything downstream. Get it wrong now and every fix is a breaking change.

### Options I Evaluated

**Option A — Emit exactly the spec minimum:** `event_id, store_id, camera_id, visitor_id, event_type, timestamp, zone_id, dwell_ms, is_staff, confidence, metadata.{queue_depth, sku_zone, session_seq}`. Clean, minimal, unambiguous.

**Option B — Enriched schema with optional nullable fields:** Add diagnostic and analytics fields to the `metadata` block. `null`-default means no breaking change for consumers that don't read them.

**Option C — Split schema (Kafka-style):** Separate "raw detection events" from the pipeline and "computed session events" from the API, with an explicit message broker boundary between them.

### What AI Suggested

The LLM recommended Option A as the base, but proposed adding a `behavior_state` enum derived from a finite-state machine. It also proposed Option C (Kafka topic split) as the "production-correct" architecture.

### What I Chose and Why

**Option B — Enriched schema.** The `behavior_state` suggestion I fully agreed with — a per-visitor FSM (`ENTERED → BROWSING → DWELLING → QUEUEING → EXITED`) makes anomaly detection dramatically cleaner and makes the audit trail human-readable.

The additional fields I added beyond the spec minimum:
- `reid_score` — the consensus score at the moment of identity association. Without this, diagnosing phantom visitors requires re-running the entire tracker.
- `reentry_count` — total times this visitor_id has cycled through ENTRY→EXIT. Immediately surfaces re-entry inflation.
- `session_duration_ms` — cumulative session time since first_seen. Enables session-level analytics without a JOIN.
- `wait_duration_ms` — billing queue dwell time at abandon. Enables future queue SLA queries.
- `det_conf, track_conf, reid_conf, zone_conf` — the four pipeline confidence components explicitly. This means operators can isolate whether a systematic drop in `det_conf` is a lighting problem or a model problem, without re-running inference.

I **rejected Option C (Kafka)** hard. The LLM said it was "architecturally correct for production at scale." That's true at 40 stores × 3 cameras × real-time. For a 5-store hackathon with 20-minute clips, adding a message broker adds another Docker service, offset management, serialisation format decisions, and consumer group configuration — for zero functional benefit. The JSONL → REST ingest pattern achieves the same decoupling and is used in production systems at significantly larger scale than this challenge requires.

---

## Choice 4 — API Storage: In-Memory Python Dicts vs. Persistent Database

### The Problem

The API must serve real-time metrics — not cached from yesterday. It needs to correlate events with POS transactions, compute conversion rates, funnel stages, heatmaps, and anomalies on every request. Where does the state live?

### Options I Evaluated

| Option | Pros | Cons |
|---|---|---|
| **SQLite** | Persistent across restarts; SQL is expressive | WAL-mode needed; schema migrations; Docker volume required |
| **PostgreSQL + TimescaleDB** | Native 7-day rolling window SQL; production-grade | Fourth Docker service; significant operational weight; first-boot initialisation time |
| **Redis** | Sub-ms reads; TTL-based staleness; sorted sets for time-series | No complex analytics built-in; manual aggregation for funnel/heatmap |
| **In-memory Python dicts** | Zero external dependency; instant reads; fully testable in isolation | Lost on restart; single-process only |

### What AI Suggested

The LLM recommended TimescaleDB because it would natively answer the 7-day rolling average query in `/anomalies` with a single SQL window function. It specifically noted that in-memory storage would lose all data on container restart. It also suggested Redis caching in front of whatever database I chose.

### What I Chose and Why

**In-memory Python dicts.** Here's the brutal reasoning:

1. **The acceptance gate is `docker compose up` with no manual steps.** TimescaleDB adds a fourth container, initialisation scripts, and port binding. On developer laptops with port conflicts or resource limits, this fails unpredictably.

2. **The data volume is not a database problem.** 5 stores × 3 cameras × 20-minute clips = ~200,000 events maximum. A `VisitorSession` object with full event history is ~2-3KB. 10,000 sessions is 30MB. Python dict lookups at this scale are faster than any database query.

3. **The 7-day rolling average claim is architecturally dishonest.** I have 20 minutes of data. I am not going to fabricate a "7-day average" from that. I modelled `historical_avg_rate` as an injectable parameter with a comment explaining that a production deployment would populate it from cold storage. This is honest engineering.

4. **Test isolation is genuinely better in-memory.** Each test case calls `.clear()` on the state objects. With TimescaleDB this requires truncating tables between tests or spinning up a test container. My test suite runs in <3 seconds with zero external dependencies.

**I agreed with the LLM on restart durability** and addressed it directly: `EventEmitter` appends every event to `events.jsonl` on disk. `ReplayEngine` re-ingests that file via `POST /stores/{id}/replay`. The API recovers its full state after a restart in seconds.

I rejected the Redis caching suggestion entirely. At 5 stores with sub-second projection computation, caching reduces latency from ~5ms to ~1ms. This is immeasurable to a human and adds operational complexity for zero user-visible benefit.

---

## Choice 5 — Onboarding: Dynamic Wizard vs. Static Configuration Files

### The Problem

The original design required setting a `STORE_ID` environment variable before starting the pipeline. Every camera had to be named correctly in a `zones_override.json` file. Running a second store required editing config files and restarting containers. This is the kind of setup friction that kills real-world adoption.

### Options I Evaluated

**Option A — Environment variables + JSON config files:** Simple to implement, familiar to engineers. But requires reading documentation before doing anything. Breaks the "git clone → run" promise.

**Option B — Interactive CLI wizard:** Prompts for store name, camera roles, etc. Still terminal-based, not demo-friendly. Hard to show calibration results.

**Option C — Browser-based onboarding wizard:** A single-page application served by the pipeline container. No config files. No environment variables. Upload clips in the browser, draw zone polygons on video frames, click Start.

### What AI Suggested

The LLM initially suggested Option A as the "least complex" solution. When I pushed back, it suggested a CLI wizard (Option B). It didn't propose a browser wizard — that came from asking "how would a non-engineer set this up?"

### What I Chose and Why

**Option C — browser wizard.** The decision was driven by three observations:

1. **The demo experience matters.** A reviewer who runs `./run.sh` and gets a browser UI that guides them step-by-step through setup is having a fundamentally different experience than one who has to read a YAML file and set environment variables. Live dashboards score higher in the rubric. Getting to the dashboard should be frictionless.

2. **The calibration problem requires visual tooling.** You cannot meaningfully define zone polygons in a JSON file without seeing the actual camera frame. The browser wizard loads the first frame of each uploaded clip as the background of a canvas where you draw the polygons. This is the only approach that produces calibrations that actually match the real camera geometry.

3. **Dynamic camera ID assignment eliminates a whole class of errors.** Rather than requiring the user to know that Camera 3 must be named `CAM_ENTRY_03`, the wizard assigns IDs automatically: the first camera with role `entry` becomes `CAM_ENTRY_01`, the second `CAM_ENTRY_02`. The calibration studio and pipeline both read from the same session state, so everything stays in sync without any user intervention.

The implementation consequence: `detect.py` needed to handle the case where `STORE_ID` and `CAMERA_MAP` are both empty (Wizard Mode) by starting only the GUI server and waiting for `/api/analysis/start`. This is a clean separation: the pipeline is either in wizard-idle mode or in analysis mode, never in an ambiguous half-configured state.

---

## Recurring Theme: Accepting Technical Debt Honestly

Every choice above involves an honest trade-off. In-memory storage loses state on restart — mitigated by replay. Custom Re-ID is less accurate than deep embeddings — calibrated to fail gracefully rather than silently. The 7-day rolling average is stubbed — documented clearly. Zone polygons use normalised coordinates that assume the camera doesn't move — stated in the code.

The alternative — pretending the debt doesn't exist — produces systems that look perfect in demos and fail in production. Every stubbed value, every fallback, every hardcoded constant in this codebase has a comment explaining exactly what it is and what a production deployment would replace it with. That's the standard I held throughout.
