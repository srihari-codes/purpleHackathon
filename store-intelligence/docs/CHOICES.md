# Architectural Choices: Retail Intelligence Pipeline

This document details the three primary architectural decisions made during the design and implementation of the Apex Retail Intelligence system, including options considered, LLM suggestions, and final rationale.

---

## 1. Decision 1: Object Detection and Tracking Model Selection

### Options Considered
1. **YOLOv8 (Ultralytics)**: Standard object detector with built-in tracking support (ByteTrack/BoT-SORT).
2. **YOLOv9 / YOLOv10**: Newer iterations of the YOLO family promising higher efficiency.
3. **RT-DETR (Real-Time DEtection TRansformer)**: Transformer-based detector avoiding non-maximum suppression (NMS) bottlenecks.
4. **MOG2 Background Subtraction + OpenCV Contours**: A lightweight, non-deep-learning fallback.

### What AI Suggested
The AI assistant strongly suggested using **YOLOv8-Medium** (`yolov8m.pt`) as the primary detector and **ByteTrack** for tracking. The AI argued that YOLOv8 has the most stable Python API integration, is fully container-friendly, and has extensive pre-trained weights for person tracking, while ByteTrack manages low-confidence detections and brief occlusions well.

### What We Chose and Why
We chose **YOLOv8-Nano** (`yolov8n.pt`) with an active GPU fallback, while preserving a pure OpenCV **MOG2 Background Subtraction** fallback for resource-constrained environments (e.g., containers running on CPUs without CUDA).
- **Why**: Retail environments run on edge hardware where computational budgets are tight. YOLOv8-Nano offers excellent latency performance (~15ms per frame on low-end GPUs) with a minimal accuracy penalty for person detection. ByteTrack was selected because it tracks people based on motion prediction (Kalman Filters) even when their detection confidence drops temporarily (due to shadows or clothing blending with displays), which is critical for maintaining consistent `track_id` sequences through partial occlusions.

---

## 2. Decision 2: Event Schema Design Rationale

### Options Considered
1. **Flat Transition-Based Schema (Selected)**: Every behavior (entry, exit, zone transitions, queue movements) is emitted as a discrete, self-contained event document containing metadata fields.
2. **Session-Nested Document Schema**: The pipeline sends a single aggregated JSON document containing the entire visitor session history upon store exit.
3. **Graph-Node Representation**: Emitting nodes and edges representing visitor trajectories between camera/shelf nodes.

### What AI Suggested
The AI assistant recommended a **Session-Nested Document Schema** emitted only when a visitor exits the store, arguing it reduces network traffic, guarantees that session details are fully consolidated, and simplifies the REST API's write database.

### What We Chose and Why
We rejected the AI suggestion and chose the **Flat Transition-Based Schema** (complying with the challenge's Event Type Catalogue).
- **Why**: 
  - **Real-Time Responsiveness**: Session-nested schemas cannot support real-time metrics. The business requires live queue depth tracking (`BILLING_QUEUE_JOIN`) and real-time anomaly detection (e.g., sudden queue spikes). Waiting until a visitor exits the store to emit their data creates a 15–30 minute telemetry lag.
  - **Fault Tolerance**: If the edge detection system crashes or loses power, all active visitor data is lost under the session-nested model. A flat stream guarantees that every zone visit is persisted immediately upon occurrence.
  - **Replay Reproducibility**: Flat streams are fully compatible with message brokers (like Kafka/RabbitMQ) and can be replayed deterministically via the `ReplayEngine` to re-materialize metrics if pipeline configurations change.

---

## 3. Decision 3: API Architecture & In-Memory Materialized Projections

### Options Considered
1. **Relational Database (SQL) with On-the-Fly JOINs**: Storing raw events in PostgreSQL/SQLite and computing metrics (conversion rate, funnel) via complex SQL queries on every GET request.
2. **Stateful In-Memory Session Store with Materialized Projections (Selected)**: Maintaining live `VisitorSession` state in memory, processing events as state-mutating transitions, and serving read requests from pre-calculated/materialized projections.
3. **Time-Series Database (InfluxDB) + BI Tooling**: Directing the event stream into a time-series DB and using Grafana for visualization.

### What AI Suggested
The AI assistant suggested using **SQLite** as a local database, storing all events in an `events` table, and using standard SQL queries (e.g., self-joins, subqueries, and window functions) to calculate conversion rates and funnel stages on-demand.

### What We Chose and Why
We chose a **FastAPI server backed by an in-memory `SessionStore` and stateful `Sessionizer`**, producing on-demand materialized projections (`MetricsProjection`, `FunnelProjection`, `HeatmapProjection`).
- **Why**:
  - **Performance**: Standard SQL self-joins on million-row event streams are computationally expensive and introduce latency spikes on dashboards. Storing session state in-memory and maintaining pre-aggregated metrics guarantees sub-millisecond response times for GET requests.
  - **Complex Session Logic**: Business metrics like "visitor-centric conversion" (where re-entering visitors must not be double-counted) and "abandonment rate" require complex state machine tracking (e.g., checking if a queue join was followed by a transaction or an exit). Implementing this logic in SQL queries is highly complex, fragile, and difficult to test. A Python-based `Sessionizer` state machine is far more maintainable, testable, and robust.
  - **Idempotency and Verification**: In-memory storage makes it trivial to deduplicate events by `event_id` and run the `VerifierEngine` rules (such as checking for clock skew or duplicate active sessions) before mutating state.
