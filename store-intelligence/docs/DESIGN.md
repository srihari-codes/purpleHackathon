# DESIGN.md — Store Intelligence Platform Architecture

## Overview

This system converts raw CCTV footage from Apex Retail stores into a live analytics API. The pipeline has four stages: **Detection → Event Stream → Intelligence API → Live Dashboard**.

```
Raw CCTV Clips
     │
     ▼
[Detection Layer]
  YOLOv8n + ByteTrack → person bounding boxes per frame
  OSNet Re-ID         → cross-camera visitor identity
  Zone Classifier     → point-in-polygon zone assignment
  Staff Detector      → HSV uniform heuristic + zone pattern
     │
     ▼ structured JSON events (JSONL + HTTP POST)
[Intelligence API]  FastAPI + SQLite
  POST /events/ingest    → dedup, validate, store
  GET  /stores/{id}/metrics    → live visitor + conversion metrics
  GET  /stores/{id}/funnel     → Entry→Zone→Billing→Purchase
  GET  /stores/{id}/heatmap    → zone dwell normalised 0-100
  GET  /stores/{id}/anomalies  → queue spike, conversion drop, dead zone
  GET  /health                 → per-store feed lag + STALE_FEED
     │
     ▼
[Live Dashboard]  Rich terminal, polling every 2s
```

---

## Detection Layer (`pipeline/`)

### Frame Sampling
The CCTV clips are 15fps. We process every 3rd frame (effective 5fps). At 5fps a 20-minute clip = 6,000 frames, which is manageable on CPU in roughly 30-40 minutes. GPU (CUDA) is used automatically if available via PyTorch.

### Person Detection
**YOLOv8n** (nano variant) — the fastest member of the YOLOv8 family. We use `classes=[0]` to detect only persons and a confidence threshold of 0.35. Low-confidence detections are retained and flagged (not silently dropped), per spec.

### Tracking
**ByteTrack** (built into ultralytics `persist=True`) assigns per-clip track IDs using a combination of high-confidence and low-confidence detections. This handles the partial occlusion edge case better than DeepSORT by retaining low-confidence tracks that DeepSORT would discard.

### Re-Identification
**OSNet (x0.25)** from `torchreid` generates a 512-dim appearance embedding per person crop. We maintain a gallery: new crops are matched against gallery embeddings by cosine similarity. Threshold 0.75 (tuned empirically). If torchreid is unavailable at runtime, we fall back to bounding box trajectory similarity.

### Cross-Camera Deduplication
The floor camera overlaps with the entry camera field of view. OSNet embedding matching handles this: the same person in both cameras gets the same `visitor_id` because their appearance embeddings match above the threshold.

### Staff Detection
Two complementary signals:
1. **HSV uniform colour heuristic**: retail staff typically wear white/light-grey uniforms. We compute the fraction of the person crop matching the uniform HSV range. ≥40% coverage = likely staff.
2. **Zone pattern heuristic**: a visitor who appears in 3+ distinct zones with ≥5 zone events is likely staff (customers rarely traverse all zones).

Both are configurable via constants in `tracker.py`.

### Direction (Entry vs Exit)
For the entry camera, we track the vertical trajectory of the bounding box centroid over 5+ frames. Positive y-delta (moving downward/inward) = ENTRY. Negative y-delta = EXIT. This handles the case where the entry camera captures both directions.

### Re-entry Detection
A visitor who was marked as EXIT and reappears within 20 minutes gets a REENTRY event (not a new ENTRY), preventing inflation of unique visitor counts.

---

## Intelligence API (`app/`)

### Storage: SQLite
SQLite was chosen for simplicity and zero-ops setup. The DB file is persisted via a Docker volume. SQLAlchemy ORM is used for portability — migrating to PostgreSQL requires only changing `DATABASE_URL`.

### Idempotency
`POST /events/ingest` checks for existing `event_id` before inserting. Duplicate `event_id` → no-op, counted as `duplicate` in response. This makes the pipeline safe to retry on failure.

### Conversion Rate Computation
There is no `customer_id` in POS data. We correlate by time window: a visitor who was in `BILLING_COUNTER` or `BILLING_QUEUE` in the 5-minute window before a POS transaction timestamp is counted as converted. This is the standard approach for POS-less customer attribution in retail analytics.

### Session Deduplication in Funnel
The funnel operates on `visitor_id` sets, not raw event counts. Re-entry events use the same `visitor_id` as the original entry, ensuring a customer who re-enters is counted once in the funnel.

### Structured Logging
Every request logs a JSON-structured line containing: `trace_id`, `store_id`, `endpoint`, `status_code`, `latency_ms`. The `trace_id` is a UUID prefix in the response header `X-Trace-Id`. This is what an on-call engineer needs to trace a specific request.

---

## AI-Assisted Decisions

### 1. ByteTrack vs DeepSORT for tracking
I asked Claude Sonnet to compare ByteTrack and DeepSORT for retail CCTV. It correctly identified that ByteTrack handles crowded scenes better by keeping a separate track pool for low-confidence detections. I agreed and used ByteTrack. The AI also suggested SORT as a simpler fallback — I disagreed, as SORT loses tracks on occlusion which is common in the billing area footage.

### 2. Conversion attribution window
I asked for advice on the 5-minute billing-zone → POS correlation window. The AI suggested 3 minutes as "tighter and more accurate." I chose 5 minutes after reasoning that a customer could be processed slowly at a manual billing counter. The problem statement also implies 5 minutes implicitly in its schema example. This is documented in `CHOICES.md`.

### 3. Staff detection approach
The AI suggested using a fine-tuned YOLO classification head for staff vs customer, which would be more accurate but requires labelled training data I don't have. I instead used the dual heuristic approach (colour + zone pattern). The AI agreed this was the right call for zero-shot deployment. I note in `CHOICES.md` that the VLM approach (GPT-4V for staff classification) was considered.
