# CHOICES.md — Three Key Design Decisions

## Decision 1: Detection Model — YOLOv8n + ByteTrack

### Options Considered
| Option | Pros | Cons |
|--------|------|------|
| YOLOv8n + ByteTrack | Fast, CPU-viable, built-in tracker, strong community | Less accurate than larger variants |
| YOLOv8x + StrongSORT | Higher accuracy, better Re-ID | Too slow for 5-camera real-time |
| RT-DETR + ByteTrack | SOTA accuracy, transformer-based | Requires GPU for real-time |
| MediaPipe Pose | Lightweight, runs on CPU | Not designed for multi-person tracking |

### What AI Suggested
Claude suggested RT-DETR because of its superior accuracy on crowded scenes (relevant for group entry edge case). It recommended a GPU-first approach. When I pushed back on the requirement to work on CPU for offline batch processing, it agreed YOLOv8n was the right call for the latency/accuracy tradeoff.

### What I Chose and Why
**YOLOv8n + ByteTrack.**

The problem says "process 5 stores × 3 cameras = 15 clips offline." At 15fps × 20 min = 18,000 frames per clip, I need something that can process a clip in reasonable time without GPU. YOLOv8n at 5fps effective = 6,000 frames per clip. On CPU, YOLOv8n runs ~8-12fps, so a 20-min clip processes in ~8 min on CPU (acceptable for offline batch). ByteTrack is the right tracker because it handles the billing queue crowding edge case better than IoU-only methods.

The tradeoff: YOLOv8n will have higher miss rates on partial occlusion than YOLOv8l. I mitigate this by keeping low-confidence detections flagged (not dropped) and by ByteTrack's low-confidence track pool.

---

## Decision 2: Event Schema Design

### The Core Question
Should `visitor_id` be stable across the entire day (across re-entries) or reset on each entry?

### Options Considered
**Option A — Stable visitor_id (what I chose):**
Same physical person = same `visitor_id` even across re-entries. Re-entry produces `REENTRY` event type, not a second `ENTRY`.

**Option B — Session-scoped visitor_id:**
Each visit = new `visitor_id`. Re-entry = second `ENTRY` with a different ID. Simpler to implement, but inflates unique visitor count.

### What AI Suggested
AI initially suggested Option B (session-scoped) because it's simpler: "each track in a clip gets a unique ID." When I pointed out the problem spec explicitly says re-entry should NOT inflate visitor counts and should produce a REENTRY event, the AI agreed Option A was correct.

### What I Chose and Why
**Option A — stable visitor_id across the day.**

The problem statement says: *"Re-entry inflation is a known vendor problem you are solving."* This makes the choice unambiguous. A visitor who steps out and returns gets the same `visitor_id` and a `REENTRY` event. The funnel deduplicates on `visitor_id` sets, so conversion rate is accurate.

The tradeoff: OSNet Re-ID can fail if appearance changes significantly (e.g. customer puts on a jacket). In that case, they'd get a new `visitor_id` — acceptable miss rate for a system that's already ahead of "no offline analytics."

The schema also preserves `session_seq` in metadata — the ordinal position of each event in a visitor's journey. This allows future session analysis without needing to query event ordering.

---

## Decision 3: API Architecture — SQLite vs PostgreSQL

### The Core Question
What database should the API use?

### Options Considered
| Option | Pros | Cons |
|--------|------|------|
| SQLite (in-container) | Zero config, no extra container, simple docker compose | Not suitable for multi-process write concurrency |
| PostgreSQL (separate container) | Production-grade, supports concurrent writes, WAL | Requires second container, adds ops complexity |
| In-memory (dict-based) | Fastest, no disk | Lost on restart, not persistent |

### What AI Suggested
AI recommended PostgreSQL immediately: "for a production API, SQLite will have write contention issues." I asked: *"What's the actual write concurrency here?"* The pipeline processes clips offline and sends batches via POST. In production with 40 stores × 3 cameras = 120 concurrent senders — that's a real concern.

### What I Chose and Why
**SQLite for this submission; PostgreSQL for production.**

For the hackathon evaluation, there's one client (the pipeline) sending batches. SQLite with WAL mode handles this fine. The schema is designed to be portable: changing `DATABASE_URL` from `sqlite:///./store_intelligence.db` to `postgresql://user:pass@db:5432/si` in `docker-compose.yml` is the only required change — SQLAlchemy abstracts the rest.

The `DATABASE_URL` is externalised as an environment variable precisely to make this migration trivial. I disagree with the AI's PostgreSQL-first recommendation for a single-developer hackathon submission because it adds a mandatory `docker compose` dependency that increases the failure surface area. However, I document the migration path in README.md.

**What would make me change this decision:** the acceptance gate says "40 live stores sending events in real time." At that scale, SQLite write locks would be a bottleneck. The answer is PostgreSQL + connection pooling (pgbouncer) or a time-series DB like TimescaleDB for the event log.
