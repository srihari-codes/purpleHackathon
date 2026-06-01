"""
config.py — Central configuration for the Detection Layer.

All tunable constants live here. Override via environment variables where noted.
Import pattern:
    from config import cfg
"""

import os


class _Config:
    # ── YOLO / Detection ──────────────────────────────────────────────────────
    # YOLO11m by default (spec requirement). Use yolo11s.pt for faster CPU inference.
    # Override with env:  YOLO_MODEL=/path/to/yolo11s.pt
    YOLO_MODEL: str        = os.environ.get("YOLO_MODEL", "yolo11m.pt")
    YOLO_CONF: float       = 0.35          # minimum detection confidence
    YOLO_IOU: float        = 0.45          # NMS IoU threshold
    YOLO_CLASSES: list     = [0]           # class 0 = person only
    YOLO_TRACKER: str      = "bytetrack.yaml"

    # ── Re-ID / Identity ──────────────────────────────────────────────────────
    # OSNet (torchreid) is opt-in: set USE_OSNET=1 to enable.
    # When disabled, falls back to HSV+HOG colour histogram.
    USE_OSNET: bool        = os.environ.get("USE_OSNET", "0") == "1"

    # Identity lifecycle timings (seconds)
    ACTIVE_TO_SUSPENDED_SEC: float  = 0.0    # immediate on track loss
    SUSPENDED_RETAIN_SEC: float     = 15.0   # spec: 15 s in SUSPENDED
    EXPIRED_PURGE_SEC: float        = 30.0   # purge after 30 s total absence

    # ReID matching thresholds
    REID_THRESHOLD: float           = 0.65   # min composite score to re-associate
    REID_OVERLAP_BONUS: float       = 0.05   # extra threshold for active-overlap match

    # Score component weights for composite ReID
    REID_W_APPEARANCE: float        = 0.55
    REID_W_SPATIAL: float           = 0.20
    REID_W_TEMPORAL: float          = 0.15
    REID_W_HANDOFF: float           = 0.10   # camera-handoff bonus weight

    # Spatial proximity gate (fraction of frame dimension for same-camera)
    SPATIAL_PIXEL_THRESHOLD: float  = 0.25

    # Cross-camera constraints
    CROSS_CAM_TIME_WINDOW_SEC: float = 5.0   # max gap for cross-camera re-ID

    # Embedding extraction: only re-extract every N frames (performance)
    EMBEDDING_REFRESH_FRAMES: int   = 15

    # ── Staff Detection ───────────────────────────────────────────────────────
    BLACK_SAT_MAX: int              = 60     # HSV saturation ≤ this → "black"
    BLACK_VAL_MAX: int              = 60     # HSV value ≤ this → "black"
    BLACK_ZONE_THRESHOLD: float     = 0.45   # fraction of zone pixels that must be black
    STAFF_SCORE_THRESHOLD: float    = 0.55   # composite score to classify as staff
    STAFF_MIN_FRAMES: int           = 15     # need this many frames before deciding

    # Staff composite weights
    STAFF_W_BLACK: float            = 0.80
    STAFF_W_PRESENCE: float         = 0.10
    STAFF_W_ZONE_DIV: float         = 0.05
    STAFF_W_CAM_DIV: float          = 0.05

    # ── Zone Engine ───────────────────────────────────────────────────────────
    ZONE_DWELL_INTERVAL_MS: int     = 30_000  # emit ZONE_DWELL every 30 s

    # ── Queue Engine ─────────────────────────────────────────────────────────
    MIN_QUEUE_DWELL_SEC: float      = 5.0    # ignore walk-throughs < 5 s
    POS_ABANDON_WINDOW_SEC: float   = float(
        os.environ.get("POS_ABANDON_WINDOW_SEC", "45")
    )

    # ── Entry/Exit ────────────────────────────────────────────────────────────
    ENTRY_MIN_CROSS_FRAMES: int     = 3      # debounce: frames on new side before commit

    # ── Behavior State Machine ────────────────────────────────────────────────
    BROWSING_MIN_DWELL_MS: int      = 10_000   # 10 s before BROWSING → DWELLING
    DWELLING_MIN_DWELL_MS: int      = 60_000   # 60 s before DWELLING stays
    QUEUEING_ZONE_IDS: set          = frozenset({
        "ZONE_BILLING_QUEUE", "ZONE_CASH_COUNTER"
    })

    # ── Confidence Pipeline ───────────────────────────────────────────────────
    # final_confidence = det × track × reid × zone (all 0–1)
    DEFAULT_TRACKING_CONF: float    = 0.80   # fallback when ByteTrack score unavailable
    DEFAULT_REID_CONF: float        = 1.00   # 1.0 = definitely the same identity (active track)
    DEFAULT_ZONE_CONF: float        = 1.00   # 1.0 = solidly inside polygon

    # ── Video / Pipeline ──────────────────────────────────────────────────────
    VIDEO_FPS: float                = 15.0
    FRAME_RESIZE_W: int             = 960    # downscale width for performance (CPU)
    FRAME_RESIZE_H: int             = 540
    GUI_ANNOTATE_EVERY_N: int       = 5      # only annotate every Nth frame for GUI
    LOST_FLUSH_GRACE_FRAMES: int    = 5      # frames before marking a track lost

    # ── Timestamps ────────────────────────────────────────────────────────────
    # Fallback datetime when OCR fails on ALL cameras
    FALLBACK_START_ISO: str         = "2026-04-10T20:10:00Z"


cfg = _Config()
