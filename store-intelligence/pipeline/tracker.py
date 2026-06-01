"""
tracker.py — Visitor identity manager.

Two-layer identity system:
  Layer 1: track_id  — ephemeral, from ByteTrack
  Layer 2: visitor_id — stable, persistent across occlusions and cameras

VisitorPassport:
  Every visitor gets a persistent passport that survives track-ID churn.
  Fields match the spec exactly.

Identity lifecycle (spec-compliant):
  ACTIVE     — track is being detected this frame
  SUSPENDED  — track was lost; retained for SUSPENDED_RETAIN_SEC for re-association
  EXPIRED    — timeout elapsed; passport archived (never re-used)

ReID score formula (spec-compliant):
  reid_score = appearance_similarity
             + time_similarity
             + spatial_similarity
             + camera_handoff_bonus

Camera handoff prediction:
  When a person disappears from Camera N and later appears in an adjacent Camera M,
  the handoff_bonus is applied when timing + adjacency both match.
"""

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, List, Tuple, Set
import numpy as np

from config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera adjacency map (store geometry)
# ---------------------------------------------------------------------------
CAMERA_ADJACENCY: Dict[str, Set[str]] = {
    "CAM_FLOOR_01":   {"CAM_FLOOR_02", "CAM_ENTRY_03"},
    "CAM_FLOOR_02":   {"CAM_FLOOR_01", "CAM_BILLING_05"},
    "CAM_ENTRY_03":   {"CAM_FLOOR_01"},
    "CAM_GODOWN_04":  {"CAM_FLOOR_02"},
    "CAM_BILLING_05": {"CAM_FLOOR_02"},
}

# Expected transit time (seconds) between adjacent camera pairs.
# Used to compute the camera_handoff_bonus.
HANDOFF_TRANSIT_SEC: Dict[Tuple[str, str], float] = {
    ("CAM_ENTRY_03",   "CAM_FLOOR_01"): 3.0,
    ("CAM_FLOOR_01",   "CAM_ENTRY_03"): 3.0,
    ("CAM_FLOOR_01",   "CAM_FLOOR_02"): 4.0,
    ("CAM_FLOOR_02",   "CAM_FLOOR_01"): 4.0,
    ("CAM_FLOOR_02",   "CAM_BILLING_05"): 3.0,
    ("CAM_BILLING_05", "CAM_FLOOR_02"): 3.0,
    ("CAM_FLOOR_02",   "CAM_GODOWN_04"): 5.0,
    ("CAM_GODOWN_04",  "CAM_FLOOR_02"): 5.0,
}
# Max deviation from expected transit time to award the bonus (seconds)
HANDOFF_TIMING_TOLERANCE_SEC: float = 4.0


# ---------------------------------------------------------------------------
# Identity lifecycle states (spec-mandated)
# ---------------------------------------------------------------------------
class IdentityState(Enum):
    ACTIVE    = auto()   # being tracked this frame
    SUSPENDED = auto()   # track lost; within re-association window
    EXPIRED   = auto()   # timed out; archived


# ---------------------------------------------------------------------------
# VisitorPassport — spec-mandated persistent identity record
# ---------------------------------------------------------------------------
@dataclass
class VisitorPassport:
    """
    Persistent identity record for one unique visitor.
    Survives track-ID churn, camera switches, and brief occlusions.
    """
    visitor_id:         str
    first_seen:         float                           # wall-clock epoch
    last_seen:          float
    last_camera:        str
    last_zone:          Optional[str]
    appearance_embedding: Optional[np.ndarray]
    is_staff:           bool                = False
    staff_confidence:   float               = 0.0
    zones_visited:      Set[str]            = field(default_factory=set)
    cumulative_dwell_ms: int                = 0        # total dwell across all zones
    reentry_count:      int                 = 0        # times visitor re-entered after EXIT
    state:              "IdentityState"     = IdentityState.ACTIVE

    # Internal tracking
    track_id:           Optional[int]       = None
    camera_id:          Optional[str]       = None     # current camera
    bbox_xyxy:          Optional[Tuple]     = None
    session_start:      float               = field(default_factory=time.time)
    session_seq:        int                 = 0

    # Entry/exit bookkeeping
    has_entered:        bool    = False
    has_exited:         bool    = False
    exit_count:         int     = 0

    # Zone dwell tracking
    zone_id:            Optional[str]   = None
    zone_enter_time:    Optional[float] = None

    # Confidence components (spec: det × track × reid × zone)
    last_det_conf:      float   = 1.0
    last_track_conf:    float   = cfg.DEFAULT_TRACKING_CONF
    last_reid_conf:     float   = cfg.DEFAULT_REID_CONF
    last_zone_conf:     float   = cfg.DEFAULT_ZONE_CONF

    @property
    def final_confidence(self) -> float:
        return round(
            self.last_det_conf
            * self.last_track_conf
            * self.last_reid_conf
            * self.last_zone_conf,
            4,
        )

    @property
    def session_duration_ms(self) -> int:
        return int((self.last_seen - self.first_seen) * 1000)


# ---------------------------------------------------------------------------
# Appearance embedding extractor
# ---------------------------------------------------------------------------
class EmbeddingExtractor:
    """
    Tries OSNet (torchreid) when USE_OSNET=1.
    Falls back to a fast HSV+spatial histogram when not available.
    """

    def __init__(self):
        self._model     = None
        self._use_torch = False
        if cfg.USE_OSNET:
            self._try_load_torch()
        else:
            logger.info("EmbeddingExtractor: OSNet disabled (USE_OSNET=0). "
                        "Using HSV+colour histogram fallback.")

    def _try_load_torch(self):
        try:
            import torch
            import torchreid
            self._model = torchreid.models.build_model(
                name="osnet_x0_25",
                num_classes=1,
                pretrained=True,
            )
            self._model.eval()
            self._use_torch = True
            logger.info("EmbeddingExtractor: OSNet (torchreid) loaded")
        except Exception as e:
            logger.info(f"EmbeddingExtractor: torchreid unavailable ({e}), "
                        f"using HSV+colour fallback")
            self._use_torch = False

    def extract(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        if self._use_torch:
            return self._extract_osnet(crop_bgr)
        return self._extract_fallback(crop_bgr)

    def _extract_osnet(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            import torch
            import cv2
            img = cv2.resize(crop_bgr, (128, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img  = (img - mean) / std
            tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
            with torch.no_grad():
                feat = self._model(tensor)
            emb  = feat.squeeze().numpy()
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception as e:
            logger.debug(f"OSNet extraction failed: {e}")
            return self._extract_fallback(crop_bgr)

    def _extract_fallback(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        HSV colour histogram (3-channel, 2 body halves) + coarse HOG proxy.
        Produces a 192-dim L2-normalised vector.
        """
        try:
            import cv2
            resized = cv2.resize(crop_bgr, (64, 128))
            hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            # Upper half: Hue+Sat; Lower half: Value
            h_hist  = np.histogram(hsv[:64, :, 0], bins=32, range=(0, 180))[0]
            s_hist  = np.histogram(hsv[:64, :, 1], bins=32, range=(0, 256))[0]
            v_hist  = np.histogram(hsv[64:, :, 2], bins=32, range=(0, 256))[0]
            # Spatial layout: left-half vs right-half brightness
            left_v  = np.histogram(hsv[:, :32, 2], bins=32, range=(0, 256))[0]
            right_v = np.histogram(hsv[:, 32:, 2], bins=32, range=(0, 256))[0]
            emb  = np.concatenate([h_hist, s_hist, v_hist, left_v, right_v]).astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception:
            return np.zeros(160, dtype=np.float32)


def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    dot = float(np.dot(a, b))
    na  = float(np.linalg.norm(a))
    nb  = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Visitor Identity Manager
# ---------------------------------------------------------------------------
class VisitorIdentityManager:
    """
    Maintains stable visitor_ids (VisitorPassports) across ephemeral track_ids.

    Identity lifecycle:
      ACTIVE    — (track_id, camera_id) present in self._passports
      SUSPENDED — track lost; passport in self._suspended for re-association
      EXPIRED   — timed out; moved to self._expired (read-only archive)

    Key operations:
      resolve(track_id, camera_id, bbox, embedding, wall_time, det_conf, track_conf)
        → (visitor_id, is_new, reid_conf)

      mark_lost(track_id, camera_id)
        → moves to SUSPENDED

      purge_stale_suspended(wall_time)
        → moves SUSPENDED → EXPIRED after SUSPENDED_RETAIN_SEC
    """

    def __init__(self):
        self._extractor = EmbeddingExtractor()

        # (track_id, camera_id) → VisitorPassport   [ACTIVE]
        self._active: Dict[Tuple[int, str], VisitorPassport] = {}

        # visitor_id → VisitorPassport               [SUSPENDED]
        self._suspended: Dict[str, VisitorPassport] = {}

        # visitor_id → VisitorPassport               [EXPIRED — archive]
        self._expired: Dict[str, VisitorPassport] = {}

        # visitor_id → session_seq counter
        self._session_seq: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_embedding(self, frame_bgr: np.ndarray,
                          bbox_xyxy: Tuple) -> Optional[np.ndarray]:
        if frame_bgr is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        return self._extractor.extract(crop)

    def resolve(
        self,
        track_id:   int,
        camera_id:  str,
        bbox_xyxy:  Tuple,
        embedding:  Optional[np.ndarray],
        wall_time:  float,
        det_conf:   float = 1.0,
        track_conf: float = cfg.DEFAULT_TRACKING_CONF,
    ) -> Tuple[str, bool, float]:
        """
        Returns (visitor_id, is_new_visitor, reid_confidence).
        reid_confidence = 1.0 for active tracks (definite identity);
                        = composite score for re-associations.
        """
        key = (track_id, camera_id)

        # 1. Already tracking this track → update and return
        if key in self._active:
            passport = self._active[key]
            passport.last_seen       = wall_time
            passport.last_camera     = camera_id
            passport.bbox_xyxy       = bbox_xyxy
            passport.last_det_conf   = det_conf
            passport.last_track_conf = track_conf
            passport.last_reid_conf  = 1.0  # definite — same active track
            if embedding is not None:
                passport.appearance_embedding = embedding
            return passport.visitor_id, False, 1.0

        # 2. New track_id — try to match against SUSPENDED passports
        matched_vid, reid_score = self._match_suspended(
            camera_id, bbox_xyxy, embedding, wall_time
        )

        if matched_vid is not None:
            passport = self._suspended.pop(matched_vid)
            passport.track_id   = track_id
            passport.camera_id  = camera_id
            passport.bbox_xyxy  = bbox_xyxy
            passport.last_seen  = wall_time
            passport.last_camera = camera_id
            passport.state      = IdentityState.ACTIVE
            passport.last_det_conf   = det_conf
            passport.last_track_conf = track_conf
            passport.last_reid_conf  = reid_score
            if embedding is not None:
                passport.appearance_embedding = embedding
            self._active[key] = passport
            logger.debug(
                f"Re-associated track {track_id}@{camera_id} → {matched_vid} "
                f"(reid_score={reid_score:.3f})"
            )
            return matched_vid, False, reid_score

        # 2.5. Try active tracks from overlapping (adjacent) cameras
        if embedding is not None:
            best_sim  = 0.0
            best_vid2 = None
            for existing_key, ep in self._active.items():
                active_cam = existing_key[1]
                if active_cam == camera_id:
                    continue
                if (camera_id in CAMERA_ADJACENCY.get(active_cam, set()) or
                        active_cam in CAMERA_ADJACENCY.get(camera_id, set())):
                    sim = cosine_similarity(ep.appearance_embedding, embedding)
                    if sim > best_sim and sim >= cfg.REID_THRESHOLD + cfg.REID_OVERLAP_BONUS:
                        best_sim  = sim
                        best_vid2 = ep.visitor_id

            if best_vid2 is not None:
                # Share the visitor_id — create a new active entry for this camera
                source_passport = next(
                    p for p in self._active.values() if p.visitor_id == best_vid2
                )
                new_passport = VisitorPassport(
                    visitor_id          = best_vid2,
                    first_seen          = source_passport.first_seen,
                    last_seen           = wall_time,
                    last_camera         = camera_id,
                    last_zone           = None,
                    appearance_embedding= embedding,
                    is_staff            = source_passport.is_staff,
                    staff_confidence    = source_passport.staff_confidence,
                    zones_visited       = source_passport.zones_visited,
                    cumulative_dwell_ms = source_passport.cumulative_dwell_ms,
                    reentry_count       = source_passport.reentry_count,
                    state               = IdentityState.ACTIVE,
                    track_id            = track_id,
                    camera_id           = camera_id,
                    bbox_xyxy           = bbox_xyxy,
                    session_start       = source_passport.session_start,
                    has_entered         = source_passport.has_entered,
                    has_exited          = source_passport.has_exited,
                    exit_count          = source_passport.exit_count,
                    last_det_conf       = det_conf,
                    last_track_conf     = track_conf,
                    last_reid_conf      = best_sim,
                )
                self._active[key] = new_passport
                logger.debug(
                    f"Cross-camera match track {track_id}@{camera_id} → {best_vid2} "
                    f"(sim={best_sim:.3f})"
                )
                return best_vid2, False, best_sim

        # 3. Genuinely new visitor
        visitor_id = "VIS_" + uuid.uuid4().hex[:6].upper()
        passport = VisitorPassport(
            visitor_id           = visitor_id,
            first_seen           = wall_time,
            last_seen            = wall_time,
            last_camera          = camera_id,
            last_zone            = None,
            appearance_embedding = embedding,
            state                = IdentityState.ACTIVE,
            track_id             = track_id,
            camera_id            = camera_id,
            bbox_xyxy            = bbox_xyxy,
            session_start        = wall_time,
            last_det_conf        = det_conf,
            last_track_conf      = track_conf,
            last_reid_conf       = 1.0,
        )
        self._active[key] = passport
        logger.debug(f"New visitor {visitor_id} track {track_id}@{camera_id}")
        return visitor_id, True, 1.0

    def mark_lost(self, track_id: int, camera_id: str):
        """Move ACTIVE track → SUSPENDED."""
        key = (track_id, camera_id)
        if key in self._active:
            passport = self._active.pop(key)
            passport.state    = IdentityState.SUSPENDED
            passport.is_active = False  # backward compat
            self._suspended[passport.visitor_id] = passport
            logger.debug(
                f"Track {track_id}@{camera_id} → SUSPENDED ({passport.visitor_id})"
            )

    def mark_exited(self, visitor_id: str):
        """Record that this visitor has physically left the store."""
        passport = self.get_passport(visitor_id)
        if passport:
            passport.exit_count      += 1
            passport.reentry_count    = passport.exit_count - 1 if passport.exit_count > 1 else 0
            passport.has_exited       = True

    def get_passport(self, visitor_id: str) -> Optional[VisitorPassport]:
        """Return passport regardless of state."""
        for p in self._active.values():
            if p.visitor_id == visitor_id:
                return p
        if visitor_id in self._suspended:
            return self._suspended[visitor_id]
        return self._expired.get(visitor_id)

    def get_active_passport_by_track(
        self, track_id: int, camera_id: str
    ) -> Optional[VisitorPassport]:
        return self._active.get((track_id, camera_id))

    # Kept for backwards compatibility with detect.py until that is updated
    def get_active_state_by_track(
        self, track_id: int, camera_id: str
    ) -> Optional[VisitorPassport]:
        return self.get_active_passport_by_track(track_id, camera_id)

    def exit_count(self, visitor_id: str) -> int:
        p = self.get_passport(visitor_id)
        return p.exit_count if p else 0

    def increment_session_seq(self, visitor_id: str) -> int:
        self._session_seq[visitor_id] += 1
        seq = self._session_seq[visitor_id]
        p = self.get_passport(visitor_id)
        if p:
            p.session_seq = seq
        return seq

    def get_session_seq(self, visitor_id: str) -> int:
        return self._session_seq.get(visitor_id, 0)

    def purge_stale_suspended(self, wall_time: float):
        """
        SUSPENDED → EXPIRED after SUSPENDED_RETAIN_SEC.
        Keeps EXPIRED archive for audit; does not delete passports.
        """
        to_expire = [
            vid for vid, p in self._suspended.items()
            if (wall_time - p.last_seen) > cfg.SUSPENDED_RETAIN_SEC
        ]
        for vid in to_expire:
            passport = self._suspended.pop(vid)
            passport.state = IdentityState.EXPIRED
            self._expired[vid] = passport
            logger.debug(f"Passport {vid} → EXPIRED")

    def purge_stale_lost(self, wall_time: float):
        """Alias kept for backwards compatibility with detect.py."""
        self.purge_stale_suspended(wall_time)

    @property
    def active_count(self) -> int:
        return len({p.visitor_id for p in self._active.values()})

    @property
    def suspended_count(self) -> int:
        return len(self._suspended)

    @property
    def expired_count(self) -> int:
        return len(self._expired)

    # Backward compat property used in detect.py
    @property
    def lost_count(self) -> int:
        return self.suspended_count

    # Backward compat — detect.py accesses _active directly
    # Keep attribute names consistent
    @property
    def _lost(self):
        return self._suspended

    # ------------------------------------------------------------------
    # Internal: ReID matching
    # ------------------------------------------------------------------

    def _match_suspended(
        self,
        camera_id:  str,
        bbox_xyxy:  Tuple,
        embedding:  Optional[np.ndarray],
        wall_time:  float,
    ) -> Tuple[Optional[str], float]:
        """
        Find best matching SUSPENDED passport.
        Returns (visitor_id, composite_reid_score) or (None, 0.0).

        reid_score = w_app × appearance_sim
                   + w_spatial × spatial_score
                   + w_temporal × temporal_score
                   + w_handoff × handoff_bonus   ← spec addition
        """
        best_vid   = None
        best_score = 0.0

        for vid, passport in self._suspended.items():
            age = wall_time - passport.last_seen

            # Hard gate: must be within SUSPENDED window
            if age > cfg.SUSPENDED_RETAIN_SEC:
                continue

            # Camera plausibility
            same_cam = (passport.last_camera == camera_id)
            adjacent = camera_id in CAMERA_ADJACENCY.get(passport.last_camera, set())
            if not same_cam and not adjacent:
                continue

            # Cross-camera hard time gate
            if not same_cam and age > cfg.CROSS_CAM_TIME_WINDOW_SEC:
                continue

            # ── Score components ──────────────────────────────────────────

            # 1. Appearance similarity
            app_sim = cosine_similarity(passport.appearance_embedding, embedding)

            # 2. Spatial proximity (same-camera only)
            if same_cam and passport.bbox_xyxy is not None:
                cx_new = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
                cy_new = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
                cx_old = (passport.bbox_xyxy[0] + passport.bbox_xyxy[2]) / 2
                cy_old = (passport.bbox_xyxy[1] + passport.bbox_xyxy[3]) / 2
                spatial_dist  = (abs(cx_new - cx_old) / 1920 +
                                 abs(cy_new - cy_old) / 1080) / 2
                spatial_score = max(0.0, 1.0 - spatial_dist / cfg.SPATIAL_PIXEL_THRESHOLD)
            else:
                spatial_score = 0.5  # cross-camera: rely on appearance

            # 3. Temporal score (more recent = better)
            temporal_score = max(0.0, 1.0 - age / cfg.SUSPENDED_RETAIN_SEC)

            # 4. Camera handoff bonus (spec: boost when adjacent camera + timing matches)
            handoff_bonus = 0.0
            if not same_cam and adjacent:
                pair_key = (passport.last_camera, camera_id)
                expected_transit = HANDOFF_TRANSIT_SEC.get(pair_key)
                if expected_transit is not None:
                    timing_error = abs(age - expected_transit)
                    if timing_error <= HANDOFF_TIMING_TOLERANCE_SEC:
                        # Smooth bonus: 1.0 at perfect timing, 0.0 at tolerance edge
                        handoff_bonus = max(0.0, 1.0 - timing_error / HANDOFF_TIMING_TOLERANCE_SEC)
                        logger.debug(
                            f"Handoff bonus for {vid}: "
                            f"{passport.last_camera}→{camera_id} "
                            f"age={age:.1f}s expected={expected_transit}s "
                            f"bonus={handoff_bonus:.2f}"
                        )

            # ── Composite score (spec formula) ─────────────────────────────
            score = (
                cfg.REID_W_APPEARANCE * app_sim
                + cfg.REID_W_SPATIAL   * spatial_score
                + cfg.REID_W_TEMPORAL  * temporal_score
                + cfg.REID_W_HANDOFF   * handoff_bonus
            )

            if score > best_score and score >= cfg.REID_THRESHOLD:
                best_score = score
                best_vid   = vid

        return best_vid, best_score
