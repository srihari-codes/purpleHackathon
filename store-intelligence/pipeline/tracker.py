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
from store_graph import StoreGraph, RetailPhysicsEngine, StoreMemoryGraph
from consensus import ConsensusIdentityEngine, ConsensusSignals
from occlusion import OcclusionReasoner, OcclusionClassification, OcclusionType
from health import TrackHealthMonitor
from ghost import GhostLayer
from shadow import ShadowTracker
from visitor_dna import VisitorDNATracker
from group import GroupTracker
from courtroom import IdentityCourtroom
from evidence import EvidenceLedger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera adjacency map (store geometry) — populated at runtime by the wizard.
# Call set_adjacency_map(role_adjacency, camera_file_map) after session load.
# ---------------------------------------------------------------------------
# Format: {camera_id: set_of_adjacent_camera_ids}
CAMERA_ADJACENCY: Dict[str, Set[str]] = {}

# Expected transit time (seconds) between adjacent camera pairs.
# Used to compute the camera_handoff_bonus.
# Format: {(cam_from, cam_to): transit_sec}
HANDOFF_TRANSIT_SEC: Dict[Tuple[str, str], float] = {}

# Max deviation from expected transit time to award the bonus (seconds)
HANDOFF_TIMING_TOLERANCE_SEC: float = 4.0


def set_adjacency_map(
    adjacency: Dict[str, Set[str]],
    transit_sec: Optional[Dict[Tuple[str, str], float]] = None,
) -> None:
    """
    Populate the module-level camera adjacency and transit maps.

    Called from run_pipeline() once the wizard session is loaded.

    adjacency   : {camera_id: {neighbour_camera_id, ...}}
    transit_sec : {(cam_from, cam_to): seconds} — if None, defaults to 4.0 for all pairs
    """
    global CAMERA_ADJACENCY, HANDOFF_TRANSIT_SEC
    CAMERA_ADJACENCY = {k: set(v) for k, v in adjacency.items()}

    if transit_sec:
        HANDOFF_TRANSIT_SEC = dict(transit_sec)
    else:
        # Default: 4 s for every declared adjacent pair (symmetric)
        HANDOFF_TRANSIT_SEC = {}
        for cam, neighbours in CAMERA_ADJACENCY.items():
            for nb in neighbours:
                for pair in [(cam, nb), (nb, cam)]:
                    HANDOFF_TRANSIT_SEC.setdefault(pair, 4.0)

    logger.info(
        "tracker: adjacency loaded — %d cameras, %d pairs",
        len(CAMERA_ADJACENCY),
        len(HANDOFF_TRANSIT_SEC),
    )


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

    # Occlusion classification (set in mark_lost)
    occlusion_type:     Optional[str]   = None
    occlusion_retain_sec: float         = cfg.SUSPENDED_RETAIN_SEC
    # Last consensus decision explanation
    last_reid_explanation: Optional[dict] = None

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

        # ── Hybrid Consensus systems ──────────────────────────────────
        self._consensus      = ConsensusIdentityEngine()
        self._health         = TrackHealthMonitor()
        self._occlusion      = OcclusionReasoner()
        self._store_graph    = RetailPhysicsEngine()
        
        # ── Identity-Centric Evolution Modules ────────────────────────
        self._memory_graph   = StoreMemoryGraph()
        self._ghosts         = GhostLayer()
        self._shadows        = ShadowTracker()
        self._dna            = VisitorDNATracker()
        self._groups         = GroupTracker()
        self._courtroom      = IdentityCourtroom()
        self._evidence       = EvidenceLedger()
        # visitor_id → most recent camera (for cross-camera tracking)
        self._visitor_cameras: Dict[str, str] = {}

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
        current_zone: Optional[str] = None,
        fingerprint = None,
        trajectory_score: float = 0.5,
    ) -> Tuple[str, bool, float]:
        """
        Returns (visitor_id, is_new_visitor, reid_confidence).
        reid_confidence = 1.0 for active tracks (definite identity);
                        = composite score for re-associations.
        """
        key = (track_id, camera_id)
        
        # Calculate centers
        cx, cy = (bbox_xyxy[0]+bbox_xyxy[2])/2, (bbox_xyxy[1]+bbox_xyxy[3])/2

        # 1. Update Shadow Tracker
        # Check if this matches an existing shadow before checking active tracks
        # (Though usually active tracks are checked first. We'll just tick shadows centrally later, 
        # but check for match here if it's not active)

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

        # 2. Shadow Match Pre-filter
        shadow_match = self._shadows.match_shadow(bbox_xyxy, camera_id, 1920, 1080)
        if shadow_match:
            shadow_vid, shadow_conf = shadow_match
            if shadow_vid in self._suspended:
                logger.debug(f"Shadow match successful for {shadow_vid} on {camera_id}")
                # We could force re-association, but let's let consensus do it with high confidence
                pass 

        # 3. Try to match against SUSPENDED passports
        matched_vid, reid_score = self._match_suspended(
            camera_id=camera_id,
            bbox_xyxy=bbox_xyxy,
            embedding=embedding,
            wall_time=wall_time,
            current_zone=current_zone,
            fingerprint=fingerprint,
            trajectory_score=trajectory_score,
            det_conf=det_conf,
            track_conf=track_conf,
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
            self._visitor_cameras[matched_vid] = camera_id
            
            # Clean up ghost and shadow
            self._ghosts.remove_ghost(matched_vid)
            self._shadows.remove_shadow(matched_vid, camera_id)
            
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
        # Track health: new visitor starts at INIT score
        self._visitor_cameras[visitor_id] = camera_id
        logger.debug(f"New visitor {visitor_id} track {track_id}@{camera_id}")
        return visitor_id, True, 1.0

    def mark_lost(
        self,
        track_id: int,
        camera_id: str,
        frame_w: int = 1920,
        frame_h: int = 1080,
        nearby_track_count: int = 0,
        confirmed_exit: bool = False,
    ):
        """Move ACTIVE track → SUSPENDED, classify occlusion type."""
        key = (track_id, camera_id)
        if key in self._active:
            passport = self._active.pop(key)
            passport.state = IdentityState.SUSPENDED

            # Classify WHY this person disappeared
            if passport.bbox_xyxy is not None:
                ocl = self._occlusion.classify(
                    last_bbox_xyxy     = passport.bbox_xyxy,
                    frame_w            = frame_w,
                    frame_h            = frame_h,
                    nearby_track_count = nearby_track_count,
                    last_zone          = passport.zone_id,
                    confirmed_exit     = confirmed_exit,
                    last_zone_is_billing = (passport.zone_id in
                        ("ZONE_BILLING_QUEUE", "ZONE_CASH_COUNTER")),
                )
                passport.occlusion_type      = ocl.occlusion_type.value
                passport.occlusion_retain_sec= ocl.retain_sec
                logger.debug(
                    f"SUSPEND {passport.visitor_id}: {ocl.occlusion_type.value} "
                    f"retain={ocl.retain_sec}s conf={ocl.confidence:.2f}"
                )

            self._suspended[passport.visitor_id] = passport
            
            # Create Ghost
            vx = 0.0; vy = 0.0 # would be calculated from history
            cx = (passport.bbox_xyxy[0] + passport.bbox_xyxy[2]) / 2 / frame_w
            cy = (passport.bbox_xyxy[1] + passport.bbox_xyxy[3]) / 2 / frame_h
            self._ghosts.create_ghost(
                passport.visitor_id, cx, cy, vx, vy, camera_id, passport.zone_id,
                passport.final_confidence, ocl.retain_sec, time.time()
            )
            
            # Create Shadow
            self._shadows.create_shadow(
                passport.visitor_id, passport.bbox_xyxy, 
                vx * frame_w, vy * frame_h, camera_id, passport.zone_id
            )

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
        SUSPENDED → EXPIRED after adaptive occlusion_retain_sec.
        Keeps EXPIRED archive for audit; does not delete passports.
        """
        to_expire = [
            vid for vid, p in self._suspended.items()
            if (wall_time - p.last_seen) > p.occlusion_retain_sec
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
        current_zone: Optional[str] = None,
        fingerprint = None,   # AppearanceFingerprint or None
        trajectory_score: float = 0.5,
        det_conf: float = 1.0,
        track_conf: float = 1.0,
    ) -> Tuple[Optional[str], float]:
        """
        Find best matching SUSPENDED passport using the ConsensusIdentityEngine.
        Includes Courtroom adversarial evaluation for ambiguous matches.
        """
        candidates = []

        for vid, passport in self._suspended.items():
            age = wall_time - passport.last_seen

            # Use per-passport occlusion retention window (adaptive)
            retain_sec = passport.occlusion_retain_sec
            if age > retain_sec:
                continue   # beyond this passport's window

            same_cam = (passport.last_camera == camera_id)

            if not same_cam and age > cfg.CROSS_CAM_TIME_WINDOW_SEC:
                continue

            # ── Signal 1: ReID embedding similarity ──────────────────
            reid_sim = cosine_similarity(passport.appearance_embedding, embedding)

            # ── Signal 2: AppearanceFingerprint holistic match ────────
            fp_score = 0.5   # neutral if unavailable
            if fingerprint is not None and passport.last_reid_explanation:
                # Use last stored fingerprint score if we have it
                fp_score = passport.last_reid_explanation.get(
                    "signals", {}
                ).get("fingerprint", 0.5)

            # ── Signal 3: Trajectory similarity (passed in) ───────────
            # Caller computes this from VisitorMemory if available

            # ── Signal 4: Temporal score ──────────────────────────────
            temporal = max(0.0, 1.0 - age / max(retain_sec, 1.0))

            # ── Signal 5: Camera transition probability ───────────────
            cam_transition = self._store_graph.camera_transition_probability(
                passport.last_camera, camera_id, age
            )

            # ── Signal 6: Zone plausibility ───────────────────────────
            zone_plaus = self._store_graph.transition_probability(
                passport.zone_id, current_zone
            )

            # ── Signal 7: Detection score (from passport) ─────────────
            det_score = passport.last_det_conf

            # ── Signal 8: Track health (normalised) ───────────────────
            health_norm = self._health.normalised(vid)
            
            # ── Signal 9: Group Continuity ────────────────────────────────
            group_boost = self._groups.group_confidence_boost(vid, set()) # Simplified for now
            
            # ── Signal 10: Staff Reputation ───────────────────────────────
            staff_score = passport.staff_confidence if passport.is_staff else 0.5
            
            # ── Signal 11: Visitor DNA ────────────────────────────────────
            # Compare current trajectory/behavior with stored DNA
            dna_score = 0.5 # Default neutral

            signals = ConsensusSignals(
                reid_score             = reid_sim,
                fingerprint_score      = fp_score,
                trajectory_score       = trajectory_score,
                temporal_score         = temporal,
                camera_transition_score= cam_transition,
                zone_plausibility_score= zone_plaus,
                detection_score        = det_score,
                track_health           = health_norm,
                group_continuity_score = group_boost,
                staff_reputation_score = staff_score,
                visitor_dna_score      = dna_score,
                candidate_visitor_id   = vid,
                age_sec                = age,
                cam_from               = passport.last_camera,
                cam_to                 = camera_id,
                zone_from              = passport.zone_id,
                zone_to                = current_zone,
            )
            candidates.append((vid, signals))

        if not candidates:
            return None, 0.0

        result = self._consensus.decide_batch(candidates)
        if result is None:
            return None, 0.0

        best_vid, decision = result
        
        # ── Identity Courtroom Evaluation ──────────────────────────────
        verdict = self._courtroom.adjudicate(
            candidate_signals=decision.explanation.get("signals", {}),
            base_score=decision.identity_score,
            confidence_band=decision.confidence_band,
            context=decision.explanation.get("context", {})
        )
        
        if verdict:
            decision.should_associate = verdict.should_match
            decision.identity_score = verdict.confidence
            decision.explanation["courtroom_verdict"] = verdict.to_dict()
            logger.debug(f"Courtroom changed match outcome for {best_vid}: {verdict.judge_rationale}")

        if not decision.should_associate:
            # Courtroom or Consensus rejected the match
            return None, 0.0

        # Store explanation on passport for GUI / audit
        passport = self._suspended.get(best_vid)
        if passport:
            passport.last_reid_explanation = decision.explanation
            
            # Record Evidence to Ledger
            self._evidence.record_from_consensus(
                visitor_id=best_vid,
                decision_dict=decision.explanation,
                matched_to=None,
                courtroom_verdict=verdict.to_dict() if verdict else None
            )
            # Update health based on decision quality
            if decision.confidence_band == "LOW":
                self._health.on_ambiguous_match(best_vid)
            elif decision.confidence_band in ("MEDIUM", "HIGH"):
                self._health.on_reasso_success(best_vid)
            if "competing_match" in decision.explanation:
                self._health.on_competing_association(best_vid)

        explanation_str = self._consensus.format_explanation(decision)
        logger.info(f"[CONSENSUS]\n{explanation_str}")

        return best_vid, decision.identity_score

    # ------------------------------------------------------------------
    # Accessors for health and consensus (used by audit + GUI)
    # ------------------------------------------------------------------

    def health_score(self, visitor_id: str) -> float:
        return self._health.score(visitor_id)

    def health_normalised(self, visitor_id: str) -> float:
        return self._health.normalised(visitor_id)

    def trust_level(self, visitor_id: str) -> str:
        return self._health.trust_level(visitor_id)

    def last_explanation(self, visitor_id: str) -> Optional[dict]:
        p = self.get_passport(visitor_id)
        return p.last_reid_explanation if p else None

    def on_frame_observed(self, visitor_id: str):
        """Call once per frame while visitor is actively tracked."""
        self._health.on_frame_observed(visitor_id)

    def all_health_scores(self) -> dict:
        return self._health.all_scores()
