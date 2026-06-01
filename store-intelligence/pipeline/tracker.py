"""
tracker.py — Visitor identity manager.

Two-layer identity system:
  Layer 1: track_id  — ephemeral, from ByteTrack
  Layer 2: visitor_id — stable, persistent across occlusions and cameras

Re-ID strategy:
  - Appearance embedding similarity (OSNet / torchreid when available,
    falls back to HOG+color histogram when not)
  - Spatial plausibility (same camera, nearby bbox)
  - Temporal plausibility (gap not too large)
  - Camera transition plausibility (known camera adjacency)

The identity manager keeps a short-term memory of recently lost tracks
so a person who briefly disappears (behind shelf, occlusion) is re-associated
rather than creating a new visitor_id.
"""

import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Camera adjacency map
# Used to gate cross-camera re-ID (only plausible transitions allowed)
# ---------------------------------------------------------------------------
CAMERA_ADJACENCY = {
    "CAM_FLOOR_01":  {"CAM_FLOOR_02", "CAM_ENTRY_03"},
    "CAM_FLOOR_02":  {"CAM_FLOOR_01", "CAM_BILLING_05"},
    "CAM_ENTRY_03":  {"CAM_FLOOR_01"},
    "CAM_GODOWN_04": {"CAM_FLOOR_02"},
    "CAM_BILLING_05":{"CAM_FLOOR_02"},
}

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
LOST_TRACK_MEMORY_SEC     = 8.0    # keep lost tracks for re-association
EMBEDDING_SIM_THRESHOLD   = 0.65   # cosine similarity to re-associate
SPATIAL_PIXEL_THRESHOLD   = 0.25   # fraction of frame dimension
CROSS_CAM_TIME_WINDOW_SEC = 5.0    # max gap for cross-camera re-ID


# ---------------------------------------------------------------------------
# Per-track state
# ---------------------------------------------------------------------------
@dataclass
class TrackState:
    visitor_id:  str
    track_id:    int
    camera_id:   str
    bbox_xyxy:   Tuple[float, float, float, float]
    embedding:   Optional[np.ndarray]   # appearance embedding
    last_seen:   float                  # wall-clock time
    is_active:   bool   = True
    is_staff:    bool   = False
    staff_conf:  float  = 0.0
    zone_id:     Optional[str] = None
    session_seq: int    = 0
    session_start: float = field(default_factory=time.time)

    # entry/exit state
    has_entered: bool   = False
    has_exited:  bool   = False
    reentry_count: int  = 0

    # zone dwell tracking
    zone_enter_time: Optional[float] = None
    last_dwell_emit: Optional[float] = None


# ---------------------------------------------------------------------------
# Appearance embedding extractor
# ---------------------------------------------------------------------------
class EmbeddingExtractor:
    """
    Tries to use torchreid (OSNet) for appearance embeddings.
    Falls back to a fast HOG+color histogram if torchreid is unavailable.
    """

    def __init__(self):
        self._model = None
        self._use_torch = False
        self._try_load_torch()

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
            logger.info("EmbeddingExtractor: using OSNet (torchreid)")
        except Exception as e:
            logger.info(f"EmbeddingExtractor: torchreid unavailable ({e}), "
                        f"using HOG+color fallback")
            self._use_torch = False

    def extract(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        if self._use_torch:
            return self._extract_osnet(crop_bgr)
        else:
            return self._extract_fallback(crop_bgr)

    def _extract_osnet(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            import torch
            import cv2
            img = cv2.resize(crop_bgr, (128, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std
            tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
            with torch.no_grad():
                feat = self._model(tensor)
            emb = feat.squeeze().numpy()
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception as e:
            logger.debug(f"OSNet extraction failed: {e}")
            return self._extract_fallback(crop_bgr)

    def _extract_fallback(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Fast fallback: HSV color histogram (128-bin) concatenated with
        a coarse spatial layout histogram.
        """
        try:
            import cv2
            resized = cv2.resize(crop_bgr, (64, 128))
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            # H: 32 bins, S: 32 bins, V: 32 bins per body half → 192 total
            h_hist = np.histogram(hsv[:64, :, 0], bins=32, range=(0, 180))[0]
            s_hist = np.histogram(hsv[:64, :, 1], bins=32, range=(0, 256))[0]
            v_hist = np.histogram(hsv[64:, :, 2], bins=32, range=(0, 256))[0]
            emb = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception:
            return np.zeros(96, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Visitor identity manager
# ---------------------------------------------------------------------------

class VisitorIdentityManager:
    """
    Maintains stable visitor_ids across ephemeral track_ids.

    Key operations:
      resolve(track_id, camera_id, bbox, embedding, timestamp)
        → visitor_id
        Checks active tracks first, then lost-track memory.

      mark_lost(track_id, camera_id)
        Moves track to lost memory.

      mark_exited(visitor_id)
        Records that this visitor left through the door.
    """

    def __init__(self):
        self._extractor = EmbeddingExtractor()

        # active: (track_id, camera_id) → TrackState
        self._active: Dict[Tuple[int, str], TrackState] = {}

        # lost: visitor_id → TrackState (for re-association window)
        self._lost: Dict[str, TrackState] = {}

        # visitor_id → list of all TrackStates (history)
        self._history: Dict[str, List[TrackState]] = defaultdict(list)

        # visitor_id → exit count (for reentry detection)
        self._exit_count: Dict[str, int] = defaultdict(int)

        # session sequence counter per visitor
        self._session_seq: Dict[str, int] = defaultdict(int)

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

    # ------------------------------------------------------------------
    def resolve(self, track_id: int, camera_id: str,
                bbox_xyxy: Tuple, embedding: Optional[np.ndarray],
                wall_time: float) -> Tuple[str, bool]:
        """
        Returns (visitor_id, is_new_visitor).
        is_new_visitor = True only when this is genuinely a brand-new person.
        """
        key = (track_id, camera_id)

        # 1. Already tracking this track → return existing visitor_id
        if key in self._active:
            state = self._active[key]
            state.last_seen = wall_time
            state.bbox_xyxy = bbox_xyxy
            if embedding is not None:
                state.embedding = embedding
            return state.visitor_id, False

        # 2. New track → try to match against lost tracks (Sequential transition)
        matched_vid = self._match_lost(camera_id, bbox_xyxy, embedding, wall_time)

        if matched_vid is not None:
            # Re-associate with a lost visitor
            lost_state = self._lost.pop(matched_vid)
            lost_state.track_id  = track_id
            lost_state.camera_id = camera_id
            lost_state.bbox_xyxy = bbox_xyxy
            lost_state.last_seen = wall_time
            lost_state.is_active = True
            if embedding is not None:
                lost_state.embedding = embedding
            self._active[key] = lost_state
            logger.debug(f"Re-associated track {track_id}@{camera_id} → {matched_vid}")
            return matched_vid, False

        # 2.5. Try to match against ACTIVE tracks in overlapping cameras (Simultaneous visibility)
        if embedding is not None:
            best_sim = 0.0
            for existing_key, existing_state in self._active.items():
                active_cam = existing_key[1]
                if active_cam == camera_id:
                    continue
                # Only check if the cameras are adjacent/overlapping
                if camera_id in CAMERA_ADJACENCY.get(active_cam, set()) or active_cam in CAMERA_ADJACENCY.get(camera_id, set()):
                    sim = cosine_similarity(existing_state.embedding, embedding)
                    # We require a slightly higher threshold (0.70) for active-overlap to prevent merging people passing each other
                    if sim > best_sim and sim >= EMBEDDING_SIM_THRESHOLD + 0.05:
                        best_sim = sim
                        matched_vid = existing_state.visitor_id

            if matched_vid is not None:
                # Share the visitor_id but create a new TrackState for this specific camera view
                state = TrackState(
                    visitor_id=matched_vid,
                    track_id=track_id,
                    camera_id=camera_id,
                    bbox_xyxy=bbox_xyxy,
                    embedding=embedding,
                    last_seen=wall_time,
                    session_start=wall_time, # Share same timeline ideally, but wall_time is fine
                )
                self._active[key] = state
                logger.debug(f"Overlapping camera match {track_id}@{camera_id} → {matched_vid}")
                return matched_vid, False

        # 3. Truly new visitor
        visitor_id = "VIS_" + uuid.uuid4().hex[:6]
        state = TrackState(
            visitor_id=visitor_id,
            track_id=track_id,
            camera_id=camera_id,
            bbox_xyxy=bbox_xyxy,
            embedding=embedding,
            last_seen=wall_time,
            session_start=wall_time,
        )
        self._active[key] = state
        self._history[visitor_id].append(state)
        logger.debug(f"New visitor {visitor_id} track {track_id}@{camera_id}")
        return visitor_id, True

    # ------------------------------------------------------------------
    def _match_lost(self, camera_id: str, bbox_xyxy: Tuple,
                    embedding: Optional[np.ndarray],
                    wall_time: float) -> Optional[str]:
        """
        Find best matching lost track.
        Returns visitor_id or None.
        """
        best_vid  = None
        best_score = 0.0

        for vid, state in self._lost.items():
            age = wall_time - state.last_seen
            if age > LOST_TRACK_MEMORY_SEC:
                continue

            # Camera plausibility
            same_cam = (state.camera_id == camera_id)
            adjacent = camera_id in CAMERA_ADJACENCY.get(state.camera_id, set())
            if not same_cam and not adjacent:
                continue

            # Cross-camera time window
            if not same_cam and age > CROSS_CAM_TIME_WINDOW_SEC:
                continue

            # Appearance similarity
            app_sim = cosine_similarity(state.embedding, embedding)

            # Spatial proximity (same camera only)
            if same_cam:
                cx_new = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
                cy_new = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
                cx_old = (state.bbox_xyxy[0] + state.bbox_xyxy[2]) / 2
                cy_old = (state.bbox_xyxy[1] + state.bbox_xyxy[3]) / 2
                # Normalise by estimated frame size (use 1920×1080 as reference)
                spatial_dist = (abs(cx_new - cx_old) / 1920 +
                                abs(cy_new - cy_old) / 1080) / 2
                spatial_score = max(0.0, 1.0 - spatial_dist / SPATIAL_PIXEL_THRESHOLD)
            else:
                spatial_score = 0.5   # cross-camera: rely on appearance

            # Temporal score (more recent = better)
            time_score = max(0.0, 1.0 - age / LOST_TRACK_MEMORY_SEC)

            # Composite
            score = app_sim * 0.55 + spatial_score * 0.25 + time_score * 0.20

            if score > best_score and score >= EMBEDDING_SIM_THRESHOLD:
                best_score = score
                best_vid   = vid

        return best_vid

    # ------------------------------------------------------------------
    def mark_lost(self, track_id: int, camera_id: str):
        """Called when tracker drops a track."""
        key = (track_id, camera_id)
        if key in self._active:
            state = self._active.pop(key)
            state.is_active = False
            self._lost[state.visitor_id] = state
            logger.debug(f"Track {track_id}@{camera_id} → lost ({state.visitor_id})")

    def mark_exited(self, visitor_id: str):
        self._exit_count[visitor_id] += 1

    def exit_count(self, visitor_id: str) -> int:
        return self._exit_count.get(visitor_id, 0)

    def get_state(self, visitor_id: str) -> Optional[TrackState]:
        # Search active
        for state in self._active.values():
            if state.visitor_id == visitor_id:
                return state
        # Search lost
        return self._lost.get(visitor_id)

    def get_active_state_by_track(self, track_id: int,
                                  camera_id: str) -> Optional[TrackState]:
        return self._active.get((track_id, camera_id))

    def increment_session_seq(self, visitor_id: str) -> int:
        self._session_seq[visitor_id] += 1
        return self._session_seq[visitor_id]

    def get_session_seq(self, visitor_id: str) -> int:
        return self._session_seq.get(visitor_id, 0)

    def purge_stale_lost(self, wall_time: float):
        """Remove lost tracks older than memory window."""
        stale = [vid for vid, s in self._lost.items()
                 if wall_time - s.last_seen > LOST_TRACK_MEMORY_SEC * 2]
        for vid in stale:
            del self._lost[vid]

    @property
    def active_count(self) -> int:
        return len({state.visitor_id for state in self._active.values()})

    @property
    def lost_count(self) -> int:
        return len(self._lost)
