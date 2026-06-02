"""
staff.py — Staff detection classifier.

Staff at Purplle stores wear ALL-BLACK shirt + pants.
This is a strong visual signal we exploit via HSV color analysis on the
detected person crop, combined with behavioural heuristics.

Classification pipeline:
  1. Black-clothing score from HSV histogram on upper + lower body crops
  2. Long-presence score (staff are present much longer than customers)
  3. Repeated-zone-traversal score (staff move through all zones)
  4. Movement-frequency score (staff appear in many frames, short stops)

Final decision: soft score → threshold → is_staff bool
"""

import logging
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional
import numpy as np

from config import cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HSV thresholds for "black" clothing
# Hue: any (doesn't matter for dark colors)
# Saturation: low  (black is desaturated)
# Value: very low  (black is dark)
# ---------------------------------------------------------------------------
BLACK_SAT_MAX   = cfg.BLACK_SAT_MAX
BLACK_VAL_MAX   = cfg.BLACK_VAL_MAX
BLACK_PIXEL_RATIO_THRESHOLD = cfg.BLACK_ZONE_THRESHOLD


def compute_black_ratio(crop_bgr: np.ndarray) -> float:
    """
    Given a BGR image crop, return fraction of pixels that are 'black'
    (low saturation AND low value in HSV).
    Returns 0.0 if crop is empty or invalid.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    try:
        import cv2
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(float)
        v = hsv[:, :, 2].astype(float)
        black_mask = (s < BLACK_SAT_MAX) & (v < BLACK_VAL_MAX)
        ratio = float(black_mask.sum()) / max(black_mask.size, 1)
        return ratio
    except Exception as e:
        logger.debug(f"black_ratio error: {e}")
        return 0.0


def black_clothing_score(frame_bgr: np.ndarray,
                         bbox_xyxy: Tuple[int, int, int, int]) -> float:
    """
    Uses a robust 5-zone body sampling technique (chest, abdomen, upper legs, left arm, right arm)
    to determine the strength of the black uniform signal, ignoring the face and shoes.
    Returns a score between 0.0 and 1.0 based on how many zones are 'black'.
    """
    if frame_bgr is None:
        return 0.0
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    H, W = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    ph = y2 - y1
    pw = x2 - x1
    if pw < 10 or ph < 20:
        return 0.0

    # Define 5 body zones specifically targeting clothing (ignoring head and shoes)
    zones = [
        frame_bgr[y1+int(ph*0.20):y1+int(ph*0.40), x1:x2],                          # chest
        frame_bgr[y1+int(ph*0.40):y1+int(ph*0.60), x1:x2],                          # abdomen
        frame_bgr[y1+int(ph*0.60):y1+int(ph*0.80), x1:x2],                          # upper legs
        frame_bgr[y1+int(ph*0.25):y1+int(ph*0.75), x1:x1+int(pw*0.35)],             # left side/arm
        frame_bgr[y1+int(ph*0.25):y1+int(ph*0.75), x1+int(pw*0.65):x2],             # right side/arm
    ]
    
    # Calculate black ratio for each zone
    zone_ratios = [compute_black_ratio(z) for z in zones]
    
    # A zone is "dark" if >= 55% of its pixels match the black HSV threshold
    # (raised from 45% to reduce false positives on dark-navy/charcoal clothing)
    dark_votes = sum(1 for r in zone_ratios if r >= 0.55)
    
    # Require at least 2 dark zones to score anything (avoids one-zone false triggers)
    if dark_votes < 2:
        return 0.0
    
    # Map votes to a continuous confidence score between 0.0 and 1.0
    # 2 zones = 0.20 (not staff), 3 zones = 0.50, 4 zones = 0.75, 5 zones = 1.0
    score_map = {2: 0.20, 3: 0.50, 4: 0.75, 5: 1.00}
    score = score_map.get(dark_votes, dark_votes / 5.0)
    
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Behavioural accumulator per visitor_id
# ---------------------------------------------------------------------------

class StaffBehaviourTracker:
    """
    Tracks per-visitor behavioural signals over time.
    Staff tend to:
      - appear in many frames (high frame_count)
      - traverse many distinct zones
      - appear across multiple cameras
      - have short dwell per zone (operational movement, not browsing)
    """

    def __init__(self):
        # visitor_id → stats
        self._frame_count:   Dict[str, int]   = defaultdict(int)
        self._zones_visited: Dict[str, set]   = defaultdict(set)
        self._cameras_seen:  Dict[str, set]   = defaultdict(set)
        self._black_scores:  Dict[str, deque] = defaultdict(lambda: deque(maxlen=30))
        self._is_staff_cache: Dict[str, bool] = {}
        self._cache_dirty:   Dict[str, bool]  = defaultdict(lambda: True)

        # Thresholds
        self.MIN_FRAMES_FOR_STAFF_DECISION = cfg.STAFF_MIN_FRAMES
        self.STAFF_SCORE_THRESHOLD         = cfg.STAFF_SCORE_THRESHOLD

    def update(self, visitor_id: str, frame_bgr: np.ndarray,
               bbox_xyxy: Tuple[int, int, int, int],
               camera_id: str, zone_id: Optional[str],
               skip_clothing: bool = False):
        self._frame_count[visitor_id] += 1
        self._cameras_seen[visitor_id].add(camera_id)
        if zone_id:
            self._zones_visited[visitor_id].add(zone_id)

        if not skip_clothing:
            bs = black_clothing_score(frame_bgr, bbox_xyxy)
            self._black_scores[visitor_id].append(bs)
        self._cache_dirty[visitor_id] = True

    def is_staff(self, visitor_id: str) -> Tuple[bool, float]:
        """
        Returns (is_staff, confidence).
        Confidence is low when we have few observations.
        """
        fc = self._frame_count.get(visitor_id, 0)
        if fc < 5:
            return False, 0.30    # not enough data yet

        # --- Black clothing score ---
        scores = list(self._black_scores.get(visitor_id, []))
        if scores:
            # Use 75th percentile: staff consistently wear black
            black_p75 = float(np.percentile(scores, 75))
        else:
            black_p75 = 0.0

        # --- Presence score (staff are present for many frames) ---
        # Cap at 500 frames ~33 seconds at 15fps
        presence_score = min(fc / 500.0, 1.0)

        # --- Zone diversity score (staff traverse many zones) ---
        n_zones = len(self._zones_visited.get(visitor_id, set()))
        zone_diversity = min(n_zones / 6.0, 1.0)

        # --- Camera diversity (staff appear across cameras) ---
        n_cams = len(self._cameras_seen.get(visitor_id, set()))
        cam_diversity = min((n_cams - 1) / 3.0, 1.0) if n_cams > 1 else 0.0

        # --- Composite score ---
        # Black clothing is the PRIMARY and MOST RELIABLE signal (80% weight)
        # Behavioural patterns provide secondary confirmation (20% weight)
        composite = (
            black_p75      * cfg.STAFF_W_BLACK +
            presence_score * cfg.STAFF_W_PRESENCE +
            zone_diversity * cfg.STAFF_W_ZONE_DIV +
            cam_diversity  * cfg.STAFF_W_CAM_DIV
        )

        # Confidence: how many frames we have
        data_confidence = min(fc / float(self.MIN_FRAMES_FOR_STAFF_DECISION), 1.0)
        # Scale confidence: low if composite is near threshold
        margin = abs(composite - self.STAFF_SCORE_THRESHOLD)
        detection_confidence = min(0.5 + margin * 2.0, 0.99)
        final_confidence = detection_confidence * data_confidence

        is_staff_flag = composite >= self.STAFF_SCORE_THRESHOLD

        if self._cache_dirty.get(visitor_id, True):
            self._is_staff_cache[visitor_id] = is_staff_flag
            self._cache_dirty[visitor_id] = False

        return is_staff_flag, round(float(final_confidence), 3)

    def get_black_score(self, visitor_id: str) -> float:
        scores = list(self._black_scores.get(visitor_id, []))
        if not scores:
            return 0.0
        return float(np.percentile(scores, 75))

    def summary(self, visitor_id: str) -> dict:
        is_s, conf = self.is_staff(visitor_id)
        return {
            "visitor_id":    visitor_id,
            "is_staff":      is_s,
            "confidence":    conf,
            "frame_count":   self._frame_count.get(visitor_id, 0),
            "zones_visited": list(self._zones_visited.get(visitor_id, set())),
            "cameras_seen":  list(self._cameras_seen.get(visitor_id, set())),
            "black_p75":     self.get_black_score(visitor_id),
        }


# ---------------------------------------------------------------------------
# StaffBehaviorProfile — behavioral patterns for staff identification.
# A person wearing black is NOT automatically staff.
# A person BEHAVING like staff raises confidence significantly.
# Combined: final_staff_score = 0.60 × uniform_score + 0.40 × behavior_score
# ---------------------------------------------------------------------------
class StaffBehaviorProfile:
    STAFF_MIN_SESSION_SEC    = 900.0
    STAFF_MIN_ZONE_COUNT     = 4
    STAFF_MIN_CAMERA_COUNT   = 3

    def __init__(self):
        self._zone_visits       = defaultdict(list)
        self._camera_transitions= defaultdict(list)
        self._session_start     = {}
        self._session_count     = defaultdict(int)

    def update(self, visitor_id: str, zone_id,
               camera_id: str, wall_time: float, event_type: str = ""):
        if visitor_id not in self._session_start:
            self._session_start[visitor_id] = wall_time
        if event_type in ("ENTRY", "REENTRY"):
            self._session_count[visitor_id] += 1
        if zone_id:
            visits = self._zone_visits[visitor_id]
            if not visits or visits[-1][0] != zone_id:
                self._zone_visits[visitor_id].append([zone_id, wall_time, wall_time])
            else:
                self._zone_visits[visitor_id][-1][2] = wall_time
        cams = self._camera_transitions[visitor_id]
        if not cams or cams[-1][0] != camera_id:
            self._camera_transitions[visitor_id].append((camera_id, wall_time))

    def behavior_staff_score(self, visitor_id: str, wall_time: float) -> float:
        visits      = self._zone_visits.get(visitor_id, [])
        cams        = self._camera_transitions.get(visitor_id, [])
        session_sec = wall_time - self._session_start.get(visitor_id, wall_time)
        if session_sec < 120:
            return 0.5
        signals = [
            min(1.0, session_sec / 3600.0),
            min(1.0, len({v[0] for v in visits}) / self.STAFF_MIN_ZONE_COUNT),
            min(1.0, len({c[0] for c in cams})   / self.STAFF_MIN_CAMERA_COUNT),
            min(1.0, self._session_count.get(visitor_id, 0) / 3.0),
        ]
        weights = [0.35, 0.30, 0.25, 0.10]
        return round(sum(s * w for s, w in zip(signals, weights)), 4)

    def combined_staff_score(self, visitor_id: str, wall_time: float,
                             uniform_score: float) -> float:
        return round(0.60 * uniform_score + 0.40 * self.behavior_staff_score(visitor_id, wall_time), 4)
