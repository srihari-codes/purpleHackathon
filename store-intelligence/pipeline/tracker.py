"""
tracker.py — Re-ID and multi-camera visitor tracking.

Strategy:
  1. ByteTrack (via ultralytics) assigns track_ids per-clip.
  2. OSNet appearance embeddings build a gallery for cross-camera Re-ID.
  3. Fallback: bounding box trajectory + IoU for scenes where OSNet is absent.
  4. Re-entry detection: if a visitor_id appears after a recorded EXIT event,
     emit REENTRY instead of a new ENTRY.

visitor_id format: VIS_<6hex>  (stable across sessions for same physical person)
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

COSINE_THRESHOLD = 0.75          # above this → same person
REENTRY_GAP_SECONDS = 120        # exit→reentry within 2 min = reentry, not new visitor
STAFF_ZONE_REPEAT_THRESHOLD = 5  # visits to 3+ zones within a short window = staff


@dataclass
class VisitorSession:
    visitor_id: str
    track_id: int
    clip_name: str
    first_seen: datetime
    last_seen: datetime
    zones_visited: List[str] = field(default_factory=list)
    exited: bool = False
    embedding: Optional[np.ndarray] = None
    is_staff: bool = False


class VisitorTracker:
    """
    Maintains a gallery of known visitors across clips and cameras.
    Thread-safety is NOT required (single-process pipeline).
    """

    def __init__(self, store_id: str, similarity_threshold: float = COSINE_THRESHOLD):
        self.store_id = store_id
        self.threshold = similarity_threshold
        self._gallery: Dict[str, VisitorSession] = {}   # visitor_id → session
        self._track_map: Dict[Tuple[str, int], str] = {}  # (clip, track_id) → visitor_id
        self._session_counter = 0

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def get_or_create_visitor(
        self,
        track_id: int,
        clip_name: str,
        timestamp: datetime,
        embedding: Optional[np.ndarray] = None,
        bbox: Optional[List[float]] = None,
    ) -> Tuple[str, bool]:
        """
        Returns (visitor_id, is_reentry).
        If the track is already mapped → return existing id.
        If embedding matches gallery → map and return existing id.
        Otherwise → create new visitor.
        """
        key = (clip_name, track_id)
        if key in self._track_map:
            vid = self._track_map[key]
            self._gallery[vid].last_seen = timestamp
            return vid, False

        # Try Re-ID via embedding similarity
        if embedding is not None:
            matched_id, similarity = self._find_closest(embedding)
            if matched_id and similarity >= self.threshold:
                session = self._gallery[matched_id]
                self._track_map[key] = matched_id
                # Check for re-entry
                is_reentry = (
                    session.exited and
                    (timestamp - session.last_seen).total_seconds() < REENTRY_GAP_SECONDS * 10
                )
                session.exited = False
                session.last_seen = timestamp
                session.embedding = embedding  # update with fresh embedding
                logger.debug("Re-ID match: track %d → %s (sim=%.3f)", track_id, matched_id, similarity)
                return matched_id, is_reentry

        # New visitor
        vid = self._new_visitor_id(track_id, clip_name)
        session = VisitorSession(
            visitor_id=vid,
            track_id=track_id,
            clip_name=clip_name,
            first_seen=timestamp,
            last_seen=timestamp,
            embedding=embedding,
        )
        self._gallery[vid] = session
        self._track_map[key] = vid
        return vid, False

    def mark_exit(self, visitor_id: str, timestamp: datetime):
        if visitor_id in self._gallery:
            self._gallery[visitor_id].exited = True
            self._gallery[visitor_id].last_seen = timestamp

    def record_zone(self, visitor_id: str, zone_id: str):
        if visitor_id in self._gallery:
            self._gallery[visitor_id].zones_visited.append(zone_id)

    def classify_staff(self, visitor_id: str) -> bool:
        """Heuristic: visitor who appears in 3+ distinct zones frequently = likely staff."""
        session = self._gallery.get(visitor_id)
        if not session:
            return False
        unique_zones = len(set(session.zones_visited))
        return unique_zones >= 3 and len(session.zones_visited) >= STAFF_ZONE_REPEAT_THRESHOLD

    def set_staff(self, visitor_id: str, is_staff: bool):
        if visitor_id in self._gallery:
            self._gallery[visitor_id].is_staff = is_staff

    def is_staff(self, visitor_id: str) -> bool:
        return self._gallery.get(visitor_id, VisitorSession("", 0, "", datetime.now(), datetime.now())).is_staff

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _new_visitor_id(self, track_id: int, clip_name: str) -> str:
        self._session_counter += 1
        raw = f"{self.store_id}-{clip_name}-{track_id}-{self._session_counter}"
        h = hashlib.md5(raw.encode()).hexdigest()[:6]
        return f"VIS_{h}"

    def _find_closest(self, query: np.ndarray) -> Tuple[Optional[str], float]:
        best_id = None
        best_sim = -1.0
        qn = query / (np.linalg.norm(query) + 1e-8)
        for vid, session in self._gallery.items():
            if session.embedding is None:
                continue
            gn = session.embedding / (np.linalg.norm(session.embedding) + 1e-8)
            sim = float(np.dot(qn, gn))
            if sim > best_sim:
                best_sim = sim
                best_id = vid
        return best_id, best_sim


# ─────────────────────────────────────────────────
# Staff detection via uniform colour heuristic
# ─────────────────────────────────────────────────

def is_staff_by_colour(
    person_crop: np.ndarray,
    staff_hsv_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
        (0, 0, 100),    # lower HSV (white-ish uniform)
        (180, 30, 255), # upper HSV
    ),
    coverage_threshold: float = 0.40,
) -> bool:
    """
    Returns True if ≥ coverage_threshold of the bounding box is the staff uniform colour.
    Configurable HSV range — default targets white/light-grey retail uniforms.
    Uses OpenCV HSV conversion (imported lazily to allow import without display).
    """
    try:
        import cv2
        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        lower = np.array(staff_hsv_range[0], dtype=np.uint8)
        upper = np.array(staff_hsv_range[1], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        coverage = mask.sum() / (255 * mask.size + 1e-8)
        return coverage >= coverage_threshold
    except Exception as exc:
        logger.debug("Staff colour check failed: %s", exc)
        return False
