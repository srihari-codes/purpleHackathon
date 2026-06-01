"""
visitor_dna.py — Behavioral Fingerprint for Identity Matching.

Tracks behavioral patterns per visitor to augment appearance-based matching.
Maintains:
- Walking speed (EWMA)
- Movement rhythm (directional histograms)
- Zone visitation patterns (ordered list of zones)
- Dwell & queue tendencies
- Transition habits
"""

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class VisitorDNA:
    visitor_id:          str
    preferred_speed:     float = 0.0  # normalised units/sec
    movement_rhythm:     List[float] = field(default_factory=lambda: [0.0] * cfg.DNA_RHYTHM_BINS)
    zone_pattern:        List[str] = field(default_factory=list)
    queue_tendency:      float = 0.0  # 0 to 1
    dwell_tendency:      float = 0.0  # average dwell seconds
    transition_habits:   Dict[Tuple[str, str], int] = field(default_factory=dict)
    
    # Internal trackers
    _last_pos:           Optional[Tuple[float, float, float]] = None # x, y, wall_time
    _total_dwell_ms:     int = 0
    _queue_joins:        int = 0
    _total_observations: int = 0

    def to_dict(self) -> dict:
        return {
            "visitor_id": self.visitor_id,
            "preferred_speed": round(self.preferred_speed, 3),
            "zone_pattern": self.zone_pattern,
            "queue_tendency": round(self.queue_tendency, 3),
            "dwell_tendency": round(self.dwell_tendency, 3),
            "total_dwell_s": self._total_dwell_ms / 1000.0,
            "queue_joins": self._queue_joins,
        }


class VisitorDNATracker:
    """Tracks and compares behavioral DNA across visitors."""

    def __init__(self):
        self._dna_store: Dict[str, VisitorDNA] = {}

    def _get(self, visitor_id: str) -> VisitorDNA:
        if visitor_id not in self._dna_store:
            self._dna_store[visitor_id] = VisitorDNA(visitor_id=visitor_id)
        return self._dna_store[visitor_id]

    def update(
        self,
        visitor_id: str,
        cx: float, cy: float,
        zone_id: Optional[str],
        camera_id: str,
        wall_time: float,
        event_type: str = ""
    ):
        dna = self._get(visitor_id)
        dna._total_observations += 1

        # 1. Update walking speed and movement rhythm
        if dna._last_pos is not None:
            last_x, last_y, last_time = dna._last_pos
            dt = wall_time - last_time
            if 0 < dt < 2.0:
                dx = cx - last_x
                dy = cy - last_y
                dist = math.sqrt(dx*dx + dy*dy)
                speed = dist / dt
                
                # EWMA update for speed
                alpha = cfg.DNA_SPEED_EWMA_ALPHA
                if dna.preferred_speed == 0.0:
                    dna.preferred_speed = speed
                else:
                    dna.preferred_speed = (1 - alpha) * dna.preferred_speed + alpha * speed

                # Update rhythm (directional histogram)
                if dist > 0.01:
                    angle = math.atan2(dy, dx)
                    if angle < 0:
                        angle += 2 * math.pi
                    bin_idx = int((angle / (2 * math.pi)) * cfg.DNA_RHYTHM_BINS) % cfg.DNA_RHYTHM_BINS
                    dna.movement_rhythm[bin_idx] += 1.0

        dna._last_pos = (cx, cy, wall_time)

        # 2. Update zone patterns and transitions
        if zone_id:
            if not dna.zone_pattern or dna.zone_pattern[-1] != zone_id:
                if dna.zone_pattern:
                    prev_zone = dna.zone_pattern[-1]
                    trans = (prev_zone, zone_id)
                    dna.transition_habits[trans] = dna.transition_habits.get(trans, 0) + 1
                dna.zone_pattern.append(zone_id)
                # Cap the pattern length
                if len(dna.zone_pattern) > cfg.DNA_PATTERN_WINDOW:
                    dna.zone_pattern = dna.zone_pattern[-cfg.DNA_PATTERN_WINDOW:]

        # 3. Update queue & dwell tendencies
        if event_type == "BILLING_QUEUE_JOIN":
            dna._queue_joins += 1
            dna.queue_tendency = min(1.0, dna._queue_joins / 2.0)
        elif event_type == "ZONE_DWELL":
            # For simplicity, we just bump total dwell by a small amount or use external tracking
            pass
            
    def record_dwell(self, visitor_id: str, dwell_ms: int):
        dna = self._get(visitor_id)
        dna._total_dwell_ms += dwell_ms
        if len(dna.zone_pattern) > 0:
            dna.dwell_tendency = (dna._total_dwell_ms / 1000.0) / len(dna.zone_pattern)

    def compare(self, vid_a: str, vid_b: str) -> float:
        """
        Compare the behavioral DNA of two visitors.
        Returns similarity [0.0 - 1.0].
        """
        if vid_a not in self._dna_store or vid_b not in self._dna_store:
            return 0.5  # Neutral if we lack data

        dna_a = self._dna_store[vid_a]
        dna_b = self._dna_store[vid_b]

        # Not enough data for meaningful comparison
        if dna_a._total_observations < 10 or dna_b._total_observations < 10:
            return 0.5

        score = 0.0

        # Speed similarity
        if dna_a.preferred_speed > 0 and dna_b.preferred_speed > 0:
            speed_diff = abs(dna_a.preferred_speed - dna_b.preferred_speed)
            # 0.1 normalized units diff drops score to ~0
            speed_sim = max(0.0, 1.0 - (speed_diff / 0.1))
            score += speed_sim * cfg.DNA_COMPARE_SPEED_WEIGHT
        else:
            score += 0.5 * cfg.DNA_COMPARE_SPEED_WEIGHT

        # Rhythm similarity (cosine sim of histograms)
        norm_a = sum(dna_a.movement_rhythm)
        norm_b = sum(dna_b.movement_rhythm)
        if norm_a > 0 and norm_b > 0:
            dot = sum(a * b for a, b in zip(dna_a.movement_rhythm, dna_b.movement_rhythm))
            rhythm_sim = dot / (math.sqrt(sum(a*a for a in dna_a.movement_rhythm)) * math.sqrt(sum(b*b for b in dna_b.movement_rhythm)))
            score += rhythm_sim * cfg.DNA_COMPARE_RHYTHM_WEIGHT
        else:
            score += 0.5 * cfg.DNA_COMPARE_RHYTHM_WEIGHT

        # Pattern similarity (Jaccard on sets, sequence match would be better but harder)
        set_a = set(dna_a.zone_pattern)
        set_b = set(dna_b.zone_pattern)
        if set_a and set_b:
            intersection = len(set_a.intersection(set_b))
            union = len(set_a.union(set_b))
            pattern_sim = intersection / union
            score += pattern_sim * cfg.DNA_COMPARE_PATTERN_WEIGHT
        else:
            score += 0.5 * cfg.DNA_COMPARE_PATTERN_WEIGHT

        # Queue tendency
        queue_diff = abs(dna_a.queue_tendency - dna_b.queue_tendency)
        queue_sim = max(0.0, 1.0 - queue_diff)
        score += queue_sim * cfg.DNA_COMPARE_QUEUE_WEIGHT

        # Dwell tendency
        if dna_a.dwell_tendency > 0 and dna_b.dwell_tendency > 0:
            dwell_ratio = min(dna_a.dwell_tendency, dna_b.dwell_tendency) / max(dna_a.dwell_tendency, dna_b.dwell_tendency)
            score += dwell_ratio * cfg.DNA_COMPARE_DWELL_WEIGHT
        else:
            score += 0.5 * cfg.DNA_COMPARE_DWELL_WEIGHT

        return min(1.0, max(0.0, score))

    def get_dna(self, visitor_id: str) -> Optional[VisitorDNA]:
        return self._dna_store.get(visitor_id)
