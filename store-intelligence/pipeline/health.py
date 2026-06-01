"""
health.py — Track Health Monitor.

Every active visitor identity has a health score in [0, 100].
Health reflects the system's confidence in the identity's integrity.

High health (≥75): identity is clean, continuously observed, no conflicts.
Medium health (40–74): some ambiguity but still trusted.
Low health (<40): high uncertainty — emit events but flag with lower confidence.

Changes:
  +0.5 / frame  — continuously observed
  +10.0         — successful re-association
  +5.0          — zone transition confirmed by StoreGraph
  −5.0          — ambiguous match (near-threshold)
  −10.0         — competing association (another visitor contested same match)
  −15.0         — zone teleport detected
  −0.3/sec      — while in SUSPENDED state (decays naturally)

Health is exposed as a normalised [0, 1] float for the consensus engine.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class HealthRecord:
    score:            float = cfg.TRACK_HEALTH_INIT
    last_update_time: float = field(default_factory=time.time)
    total_frames:     int   = 0
    reasso_count:     int   = 0
    ambiguity_count:  int   = 0
    teleport_count:   int   = 0


class TrackHealthMonitor:
    """
    Per-visitor health score tracker.

    Thread-safety note: This is called from a single processing thread
    in the current architecture (one detect.py main loop). No locking needed.
    """

    def __init__(self):
        self._records: Dict[str, HealthRecord] = {}

    def _get(self, visitor_id: str) -> HealthRecord:
        if visitor_id not in self._records:
            self._records[visitor_id] = HealthRecord()
        return self._records[visitor_id]

    def _clamp(self, rec: HealthRecord) -> HealthRecord:
        rec.score = max(cfg.TRACK_HEALTH_MIN, min(cfg.TRACK_HEALTH_MAX, rec.score))
        return rec

    # ------------------------------------------------------------------
    # Update events
    # ------------------------------------------------------------------

    def on_frame_observed(self, visitor_id: str):
        """Call once per frame while the visitor is actively detected."""
        rec = self._get(visitor_id)
        rec.score += cfg.TRACK_HEALTH_PER_FRAME_GAIN
        rec.total_frames += 1
        rec.last_update_time = time.time()
        self._clamp(rec)

    def on_reasso_success(self, visitor_id: str):
        """Call when identity successfully re-associated from SUSPENDED."""
        rec = self._get(visitor_id)
        rec.score += cfg.TRACK_HEALTH_REASSO_GAIN
        rec.reasso_count += 1
        rec.last_update_time = time.time()
        self._clamp(rec)
        logger.debug(f"Health +10 (reasso): {visitor_id} → {rec.score:.1f}")

    def on_zone_transition_confirmed(self, visitor_id: str):
        """Call when zone transition is confirmed as plausible by StoreGraph."""
        rec = self._get(visitor_id)
        rec.score += 5.0
        self._clamp(rec)

    def on_ambiguous_match(self, visitor_id: str):
        """Call when a re-association was accepted but near the threshold."""
        rec = self._get(visitor_id)
        rec.score -= cfg.TRACK_HEALTH_AMBIGUOUS_LOSS
        rec.ambiguity_count += 1
        self._clamp(rec)
        logger.debug(f"Health −5 (ambiguous): {visitor_id} → {rec.score:.1f}")

    def on_competing_association(self, visitor_id: str):
        """Call when another candidate also scored near this visitor's match."""
        rec = self._get(visitor_id)
        rec.score -= cfg.TRACK_HEALTH_COMPETE_LOSS
        self._clamp(rec)
        logger.debug(f"Health −10 (compete): {visitor_id} → {rec.score:.1f}")

    def on_zone_teleport(self, visitor_id: str):
        """Call when visitor appears in an implausible zone sequence."""
        rec = self._get(visitor_id)
        rec.score -= cfg.TRACK_HEALTH_TELEPORT_LOSS
        rec.teleport_count += 1
        self._clamp(rec)
        logger.warning(f"Health −15 (teleport): {visitor_id} → {rec.score:.1f}")

    def on_suspended(self, visitor_id: str, elapsed_sec: float):
        """Decay health while in SUSPENDED state. Call periodically."""
        rec = self._get(visitor_id)
        rec.score -= cfg.TRACK_HEALTH_DECAY_PER_SEC * elapsed_sec
        self._clamp(rec)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def score(self, visitor_id: str) -> float:
        """Raw score in [0, 100]."""
        return self._records.get(visitor_id, HealthRecord()).score

    def normalised(self, visitor_id: str) -> float:
        """Score normalised to [0, 1] for use in ConsensusSignals."""
        return self.score(visitor_id) / cfg.TRACK_HEALTH_MAX

    def trust_level(self, visitor_id: str) -> str:
        s = self.score(visitor_id)
        if s >= 75:
            return "HIGH"
        elif s >= 40:
            return "MEDIUM"
        else:
            return "LOW"

    def is_healthy(self, visitor_id: str) -> bool:
        return self.score(visitor_id) >= 40.0

    def summary(self, visitor_id: str) -> dict:
        rec = self._records.get(visitor_id, HealthRecord())
        return {
            "visitor_id":       visitor_id,
            "health_score":     round(rec.score, 1),
            "trust_level":      self.trust_level(visitor_id),
            "total_frames":     rec.total_frames,
            "reasso_count":     rec.reasso_count,
            "ambiguity_count":  rec.ambiguity_count,
            "teleport_count":   rec.teleport_count,
        }

    def all_scores(self) -> Dict[str, float]:
        return {vid: round(rec.score, 1) for vid, rec in self._records.items()}
