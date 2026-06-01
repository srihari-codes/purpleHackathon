"""
ghost.py — Ghost Identity Layer.

When a person disappears, they become a Ghost — not "lost".
Ghosts store predicted future state and remain matchable.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class GhostIdentity:
    visitor_id:              str
    last_location:           Tuple[float, float]  # cx, cy normalized
    last_velocity:           Tuple[float, float]  # dx/dt, dy/dt
    last_zone:               Optional[str]
    last_camera:             str
    last_confidence:         float
    predicted_next_location: Tuple[float, float]
    predicted_next_camera:   Optional[str]
    created_at:              float
    ttl_sec:                 float

    def is_expired(self, wall_time: float) -> bool:
        return wall_time - self.created_at > self.ttl_sec


class GhostLayer:
    """
    Maintains active ghosts and attempts resurrections.
    """

    def __init__(self):
        self._ghosts: Dict[str, GhostIdentity] = {}

    def create_ghost(
        self,
        visitor_id: str,
        cx: float, cy: float,
        vx: float, vy: float,
        camera_id: str,
        zone_id: Optional[str],
        confidence: float,
        retain_sec: float,
        wall_time: float
    ) -> GhostIdentity:
        """Create a new ghost when a track is lost."""
        if confidence < cfg.GHOST_MIN_CONFIDENCE:
            # Not confident enough to maintain a ghost
            return None
            
        ghost = GhostIdentity(
            visitor_id=visitor_id,
            last_location=(cx, cy),
            last_velocity=(vx, vy),
            last_zone=zone_id,
            last_camera=camera_id,
            last_confidence=confidence,
            predicted_next_location=(cx, cy),  # Will be updated
            predicted_next_camera=None,        # To be determined by physics engine
            created_at=wall_time,
            ttl_sec=retain_sec or cfg.GHOST_TTL_SEC
        )
        
        # Enforce max count to prevent memory bloat
        if len(self._ghosts) >= cfg.GHOST_MAX_COUNT:
            # Remove oldest
            oldest = min(self._ghosts.values(), key=lambda g: g.created_at)
            del self._ghosts[oldest.visitor_id]
            
        self._ghosts[visitor_id] = ghost
        logger.debug(f"Ghost created for {visitor_id} on {camera_id}")
        return ghost

    def update_predictions(self, wall_time: float):
        """Update predicted positions based on velocity and elapsed time."""
        for ghost in self._ghosts.values():
            dt = wall_time - ghost.created_at
            if dt > 0:
                # Apply velocity decay
                decay = cfg.GHOST_VELOCITY_DECAY ** dt
                vx, vy = ghost.last_velocity
                # Calculate new position based on decayed velocity
                new_x = ghost.last_location[0] + (vx * dt * decay)
                new_y = ghost.last_location[1] + (vy * dt * decay)
                ghost.predicted_next_location = (new_x, new_y)

    def prune_expired(self, wall_time: float):
        """Remove ghosts that have exceeded their TTL."""
        expired = [vid for vid, g in self._ghosts.items() if g.is_expired(wall_time)]
        for vid in expired:
            del self._ghosts[vid]

    def attempt_resurrection(
        self,
        camera_id: str,
        cx: float, cy: float,
        wall_time: float,
        consensus_engine=None,
        new_track_data=None,
        identity_manager=None
    ) -> Optional[str]:
        """
        Check if a new track matches an active ghost.
        This is a pre-filter before normal re-identification.
        Requires the consensus engine to perform the actual evaluation.
        """
        best_match = None
        best_score = cfg.GHOST_RESURRECTION_THRESHOLD

        for visitor_id, ghost in self._ghosts.items():
            if ghost.is_expired(wall_time):
                continue
                
            # Quick spatial filter: if the ghost was on a different camera,
            # wait for the consensus engine to evaluate temporal/spatial plausibility.
            # But if it's the SAME camera, the predicted position should be nearby.
            if ghost.last_camera == camera_id:
                px, py = ghost.predicted_next_location
                dist = ((cx - px)**2 + (cy - py)**2)**0.5
                if dist > 0.3:  # Too far from predicted location
                    continue
            
            # If we pass spatial filter, use consensus engine
            if consensus_engine and identity_manager and new_track_data:
                passport = identity_manager.get_passport(visitor_id)
                if passport:
                    decision = consensus_engine.evaluate(passport, new_track_data, wall_time)
                    if decision.action == "ASSOCIATE" and decision.identity_score > best_score:
                        best_match = visitor_id
                        best_score = decision.identity_score

        return best_match

    def remove_ghost(self, visitor_id: str):
        if visitor_id in self._ghosts:
            del self._ghosts[visitor_id]

    def get_all(self) -> List[GhostIdentity]:
        return list(self._ghosts.values())
