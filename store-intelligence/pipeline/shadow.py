"""
shadow.py — Shadow Tracking Layer.

Invisible continuation of tracks during occlusion.
Shadow tracks move according to physics even when the person is not visible.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class ShadowTrack:
    visitor_id:        str
    predicted_bbox:    Tuple[float, float, float, float]  # x1, y1, x2, y2 normalized
    velocity_x:        float
    velocity_y:        float
    last_zone:         Optional[str]
    camera_id:         str
    confidence:        float
    frames_shadowed:   int = 0

    @property
    def is_expired(self) -> bool:
        return self.frames_shadowed > cfg.SHADOW_MAX_FRAMES or self.confidence < 0.1


class ShadowTracker:
    """
    Maintains physical shadow tracks on a per-camera basis.
    """

    def __init__(self):
        # camera_id -> visitor_id -> ShadowTrack
        self._shadows: Dict[str, Dict[str, ShadowTrack]] = {}

    def create_shadow(
        self,
        visitor_id: str,
        last_bbox: Tuple[float, float, float, float],
        vx: float, vy: float,
        camera_id: str,
        zone_id: Optional[str]
    ) -> Optional[ShadowTrack]:
        """Create a shadow track when a track is lost."""
        if camera_id not in self._shadows:
            self._shadows[camera_id] = {}
            
        # Don't create if velocity is zero/unknown
        if abs(vx) < 0.001 and abs(vy) < 0.001:
            return None
            
        shadow = ShadowTrack(
            visitor_id=visitor_id,
            predicted_bbox=last_bbox,
            velocity_x=vx,
            velocity_y=vy,
            last_zone=zone_id,
            camera_id=camera_id,
            confidence=1.0,
            frames_shadowed=0
        )
        self._shadows[camera_id][visitor_id] = shadow
        return shadow

    def tick(self, wall_time: float):
        """Advance all shadows by one step using velocity."""
        for cam_id, cam_shadows in list(self._shadows.items()):
            expired = []
            for vid, shadow in cam_shadows.items():
                shadow.frames_shadowed += 1
                
                # Apply velocity to bounding box
                x1, y1, x2, y2 = shadow.predicted_bbox
                x1 += shadow.velocity_x
                x2 += shadow.velocity_x
                y1 += shadow.velocity_y
                y2 += shadow.velocity_y
                
                # Clamp to frame [0, 1]
                x1 = max(0.0, min(1.0, x1))
                x2 = max(0.0, min(1.0, x2))
                y1 = max(0.0, min(1.0, y1))
                y2 = max(0.0, min(1.0, y2))
                shadow.predicted_bbox = (x1, y1, x2, y2)
                
                # Decay velocity
                shadow.velocity_x *= cfg.SHADOW_VELOCITY_DAMPING
                shadow.velocity_y *= cfg.SHADOW_VELOCITY_DAMPING
                
                # Decay confidence
                shadow.confidence *= cfg.SHADOW_CONFIDENCE_DECAY
                
                if shadow.is_expired:
                    expired.append(vid)
                    
            for vid in expired:
                del cam_shadows[vid]

    def match_shadow(
        self,
        new_bbox: Tuple[float, float, float, float],
        camera_id: str,
        frame_w: int,
        frame_h: int
    ) -> Optional[Tuple[str, float]]:
        """
        Check if a new detection matches any active shadow on this camera.
        Returns (visitor_id, match_confidence)
        """
        if camera_id not in self._shadows or not self._shadows[camera_id]:
            return None
            
        nx1, ny1, nx2, ny2 = new_bbox
        ncx = (nx1 + nx2) / 2
        ncy = (ny1 + ny2) / 2
        
        best_match = None
        best_conf = 0.0
        best_dist = float('inf')
        
        for vid, shadow in self._shadows[camera_id].items():
            sx1, sy1, sx2, sy2 = shadow.predicted_bbox
            scx = (sx1 + sx2) / 2
            scy = (sy1 + sy2) / 2
            
            # Pixel distance
            dx = (ncx - scx) * frame_w
            dy = (ncy - scy) * frame_h
            dist = (dx*dx + dy*dy)**0.5
            
            if dist < cfg.SHADOW_MATCH_DISTANCE_PX:
                # Closer match = higher confidence
                dist_conf = max(0.0, 1.0 - (dist / cfg.SHADOW_MATCH_DISTANCE_PX))
                # Combined with shadow's internal confidence
                final_conf = dist_conf * shadow.confidence
                
                if final_conf > cfg.SHADOW_MATCH_MIN_CONF and final_conf > best_conf:
                    best_conf = final_conf
                    best_match = vid
                    best_dist = dist
                    
        if best_match:
            logger.debug(f"Shadow match: {best_match} at dist {best_dist:.1f}px (conf {best_conf:.2f})")
            return best_match, best_conf
            
        return None

    def active_shadows(self, camera_id: str) -> List[ShadowTrack]:
        """Return active shadows for rendering on GUI."""
        if camera_id in self._shadows:
            return list(self._shadows[camera_id].values())
        return []
        
    def remove_shadow(self, visitor_id: str, camera_id: str):
        if camera_id in self._shadows and visitor_id in self._shadows[camera_id]:
            del self._shadows[camera_id][visitor_id]
