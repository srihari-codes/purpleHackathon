"""
occlusion.py — Occlusion Reasoner.

When a person disappears from a camera frame, the system must decide
WHY they disappeared before choosing a recovery strategy.

Disappearance types and their recovery windows:
  SHELF_OCCLUSION      — brief, same zone, person will reappear soon (20s)
  CROWD_OCCLUSION      — multiple people nearby caused the loss (15s)
  CAMERA_BOUNDARY_EXIT — last bbox near frame edge → handoff expected (8s)
  TRUE_STORE_EXIT      — crossed entry line or detected as exiting (60s for REENTRY)
  UNKNOWN              — default fallback (15s)

Classification uses:
  - bbox position relative to frame boundaries
  - number of nearby active tracks
  - last zone identity
  - whether an entry/exit line crossing was detected

This makes the SUSPENDED window adaptive rather than a fixed 15s for everyone.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

from config import cfg

logger = logging.getLogger(__name__)


class OcclusionType(Enum):
    SHELF_OCCLUSION      = "SHELF_OCCLUSION"
    CROWD_OCCLUSION      = "CROWD_OCCLUSION"
    CAMERA_BOUNDARY_EXIT = "CAMERA_BOUNDARY_EXIT"
    TRUE_STORE_EXIT      = "TRUE_STORE_EXIT"
    UNKNOWN              = "UNKNOWN"


@dataclass
class OcclusionClassification:
    occlusion_type:   OcclusionType
    retain_sec:       float      # how long to keep SUSPENDED
    confidence:       float      # 0–1 confidence in classification
    reason:           str


class OcclusionReasoner:
    """
    Classifies disappearances and returns per-type retention policies.

    Called from VisitorIdentityManager.mark_lost() to tag each
    SUSPENDED passport with its expected recovery window.
    """

    def classify(
        self,
        last_bbox_xyxy:      Tuple[int, int, int, int],
        frame_w:             int,
        frame_h:             int,
        nearby_track_count:  int,
        last_zone:           Optional[str],
        confirmed_exit:      bool = False,
        last_zone_is_billing:bool = False,
    ) -> OcclusionClassification:
        """
        Parameters
        ----------
        last_bbox_xyxy     : final detected bounding box
        frame_w, frame_h   : frame dimensions
        nearby_track_count : number of other active tracks within 100px
        last_zone          : zone_id where visitor was last seen
        confirmed_exit     : True if entry/exit line crossing was detected
        last_zone_is_billing: True if visitor was in billing area
        """

        # ── TRUE_STORE_EXIT (highest priority) ────────────────────────
        if confirmed_exit:
            return OcclusionClassification(
                occlusion_type = OcclusionType.TRUE_STORE_EXIT,
                retain_sec     = cfg.OCCLUSION_EXIT_RETAIN_SEC,
                confidence     = 0.95,
                reason         = "Confirmed entry/exit line crossing detected",
            )

        x1, y1, x2, y2 = last_bbox_xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        margin = cfg.OCCLUSION_BOUNDARY_MARGIN

        # ── CAMERA_BOUNDARY_EXIT ──────────────────────────────────────
        near_left   = cx < frame_w * margin
        near_right  = cx > frame_w * (1 - margin)
        near_top    = cy < frame_h * margin
        near_bottom = cy > frame_h * (1 - margin)
        near_edge   = near_left or near_right or near_top or near_bottom

        if near_edge:
            # Extra signal: if near top/bottom, more likely true edge exit
            confidence = 0.80 if (near_left or near_right) else 0.70
            return OcclusionClassification(
                occlusion_type = OcclusionType.CAMERA_BOUNDARY_EXIT,
                retain_sec     = cfg.OCCLUSION_BOUNDARY_RETAIN_SEC,
                confidence     = confidence,
                reason         = (
                    f"Last bbox centre ({cx:.0f}, {cy:.0f}) near frame edge "
                    f"({frame_w}×{frame_h}, margin={margin})"
                ),
            )

        # ── CROWD_OCCLUSION ───────────────────────────────────────────
        if nearby_track_count >= 3:
            return OcclusionClassification(
                occlusion_type = OcclusionType.CROWD_OCCLUSION,
                retain_sec     = cfg.OCCLUSION_CROWD_RETAIN_SEC,
                confidence     = min(0.90, 0.60 + 0.10 * nearby_track_count),
                reason         = (
                    f"{nearby_track_count} nearby active tracks → likely crowd occlusion"
                ),
            )

        # ── SHELF_OCCLUSION ───────────────────────────────────────────
        # Mid-frame disappearance in a product zone → shelf most likely
        in_product_zone = last_zone is not None and last_zone not in (
            "ZONE_ENTRANCE", "ZONE_BILLING_QUEUE", "ZONE_CASH_COUNTER",
            "ZONE_STAFF_AREA",
        )
        if in_product_zone:
            return OcclusionClassification(
                occlusion_type = OcclusionType.SHELF_OCCLUSION,
                retain_sec     = cfg.OCCLUSION_SHELF_RETAIN_SEC,
                confidence     = 0.65,
                reason         = (
                    f"Disappeared mid-frame in product zone {last_zone} → shelf occlusion"
                ),
            )

        # ── UNKNOWN (fallback) ────────────────────────────────────────
        return OcclusionClassification(
            occlusion_type = OcclusionType.UNKNOWN,
            retain_sec     = cfg.OCCLUSION_UNKNOWN_RETAIN_SEC,
            confidence     = 0.50,
            reason         = "Could not classify disappearance type",
        )

    def confidence_multiplier(
        self,
        occlusion_type: OcclusionType,
        age_sec: float,
        retain_sec: float,
    ) -> float:
        """
        Returns a confidence multiplier [0, 1] that decays with age.
        Different types decay at different rates.
        """
        if retain_sec <= 0:
            return 0.0
        progress = age_sec / retain_sec   # 0 → 1 as age → retain window

        if occlusion_type == OcclusionType.SHELF_OCCLUSION:
            # Slow decay — we're quite confident they'll reappear
            return max(0.0, 1.0 - 0.5 * progress)
        elif occlusion_type == OcclusionType.CROWD_OCCLUSION:
            return max(0.0, 1.0 - 0.7 * progress)
        elif occlusion_type == OcclusionType.CAMERA_BOUNDARY_EXIT:
            # Fast decay — either they showed up elsewhere or they're gone
            return max(0.0, 1.0 - 1.0 * progress)
        elif occlusion_type == OcclusionType.TRUE_STORE_EXIT:
            # Very slow — REENTRY possible up to 60s
            return max(0.0, 1.0 - 0.3 * progress)
        else:
            return max(0.0, 1.0 - 0.7 * progress)
