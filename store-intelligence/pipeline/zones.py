"""
zones.py — Zone polygon definitions for the store.

Auto-inferred from store_layout_marked.png proportions.
The layout image shows a top-down floor plan. We derive normalised (0-1)
coordinates per zone from the visible layout, then project them into each
camera's field of view via a calibration map.

Coordinate system: (x, y) where (0,0) = top-left of camera frame.
All polygons are lists of (x, y) normalised 0–1 within the camera frame.

You can override any polygon by editing the CAMERA_ZONES dict below or by
supplying a zones_override.json file next to this file.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    zone_id: str
    sku_zone: str          # from store_layout.json vocabulary
    camera_id: str
    polygon_norm: List[Tuple[float, float]]   # normalised coords in camera frame
    color_bgr: Tuple[int, int, int] = (0, 255, 128)

    def pixel_polygon(self, frame_w: int, frame_h: int) -> np.ndarray:
        pts = [(int(x * frame_w), int(y * frame_h)) for x, y in self.polygon_norm]
        return np.array(pts, dtype=np.int32)

    def contains_point(self, px: float, py: float, frame_w: int, frame_h: int) -> bool:
        """px, py are pixel coords in the camera frame."""
        import cv2
        poly = self.pixel_polygon(frame_w, frame_h)
        return cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0


# ---------------------------------------------------------------------------
# Layout inference notes
# ---------------------------------------------------------------------------
# From store_layout_marked.png (top-down plan, ~1400x490 px representation):
#
# Top wall shelves (left→right): EB, TFS, [gap], Minimalist, Aqualogica,
#                                 Pilgrim, D&K
# Left side: Entrance/Exit door (Camera 3)
# Right side: Camera 4 (godown/staff area), Camera 5 (billing counter)
# Centre floor: Fragrance, Nail, F.O.H, Makeup Unit x2
# Bottom wall shelves (left→right): [backlit], Faces, [gap], Swiss+,
#                                    Mars+Nybae, [gap], Lo'real, Beauty
#
# Camera 1 covers LEFT half of floor (top-down left)
# Camera 2 covers RIGHT half of floor (top-down right)
# Camera 3 is at the left wall — entrance, faces right into store
# Camera 5 is at the right wall — billing counter area
#
# For Camera 1 (facing right, covering left half):
#   - Visible zones: EB, TFS top-shelf; Fragrance, Nail, F.O.H centre;
#                    Faces, Swiss+ bottom-shelf
# For Camera 2 (facing left, covering right half):
#   - Visible zones: Minimalist, Aqualogica, Pilgrim, D&K top-shelf;
#                    Makeup Unit centre; Mars+Nybae, Lo'real, Beauty bottom-shelf
# For Camera 5 (billing area):
#   - Billing queue zone, Cash Counter, Accessories
# ---------------------------------------------------------------------------

# CAM_ENTRY_03 — entrance/exit camera
# Polygon covers the doorway threshold for line-crossing.
# The entry line is a horizontal band near the door edge.
ENTRY_LINE_NORM = {
    # (x_start_norm, x_end_norm, y_norm) — horizontal line in camera frame
    # Visitors crossing this y threshold inbound/outbound trigger ENTRY/EXIT
    "CAM_ENTRY_03": {
        "line_y": 0.50,          # mid-frame vertical position of door threshold
        "line_x_start": 0.10,
        "line_x_end": 0.90,
        "inside_is": "below",    # "below" means higher y = inside store
    }
}

# ---------------------------------------------------------------------------
# Zone polygons per camera (normalised 0–1)
# ---------------------------------------------------------------------------

CAMERA_ZONES: Dict[str, List[Zone]] = {

    # -------------------------------------------------------------------------
    # CAM_FLOOR_01 — left half of store
    # Camera is mounted at the right side looking left, or left-top corner
    # looking across. Zones appear roughly:
    #   Top-shelf brands in upper portion of frame
    #   Centre floor fixtures in middle
    #   Bottom-shelf brands in lower portion
    # -------------------------------------------------------------------------
    "CAM_FLOOR_01": [
        Zone("ZONE_EB", "EB",
             "CAM_FLOOR_01",
             [(0.02, 0.02), (0.22, 0.02), (0.22, 0.22), (0.02, 0.22)],
             (0, 200, 100)),
        Zone("ZONE_TFS", "TFS",
             "CAM_FLOOR_01",
             [(0.22, 0.02), (0.42, 0.02), (0.42, 0.22), (0.22, 0.22)],
             (0, 180, 120)),
        Zone("ZONE_FRAGRANCE", "FRAGRANCE",
             "CAM_FLOOR_01",
             [(0.05, 0.35), (0.28, 0.35), (0.28, 0.65), (0.05, 0.65)],
             (180, 100, 0)),
        Zone("ZONE_NAIL", "NAIL",
             "CAM_FLOOR_01",
             [(0.28, 0.35), (0.48, 0.35), (0.48, 0.65), (0.28, 0.65)],
             (160, 80, 20)),
        Zone("ZONE_FOH", "FOH",
             "CAM_FLOOR_01",
             [(0.48, 0.30), (0.80, 0.30), (0.80, 0.70), (0.48, 0.70)],
             (200, 50, 50)),
        Zone("ZONE_FACES", "FACES",
             "CAM_FLOOR_01",
             [(0.02, 0.75), (0.28, 0.75), (0.28, 0.98), (0.02, 0.98)],
             (0, 100, 200)),
        Zone("ZONE_SWISS_PLUS", "SWISS_PLUS",
             "CAM_FLOOR_01",
             [(0.38, 0.75), (0.60, 0.75), (0.60, 0.98), (0.38, 0.98)],
             (0, 80, 220)),
    ],

    # -------------------------------------------------------------------------
    # CAM_FLOOR_02 — right half of store
    # -------------------------------------------------------------------------
    "CAM_FLOOR_02": [
        Zone("ZONE_MINIMALIST", "MINIMALIST",
             "CAM_FLOOR_02",
             [(0.05, 0.02), (0.25, 0.02), (0.25, 0.22), (0.05, 0.22)],
             (100, 200, 0)),
        Zone("ZONE_AQUALOGICA", "AQUALOGICA",
             "CAM_FLOOR_02",
             [(0.25, 0.02), (0.45, 0.02), (0.45, 0.22), (0.25, 0.22)],
             (80, 180, 20)),
        Zone("ZONE_PILGRIM", "PILGRIM",
             "CAM_FLOOR_02",
             [(0.45, 0.02), (0.65, 0.02), (0.65, 0.22), (0.45, 0.22)],
             (60, 160, 40)),
        Zone("ZONE_DK", "D_AND_K",
             "CAM_FLOOR_02",
             [(0.65, 0.02), (0.88, 0.02), (0.88, 0.22), (0.65, 0.22)],
             (40, 140, 60)),
        Zone("ZONE_MAKEUP", "MAKEUP_UNIT",
             "CAM_FLOOR_02",
             [(0.30, 0.30), (0.75, 0.30), (0.75, 0.70), (0.30, 0.70)],
             (200, 0, 150)),
        Zone("ZONE_MARS_NYBAE", "MARS_NYBAE",
             "CAM_FLOOR_02",
             [(0.38, 0.75), (0.58, 0.75), (0.58, 0.98), (0.38, 0.98)],
             (0, 60, 200)),
        Zone("ZONE_LOREAL", "LOREAL",
             "CAM_FLOOR_02",
             [(0.58, 0.75), (0.75, 0.75), (0.75, 0.98), (0.58, 0.98)],
             (0, 40, 220)),
        Zone("ZONE_BEAUTY", "BEAUTY",
             "CAM_FLOOR_02",
             [(0.75, 0.75), (0.95, 0.75), (0.95, 0.98), (0.75, 0.98)],
             (20, 20, 240)),
    ],

    # -------------------------------------------------------------------------
    # CAM_ENTRY_03 — entrance camera, minimal zones (just doorway logic)
    # -------------------------------------------------------------------------
    "CAM_ENTRY_03": [
        Zone("ZONE_ENTRANCE", "ENTRANCE",
             "CAM_ENTRY_03",
             [(0.10, 0.10), (0.90, 0.10), (0.90, 0.90), (0.10, 0.90)],
             (255, 255, 0)),
    ],

    # -------------------------------------------------------------------------
    # CAM_GODOWN_04 — staff / godown, low priority
    # -------------------------------------------------------------------------
    "CAM_GODOWN_04": [
        Zone("ZONE_STAFF_AREA", "STAFF_AREA",
             "CAM_GODOWN_04",
             [(0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)],
             (128, 128, 128)),
    ],

    # -------------------------------------------------------------------------
    # CAM_BILLING_05 — billing counter + queue area
    #
    # Layout (right-wall mounted camera looking left into the billing area):
    #   LEFT side of frame (x < 0.72): customer waiting / queue area
    #   RIGHT side of frame (x > 0.72): cashier's counter
    #     The cashier stands BEHIND the counter at the top of the right region.
    #     Customers approach from the left, not from the right.
    #
    # ZONE_BILLING_QUEUE: full customer-facing area — x from 0.05 to 0.72, all y
    # ZONE_CASH_COUNTER:  STRICT behind-counter space — top-right quadrant only
    #   (the cashier's feet and body appear in the top-right; customers never go there)
    # -------------------------------------------------------------------------
    "CAM_BILLING_05": [
        Zone("ZONE_BILLING_QUEUE", "BILLING_QUEUE",
             "CAM_BILLING_05",
             [(0.05, 0.15), (0.72, 0.15), (0.72, 0.98), (0.05, 0.98)],
             (0, 255, 255)),
        Zone("ZONE_CASH_COUNTER", "CASH_COUNTER",
             "CAM_BILLING_05",
             # Strictly the space BEHIND the counter — upper-right quadrant
             # where the cashier stands. Customers do not enter this region.
             [(0.72, 0.05), (0.98, 0.05), (0.98, 0.55), (0.72, 0.55)],
             (255, 128, 0)),
        Zone("ZONE_ACCESSORIES", "ACCESSORIES",
             "CAM_BILLING_05",
             [(0.05, 0.02), (0.72, 0.02), (0.72, 0.15), (0.05, 0.15)],
             (200, 200, 0)),
    ],
}


def get_zones_for_camera(camera_id: str) -> List[Zone]:
    """Return zone list for a camera. Tries zone_mapper (new JSON) → zones_override.json → hardcoded."""
    # 1. Try new versioned calibration JSON via zone_mapper
    try:
        from zone_mapper import get_mapper  # type: ignore
        zones = get_mapper().get_zones_for_camera(camera_id)
        if zones:
            return zones
    except Exception:
        pass
    # 2. Legacy zones_override.json
    override_path = os.path.join(os.path.dirname(__file__), "zones_override.json")
    if os.path.exists(override_path):
        with open(override_path) as f:
            overrides = json.load(f)
        if camera_id in overrides:
            zones = []
            for z in overrides[camera_id]:
                zones.append(Zone(
                    z["zone_id"], z["sku_zone"], camera_id,
                    [tuple(p) for p in z["polygon_norm"]],
                    tuple(z.get("color_bgr", [0, 255, 128]))
                ))
            return zones
    # 3. Hardcoded fallback (STORE_BLR_002 specific layout — may not match this store)
    fallback = CAMERA_ZONES.get(camera_id)
    if not fallback:
        # Fall back to role-matched template for dynamically named cameras
        cam_upper = camera_id.upper()
        if "ENTRY" in cam_upper:
            fallback = CAMERA_ZONES.get("CAM_ENTRY_03")
        elif "BILLING" in cam_upper:
            fallback = CAMERA_ZONES.get("CAM_BILLING_05")
        elif "GODOWN" in cam_upper or "STAFF" in cam_upper:
            fallback = CAMERA_ZONES.get("CAM_GODOWN_04")
        elif "FLOOR" in cam_upper:
            if "_02" in cam_upper:
                fallback = CAMERA_ZONES.get("CAM_FLOOR_02")
            else:
                fallback = CAMERA_ZONES.get("CAM_FLOOR_01")

    if fallback:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "zones: using hardcoded fallback zones for %s. "
            "These may be wrong for this store. Run calibration to define correct zones.",
            camera_id,
        )
        # Re-bind the fallback zones to the actual camera_id
        bound_fallback = []
        for zone in fallback:
            bound_fallback.append(Zone(
                zone_id=zone.zone_id,
                sku_zone=zone.sku_zone,
                camera_id=camera_id,
                polygon_norm=zone.polygon_norm,
                color_bgr=zone.color_bgr
            ))
        return bound_fallback
    return []



def get_all_zones() -> Dict[str, List[Zone]]:
    return {cam: get_zones_for_camera(cam) for cam in CAMERA_ZONES}


def zone_for_point(px: float, py: float, frame_w: int, frame_h: int,
                   camera_id: str) -> Optional[Zone]:
    """Return first zone containing (px, py) in the given camera frame."""
    for zone in get_zones_for_camera(camera_id):
        if zone.contains_point(px, py, frame_w, frame_h):
            return zone
    return None


def get_entry_line_for_camera(camera_id: str) -> dict:
    """Return entry/exit line config for camera. Tries zone_mapper → zones_override.json → default."""
    # 1. Try new versioned calibration JSON via zone_mapper
    try:
        from zone_mapper import get_mapper  # type: ignore
        cfg = get_mapper().get_entry_line(camera_id)
        if cfg and cfg.get("p1") and cfg["p1"] != [0.10, 0.50]:
            return cfg
    except Exception:
        pass
    override_path = os.path.join(os.path.dirname(__file__), "zones_override.json")
    if os.path.exists(override_path):
        try:
            with open(override_path) as f:
                overrides = json.load(f)
            if "entry_line_norm" in overrides and camera_id in overrides["entry_line_norm"]:
                val = overrides["entry_line_norm"][camera_id]
                if isinstance(val, list) and len(val) == 2:
                    return {
                        "p1": val[0],
                        "p2": val[1],
                        "inside_is": "below"
                    }
                elif isinstance(val, dict):
                    return val
        except Exception:
            pass
            
    # Default to ENTRY_LINE_NORM config or fallback based on camera name substring
    cfg = ENTRY_LINE_NORM.get(camera_id)
    if not cfg:
        if "ENTRY" in camera_id.upper():
            cfg = ENTRY_LINE_NORM.get("CAM_ENTRY_03", {})
        else:
            cfg = {}
            
    return {
        "p1": [cfg.get("line_x_start", 0.10), cfg.get("line_y", 0.50)],
        "p2": [cfg.get("line_x_end", 0.90), cfg.get("line_y", 0.50)],
        "inside_is": cfg.get("inside_is", "below")
    }
