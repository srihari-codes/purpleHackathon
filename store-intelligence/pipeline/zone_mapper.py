"""
zone_mapper.py — Single source-of-truth geometry loader for the zone calibration system.

Reads from config/calibration/{store_id}.json (new versioned format).
Falls back to zones_override.json then to hardcoded CAMERA_ZONES.

Hot-reload: checks file mtime on every call — no restart needed when
the calibration UI saves a new version.

Public API (drop-in replacements for zones.py functions):
    zm = ZoneMapper()
    zm.get_zones_for_camera(camera_id)       -> List[Zone]
    zm.get_entry_line(camera_id)             -> dict
    zm.get_shapes_by_role(camera_id, role)   -> list[dict]
    zm.get_billing_zone_id(camera_id)        -> str | None
    zm.get_queue_zone_id(camera_id)          -> str | None
    zm.reload()                              -> bool  (True if reloaded)
    zm.get_all_cameras()                     -> list[str]
    zm.get_raw_calibration()                 -> dict
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent
_DEFAULT_STORE = "STORE_BLR_002"
_OVERRIDE_JSON = _HERE / "zones_override.json"

# Calibration JSON search order:
# 1. CALIB_DIR env var (explicit override)
# 2. /data/calibration/  — Docker shared volume (both pipeline + calibrate containers)
# 3. data/calibration/ — relative data directory
# 4. config/calibration/ — local dev / host path
_CALIB_DIR_CANDIDATES = []
if os.environ.get("CALIB_DIR"):
    _CALIB_DIR_CANDIDATES.append(Path(os.environ["CALIB_DIR"]))
_CALIB_DIR_CANDIDATES.extend([
    Path("/data/calibration"),
    _PROJECT_ROOT / "data" / "calibration",
    _PROJECT_ROOT / "config" / "calibration",
])

def _find_calib_dir() -> Path:
    """Return first existing candidate directory, creating it if needed."""
    for c in _CALIB_DIR_CANDIDATES:
        if c.exists():
            return c
    # Default fallback
    default = _PROJECT_ROOT / "data" / "calibration"
    default.mkdir(parents=True, exist_ok=True)
    return default

_CALIB_DIR = _find_calib_dir()



# Role constants
ROLE_ZONE            = "zone"
ROLE_ENTRY_LINE      = "entry_line"
ROLE_INSIDE_REGION   = "inside_region"
ROLE_OUTSIDE_REGION  = "outside_region"
ROLE_BILLING_COUNTER = "billing_counter"
ROLE_QUEUE_AREA      = "queue_area"
ROLE_STAFF_AREA      = "staff_area"

# Camera canonical IDs
CAMERA_IDS = [
    "CAM_FLOOR_01",
    "CAM_FLOOR_02",
    "CAM_ENTRY_03",
    "CAM_GODOWN_04",
    "CAM_BILLING_05",
]


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert #RRGGBB to (B, G, R) for OpenCV."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02X}{g:02X}{b:02X}"


# ---------------------------------------------------------------------------
# Lazy Zone import to avoid circular imports at module level
# ---------------------------------------------------------------------------
def _make_zone(shape: dict, camera_id: str):
    """Build a Zone dataclass from a calibration shape dict."""
    # Import here to avoid top-level circular import
    from zones import Zone  # type: ignore
    color_bgr = _hex_to_bgr(shape.get("color", "#00FF80"))
    points = [tuple(p) for p in shape["points"]]
    label = shape.get("label", shape["shape_id"])
    return Zone(
        zone_id=shape["shape_id"],
        sku_zone=label,
        camera_id=camera_id,
        polygon_norm=points,
        color_bgr=color_bgr,
    )


# ---------------------------------------------------------------------------
# ZoneMapper
# ---------------------------------------------------------------------------
class ZoneMapper:
    """
    Hot-reloadable geometry loader.

    Thread-safe. Checks file mtime on every public call and reloads
    automatically when the calibration JSON is modified by the UI.
    """

    def __init__(self, store_id: str = _DEFAULT_STORE) -> None:
        self._store_id = store_id
        self._calib_path = _CALIB_DIR / f"{store_id}.json"
        self._lock = threading.RLock()

        # Cached state
        self._data: Optional[Dict[str, Any]] = None
        self._mtime: float = 0.0
        self._shapes_by_cam: Dict[str, List[dict]] = {}

        # Initial load
        self._load()

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self) -> bool:
        """Load or reload calibration JSON. Returns True if actually reloaded."""
        try:
            path = self._calib_path
            if not path.exists():
                logger.warning(
                    "Calibration JSON not found at %s — will try zones_override.json",
                    path,
                )
                return self._load_legacy_override()

            mtime = path.stat().st_mtime
            with self._lock:
                if mtime == self._mtime and self._data is not None:
                    return False  # up to date

                with open(path, "r") as f:
                    raw = json.load(f)

                self._data = raw
                self._mtime = mtime
                self._shapes_by_cam = {}
                for cam_id, cam_data in raw.get("cameras", {}).items():
                    shapes = cam_data.get("shapes", [])
                    # Only include enabled shapes
                    self._shapes_by_cam[cam_id] = [
                        s for s in shapes if s.get("enabled", True)
                    ]

                logger.info(
                    "ZoneMapper: loaded calibration store=%s version=%s cameras=%d",
                    raw.get("store_id"), raw.get("version"),
                    len(self._shapes_by_cam),
                )
                return True

        except Exception as exc:
            logger.error("ZoneMapper: failed to load calibration: %s", exc)
            return False

    def _load_legacy_override(self) -> bool:
        """Fall back to zones_override.json if new format not present."""
        if not _OVERRIDE_JSON.exists():
            return False
        try:
            with open(_OVERRIDE_JSON, "r") as f:
                override = json.load(f)

            # Convert legacy format to new shape dicts
            ROLE_FOR_CAM = {
                "CAM_FLOOR_01": ROLE_ZONE,
                "CAM_FLOOR_02": ROLE_ZONE,
                "CAM_ENTRY_03": ROLE_ZONE,
                "CAM_GODOWN_04": ROLE_STAFF_AREA,
                "CAM_BILLING_05": ROLE_ZONE,
            }
            with self._lock:
                self._shapes_by_cam = {}
                for cam_id in CAMERA_IDS:
                    shapes = []
                    for z in override.get(cam_id, []):
                        b, g, r = z.get("color_bgr", [0, 255, 128])
                        shapes.append({
                            "shape_id": z["zone_id"],
                            "shape_type": "polygon",
                            "role": ROLE_FOR_CAM.get(cam_id, ROLE_ZONE),
                            "label": z.get("sku_zone", z["zone_id"]),
                            "points": z["polygon_norm"],
                            "color": f"#{r:02X}{g:02X}{b:02X}",
                            "enabled": True,
                        })
                    # Also handle entry_line_norm
                    if cam_id == "CAM_ENTRY_03":
                        entry_lines = override.get("entry_line_norm", {})
                        if cam_id in entry_lines:
                            val = entry_lines[cam_id]
                            if isinstance(val, list) and len(val) == 2:
                                shapes.append({
                                    "shape_id": "ENTRY_LINE",
                                    "shape_type": "line",
                                    "role": ROLE_ENTRY_LINE,
                                    "label": "ENTRY_DOOR",
                                    "points": val,
                                    "color": "#FF0033",
                                    "enabled": True,
                                    "meta": {"inside_is": "below"},
                                })
                    if shapes:
                        self._shapes_by_cam[cam_id] = shapes
            logger.info("ZoneMapper: loaded legacy zones_override.json")
            return True
        except Exception as exc:
            logger.error("ZoneMapper: failed to load legacy override: %s", exc)
            return False

    def reload(self) -> bool:
        """Force a reload check. Returns True if data was reloaded."""
        return self._load()

    # ------------------------------------------------------------------
    # Hot-reload check (call before every public read)
    # ------------------------------------------------------------------

    def _check_reload(self) -> None:
        """Check if the calibration file changed and reload if needed."""
        try:
            if self._calib_path.exists():
                mtime = self._calib_path.stat().st_mtime
                if mtime != self._mtime:
                    self._load()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_shapes_by_role(self, camera_id: str, role: str) -> List[dict]:
        """Return all enabled shapes for a camera with the given role."""
        self._check_reload()
        with self._lock:
            return [
                s for s in self._shapes_by_cam.get(camera_id, [])
                if s.get("role") == role
            ]

    def get_shapes_for_camera(self, camera_id: str) -> List[dict]:
        """Return all enabled shapes for a camera."""
        self._check_reload()
        with self._lock:
            return list(self._shapes_by_cam.get(camera_id, []))

    def get_zones_for_camera(self, camera_id: str) -> list:
        """
        Return Zone objects for all polygon shapes (roles: zone, billing_counter,
        queue_area, staff_area, inside_region) in a camera.
        Compatible with zones.py Zone dataclass.
        """
        self._check_reload()
        polygon_roles = {
            ROLE_ZONE, ROLE_BILLING_COUNTER, ROLE_QUEUE_AREA,
            ROLE_STAFF_AREA, ROLE_INSIDE_REGION, ROLE_OUTSIDE_REGION,
        }
        zones = []
        with self._lock:
            for shape in self._shapes_by_cam.get(camera_id, []):
                if shape.get("shape_type") != "line" and shape.get("role") in polygon_roles:
                    try:
                        zones.append(_make_zone(shape, camera_id))
                    except Exception as exc:
                        logger.warning("ZoneMapper: skipping shape %s: %s", shape.get("shape_id"), exc)
        return zones

    def get_entry_line(self, camera_id: str) -> dict:
        """
        Return entry line config for a camera.
        Returns dict with keys: p1, p2, inside_is
        """
        self._check_reload()
        with self._lock:
            for shape in self._shapes_by_cam.get(camera_id, []):
                if shape.get("role") == ROLE_ENTRY_LINE and shape.get("shape_type") == "line":
                    pts = shape["points"]
                    meta = shape.get("meta", {})
                    return {
                        "p1": list(pts[0]),
                        "p2": list(pts[1]),
                        "inside_is": meta.get("inside_is", "below"),
                    }
        # Default fallback
        return {"p1": [0.10, 0.50], "p2": [0.90, 0.50], "inside_is": "below"}

    def get_billing_zone_id(self, camera_id: str = "CAM_BILLING_05") -> Optional[str]:
        """Return the shape_id of the billing_counter shape, or None."""
        self._check_reload()
        with self._lock:
            for shape in self._shapes_by_cam.get(camera_id, []):
                if shape.get("role") == ROLE_BILLING_COUNTER:
                    return shape["shape_id"]
        return None

    def get_queue_zone_id(self, camera_id: str = "CAM_BILLING_05") -> Optional[str]:
        """Return the shape_id of the queue_area shape, or None."""
        self._check_reload()
        with self._lock:
            for shape in self._shapes_by_cam.get(camera_id, []):
                if shape.get("role") == ROLE_QUEUE_AREA:
                    return shape["shape_id"]
        return None

    def get_all_cameras(self) -> List[str]:
        """Return list of camera IDs present in the calibration."""
        self._check_reload()
        with self._lock:
            return list(self._shapes_by_cam.keys())

    def get_raw_calibration(self) -> dict:
        """Return the full raw calibration dict."""
        self._check_reload()
        with self._lock:
            return self._data or {}

    def get_store_id(self) -> str:
        return self._store_id

    def get_calib_path(self) -> Path:
        return self._calib_path

    def save(self, data: dict) -> None:
        """
        Save a new calibration dict. Updates mtime to prevent spurious reload.
        Thread-safe.
        """
        _CALIB_DIR.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(self._calib_path, "w") as f:
                json.dump(data, f, indent=2)
            # Force reload of internal cache
            self._mtime = 0.0
        self._load()
        logger.info("ZoneMapper: saved calibration to %s", self._calib_path)


# ---------------------------------------------------------------------------
# Module-level singleton (used by zones.py, entry_exit.py, billing_queue.py)
# ---------------------------------------------------------------------------
_mapper: Optional[ZoneMapper] = None
_mapper_lock = threading.Lock()


def get_mapper(store_id: str = _DEFAULT_STORE) -> ZoneMapper:
    """Return the module-level ZoneMapper singleton."""
    global _mapper
    with _mapper_lock:
        if _mapper is None:
            _mapper = ZoneMapper(store_id)
    return _mapper


def reset_mapper() -> None:
    """Reset the singleton (useful for testing)."""
    global _mapper
    with _mapper_lock:
        _mapper = None
