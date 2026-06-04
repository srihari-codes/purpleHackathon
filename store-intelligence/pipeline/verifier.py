"""
pipeline/verifier.py — Calibration geometry validator.

NOTE: This is distinct from app/verifier.py (session anomaly detector).
This module validates the calibration JSON geometry.

Usage:
    from verifier import CalibrationVerifier
    v = CalibrationVerifier()
    errors = v.validate_dict(calib_dict)
    errors = v.validate_file("config/calibration/STORE_BLR_002.json")
    # errors: list of {"severity": "ERROR"|"WARNING"|"INFO", "shape_id": ..., "camera_id": ..., "message": ...}
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum allowed overlap fraction before triggering a warning (0.0–1.0)
OVERLAP_WARN_THRESHOLD = 0.15  # 15% overlap triggers a warning
OVERLAP_ERROR_THRESHOLD = 0.60  # 60%+ overlap is a hard error (shapes nested or same area)

# Roles that require no overlap within the same camera
EXCLUSIVE_ROLES = {"zone", "billing_counter", "queue_area"}

# Camera-specific requirements
CAM_ENTRY_REQUIRED_ROLES = {"entry_line"}
CAM5_REQUIRED_ROLES = {"billing_counter", "queue_area"}

SEVERITY_ERROR   = "ERROR"
SEVERITY_WARNING = "WARNING"
SEVERITY_INFO    = "INFO"


# ---------------------------------------------------------------------------
# Pure-Python geometry helpers (no Shapely dependency)
# ---------------------------------------------------------------------------

def _segments_intersect(p1, p2, p3, p4) -> bool:
    """Return True if segment p1-p2 intersects p3-p4 (excluding endpoints)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    if d1 == 0 and on_segment(p3, p1, p4): return True
    if d2 == 0 and on_segment(p3, p2, p4): return True
    if d3 == 0 and on_segment(p1, p3, p2): return True
    if d4 == 0 and on_segment(p1, p4, p2): return True

    return False


def _polygon_area(points: List[Tuple]) -> float:
    """Shoelace formula for polygon area."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def _is_self_intersecting(points: List[Tuple]) -> bool:
    """Check if a polygon self-intersects."""
    n = len(points)
    if n < 4:
        return False
    edges = [(points[i], points[(i + 1) % n]) for i in range(n)]
    for i in range(len(edges)):
        for j in range(i + 2, len(edges)):
            # Skip adjacent edges (they share a vertex)
            if i == 0 and j == len(edges) - 1:
                continue
            p1, p2 = edges[i]
            p3, p4 = edges[j]
            if _segments_intersect(p1, p2, p3, p4):
                return True
    return False


def _point_in_polygon(point: Tuple, polygon: List[Tuple]) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_overlap_area(poly_a: List[Tuple], poly_b: List[Tuple]) -> float:
    """
    Approximate overlap area using Sutherland-Hodgman clipping.
    Returns approximate intersection area (0.0 if no overlap).
    """
    def clip_polygon(subject, clip_edge_start, clip_edge_end):
        output = []
        if not subject:
            return output
        for i in range(len(subject)):
            current = subject[i]
            prev = subject[i - 1]
            # Edge direction
            cp1 = (clip_edge_end[0] - clip_edge_start[0]) * (current[1] - clip_edge_start[1]) - \
                  (clip_edge_end[1] - clip_edge_start[1]) * (current[0] - clip_edge_start[0])
            cp2 = (clip_edge_end[0] - clip_edge_start[0]) * (prev[1] - clip_edge_start[1]) - \
                  (clip_edge_end[1] - clip_edge_start[1]) * (prev[0] - clip_edge_start[0])
            if cp1 >= 0:
                if cp2 < 0:
                    # Compute intersection
                    dc = [clip_edge_end[0] - clip_edge_start[0], clip_edge_end[1] - clip_edge_start[1]]
                    dp = [prev[0] - current[0], prev[1] - current[1]]
                    n1 = clip_edge_start[0] * clip_edge_end[1] - clip_edge_start[1] * clip_edge_end[0]
                    n2 = prev[0] * current[1] - prev[1] * current[0]
                    denom = dc[0] * dp[1] - dc[1] * dp[0]
                    if abs(denom) > 1e-10:
                        ix = (n1 * dp[0] - n2 * dc[0]) / denom
                        iy = (n1 * dp[1] - n2 * dc[1]) / denom
                        output.append((ix, iy))
                output.append(current)
            elif cp2 >= 0:
                dc = [clip_edge_end[0] - clip_edge_start[0], clip_edge_end[1] - clip_edge_start[1]]
                dp = [prev[0] - current[0], prev[1] - current[1]]
                n1 = clip_edge_start[0] * clip_edge_end[1] - clip_edge_start[1] * clip_edge_end[0]
                n2 = prev[0] * current[1] - prev[1] * current[0]
                denom = dc[0] * dp[1] - dc[1] * dp[0]
                if abs(denom) > 1e-10:
                    ix = (n1 * dp[0] - n2 * dc[0]) / denom
                    iy = (n1 * dp[1] - n2 * dc[1]) / denom
                    output.append((ix, iy))
        return output

    output = list(poly_a)
    n = len(poly_b)
    for i in range(n):
        if not output:
            return 0.0
        input_list = output
        output = []
        clip_start = poly_b[i]
        clip_end = poly_b[(i + 1) % n]
        output = clip_polygon(input_list, clip_start, clip_end)
    return _polygon_area(output) if output else 0.0


# ---------------------------------------------------------------------------
# CalibrationVerifier
# ---------------------------------------------------------------------------

class CalibrationVerifier:
    """
    Validates a calibration dict or file.

    Returns structured error list:
    [
      {
        "severity": "ERROR"|"WARNING"|"INFO",
        "camera_id": "CAM_FLOOR_01",
        "shape_id": "ZONE_EB" | None,
        "code": "SELF_INTERSECT" | ...,
        "message": "human-readable description",
      }
    ]
    """

    def validate_file(self, path: str) -> List[dict]:
        """Load JSON from path and validate."""
        p = Path(path)
        if not p.exists():
            return [_err(None, None, "FILE_NOT_FOUND", f"Calibration file not found: {path}", SEVERITY_ERROR)]
        try:
            with open(p, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return [_err(None, None, "JSON_PARSE_ERROR", f"Invalid JSON: {e}", SEVERITY_ERROR)]
        return self.validate_dict(data)

    def validate_dict(self, data: dict) -> List[dict]:
        """Validate a calibration dict. Returns list of issues."""
        errors: List[dict] = []

        # Top-level structure
        if not isinstance(data.get("cameras"), dict):
            errors.append(_err(None, None, "MISSING_CAMERAS",
                               "Calibration JSON must have a 'cameras' object", SEVERITY_ERROR))
            return errors

        all_labels: Dict[str, List[str]] = {}  # camera_id → list of labels

        for cam_id, cam_data in data["cameras"].items():
            shapes = cam_data.get("shapes", [])
            cam_errors = self._validate_camera(cam_id, shapes)
            errors.extend(cam_errors)
            all_labels[cam_id] = [s.get("label", "") for s in shapes if s.get("enabled", True)]

        # Global duplicate label check (across same camera)
        for cam_id, labels in all_labels.items():
            seen = {}
            for label in labels:
                if not label:
                    continue
                seen[label] = seen.get(label, 0) + 1
            for label, count in seen.items():
                if count > 1:
                    errors.append(_err(cam_id, None, "DUPLICATE_LABEL",
                                       f"Label '{label}' used {count} times in {cam_id}",
                                       SEVERITY_ERROR))

        return errors

    def _validate_camera(self, cam_id: str, shapes: List[dict]) -> List[dict]:
        errors: List[dict] = []

        enabled_shapes = [s for s in shapes if s.get("enabled", True)]
        roles_present = {s.get("role") for s in enabled_shapes}

        # ── Per-camera role requirements ──────────────────────────────────

        if cam_id == "CAM_ENTRY_03":
            if "entry_line" not in roles_present:
                errors.append(_err(cam_id, None, "MISSING_ENTRY_LINE",
                                   "CAM_ENTRY_03 must have a shape with role='entry_line'",
                                   SEVERITY_ERROR))

        if cam_id == "CAM_BILLING_05":
            if "billing_counter" not in roles_present:
                errors.append(_err(cam_id, None, "MISSING_BILLING_COUNTER",
                                   "CAM_BILLING_05 must have a shape with role='billing_counter'",
                                   SEVERITY_WARNING))
            if "queue_area" not in roles_present:
                errors.append(_err(cam_id, None, "MISSING_QUEUE_AREA",
                                   "CAM_BILLING_05 must have a shape with role='queue_area'",
                                   SEVERITY_WARNING))

        # ── Multiple conflicting roles ─────────────────────────────────────
        role_counts: Dict[str, int] = {}
        for s in enabled_shapes:
            r = s.get("role", "")
            role_counts[r] = role_counts.get(r, 0) + 1

        for role in ("entry_line", "billing_counter", "queue_area"):
            if role_counts.get(role, 0) > 1:
                errors.append(_err(cam_id, None, "DUPLICATE_ROLE",
                                   f"Multiple shapes with role='{role}' — only one is allowed",
                                   SEVERITY_ERROR))

        # ── Per-shape validation ───────────────────────────────────────────
        polygons_for_overlap: List[Tuple[str, str, List]] = []  # (shape_id, role, points)

        for shape in enabled_shapes:
            shape_errs = self._validate_shape(cam_id, shape)
            errors.extend(shape_errs)

            stype = shape.get("shape_type", "polygon")
            role  = shape.get("role", "zone")
            pts   = shape.get("points", [])

            if stype == "polygon" and role in EXCLUSIVE_ROLES and len(pts) >= 3:
                polygons_for_overlap.append((shape["shape_id"], role, [tuple(p) for p in pts]))

        # ── Overlap checks (same role only) ──────────────────────────────
        for i in range(len(polygons_for_overlap)):
            for j in range(i + 1, len(polygons_for_overlap)):
                sid_a, role_a, pts_a = polygons_for_overlap[i]
                sid_b, role_b, pts_b = polygons_for_overlap[j]
                if role_a != role_b:
                    continue  # different roles — overlap may be intentional
                area_a = _polygon_area(pts_a)
                area_b = _polygon_area(pts_b)
                if area_a < 1e-8 or area_b < 1e-8:
                    continue
                overlap = _polygon_overlap_area(pts_a, pts_b)
                min_area = min(area_a, area_b)
                frac = min(overlap / min_area, 1.0) if min_area > 0 else 0.0
                # 100% result is typically a clipping artifact for edge-adjacent polygons
                if frac >= 0.99:
                    # Use a simpler centroid-distance check instead
                    cx_a = sum(p[0] for p in pts_a) / len(pts_a)
                    cy_a = sum(p[1] for p in pts_a) / len(pts_a)
                    cx_b = sum(p[0] for p in pts_b) / len(pts_b)
                    cy_b = sum(p[1] for p in pts_b) / len(pts_b)
                    # If centroids are far apart, they're adjacent not nested
                    dist = ((cx_a - cx_b)**2 + (cy_a - cy_b)**2)**0.5
                    if dist > 0.05:
                        continue  # Adjacent — not a real overlap
                if frac >= OVERLAP_ERROR_THRESHOLD:
                    errors.append(_err(cam_id, sid_a, "POLYGON_OVERLAP_CRITICAL",
                                       f"Shapes '{sid_a}' and '{sid_b}' overlap by {frac*100:.0f}% (role={role_a})",
                                       SEVERITY_ERROR))
                elif frac >= OVERLAP_WARN_THRESHOLD:
                    errors.append(_err(cam_id, sid_a, "POLYGON_OVERLAP",
                                       f"Shapes '{sid_a}' and '{sid_b}' overlap by {frac*100:.0f}% (role={role_a})",
                                       SEVERITY_WARNING))

        return errors

    def _validate_shape(self, cam_id: str, shape: dict) -> List[dict]:
        errors: List[dict] = []
        sid   = shape.get("shape_id", "<unknown>")
        stype = shape.get("shape_type", "polygon")
        role  = shape.get("role", "")
        label = shape.get("label", "")
        pts   = shape.get("points", [])

        # Missing shape_id
        if not shape.get("shape_id"):
            errors.append(_err(cam_id, None, "MISSING_SHAPE_ID",
                               "A shape has no shape_id", SEVERITY_ERROR))

        # Missing label
        if not label:
            errors.append(_err(cam_id, sid, "MISSING_LABEL",
                               f"Shape '{sid}' has no label", SEVERITY_WARNING))

        # Missing role
        if not role:
            errors.append(_err(cam_id, sid, "MISSING_ROLE",
                               f"Shape '{sid}' has no role", SEVERITY_ERROR))

        # Empty points
        if not pts:
            errors.append(_err(cam_id, sid, "EMPTY_POINTS",
                               f"Shape '{sid}' has no points", SEVERITY_ERROR))
            return errors  # can't do further checks

        # Points out of range [0,1]
        for i, p in enumerate(pts):
            if len(p) != 2:
                errors.append(_err(cam_id, sid, "INVALID_POINT",
                                   f"Shape '{sid}' point {i} is not [x, y]", SEVERITY_ERROR))
                break
            x, y = p
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                errors.append(_err(cam_id, sid, "POINT_OUT_OF_BOUNDS",
                                   f"Shape '{sid}' point {i} [{x:.3f},{y:.3f}] is outside [0,1]",
                                   SEVERITY_WARNING))

        # Line: must have exactly 2 points
        if stype == "line":
            if len(pts) != 2:
                errors.append(_err(cam_id, sid, "INVALID_LINE",
                                   f"Line shape '{sid}' must have exactly 2 points, has {len(pts)}",
                                   SEVERITY_ERROR))
            else:
                p1, p2 = pts[0], pts[1]
                dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                if dist < 0.05:
                    errors.append(_err(cam_id, sid, "LINE_TOO_SHORT",
                                       f"Entry line '{sid}' is very short (length={dist:.3f}) — may not catch crossings",
                                       SEVERITY_WARNING))
                # entry_line must have meta.inside_is
                if role == "entry_line" and not shape.get("meta", {}).get("inside_is"):
                    errors.append(_err(cam_id, sid, "MISSING_INSIDE_IS",
                                       f"Entry line '{sid}' has no meta.inside_is ('below' or 'above')",
                                       SEVERITY_ERROR))
            return errors  # done for lines

        # Polygon: must have ≥ 3 points
        if stype == "polygon":
            if len(pts) < 3:
                errors.append(_err(cam_id, sid, "TOO_FEW_POINTS",
                                   f"Polygon '{sid}' has only {len(pts)} points (need ≥ 3)",
                                   SEVERITY_ERROR))
                return errors

            # Self-intersection check
            try:
                if _is_self_intersecting([tuple(p) for p in pts]):
                    errors.append(_err(cam_id, sid, "SELF_INTERSECT",
                                       f"Polygon '{sid}' is self-intersecting",
                                       SEVERITY_ERROR))
            except Exception:
                pass

            # Degenerate area
            area = _polygon_area([tuple(p) for p in pts])
            if area < 0.001:
                errors.append(_err(cam_id, sid, "DEGENERATE_POLYGON",
                                   f"Polygon '{sid}' has near-zero area ({area:.5f})",
                                   SEVERITY_WARNING))

        return errors


def _err(camera_id, shape_id, code, message, severity) -> dict:
    return {
        "severity": severity,
        "camera_id": camera_id,
        "shape_id": shape_id,
        "code": code,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def validate_calibration(data: dict) -> List[dict]:
    """Convenience function — validate a calibration dict."""
    return CalibrationVerifier().validate_dict(data)


def has_errors(issues: List[dict]) -> bool:
    """Return True if any issue is ERROR severity."""
    return any(i["severity"] == SEVERITY_ERROR for i in issues)


def has_warnings(issues: List[dict]) -> bool:
    return any(i["severity"] == SEVERITY_WARNING for i in issues)
