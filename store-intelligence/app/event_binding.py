"""
app/event_binding.py — Maps calibration shape roles to event types and pipeline config.

This module reads the ZoneMapper and provides a single place where
"what role does this zone play?" maps to "what events should fire?".

When calibration changes (new labels, moved polygons), this module
automatically reflects the new binding on next call — no manual edits.

Usage:
    from event_binding import EventBinding
    eb = EventBinding()

    eb.get_billing_zone_id()     → "BILLING_COUNTER"
    eb.get_queue_zone_id()       → "QUEUE_AREA"
    eb.get_entry_config()        → {"p1":..., "p2":..., "inside_is":...}
    eb.is_billing_zone(zone_id)  → True/False
    eb.is_queue_zone(zone_id)    → True/False
    eb.is_staff_zone(zone_id)    → True/False
    eb.get_sku_zone(zone_id)     → "SKINCARE" | None
    eb.role_to_event_hints(role) → list of event type strings
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Add pipeline to path for zone_mapper import
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE = os.path.join(os.path.dirname(_HERE), "pipeline")
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)


# ---------------------------------------------------------------------------
# Role → event type mapping
# ---------------------------------------------------------------------------

ROLE_EVENT_MAP: Dict[str, List[str]] = {
    "zone":             ["ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL"],
    "entry_line":       ["ENTRY", "EXIT", "REENTRY"],
    "inside_region":    ["ENTRY", "REENTRY"],
    "outside_region":   ["EXIT"],
    "billing_counter":  ["ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL"],
    "queue_area":       ["BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "ZONE_DWELL"],
    "staff_area":       [],  # staff area does not drive customer events
}


class EventBinding:
    """
    Reads ZoneMapper and exposes event-relevant configuration.

    Hot-reload aware — delegates to ZoneMapper which checks mtime on each call.
    """

    def __init__(self, store_id: Optional[str] = None) -> None:
        self._store_id = store_id or os.environ.get("STORE_ID", "")
        self._mapper = None
        self._init_mapper()

    def _init_mapper(self) -> None:
        try:
            from zone_mapper import get_mapper  # type: ignore
            self._mapper = get_mapper(self._store_id)
        except Exception as exc:
            logger.warning("EventBinding: could not load ZoneMapper: %s", exc)
            self._mapper = None

    def _m(self):
        if self._mapper is None:
            self._init_mapper()
        return self._mapper

    # ------------------------------------------------------------------
    # Entry / Exit
    # ------------------------------------------------------------------

    def get_entry_config(self, camera_id: str = "CAM_ENTRY_03") -> dict:
        """Return entry line config: {p1, p2, inside_is}."""
        m = self._m()
        if m:
            return m.get_entry_line(camera_id)
        return {"p1": [0.10, 0.50], "p2": [0.90, 0.50], "inside_is": "below"}

    # ------------------------------------------------------------------
    # Billing / Queue
    # ------------------------------------------------------------------

    def get_billing_zone_id(self, camera_id: str = "CAM_BILLING_05") -> Optional[str]:
        """Return shape_id of the billing_counter, or None."""
        m = self._m()
        return m.get_billing_zone_id(camera_id) if m else None

    def get_queue_zone_id(self, camera_id: str = "CAM_BILLING_05") -> Optional[str]:
        """Return shape_id of the queue_area, or None."""
        m = self._m()
        return m.get_queue_zone_id(camera_id) if m else None

    def is_billing_zone(self, zone_id: str, camera_id: str = "CAM_BILLING_05") -> bool:
        return zone_id == self.get_billing_zone_id(camera_id)

    def is_queue_zone(self, zone_id: str, camera_id: str = "CAM_BILLING_05") -> bool:
        return zone_id == self.get_queue_zone_id(camera_id)

    # ------------------------------------------------------------------
    # Zone metadata
    # ------------------------------------------------------------------

    def get_sku_zone(self, zone_id: str, camera_id: str) -> Optional[str]:
        """Return the human label (sku_zone) for a zone_id, or None."""
        m = self._m()
        if not m:
            return None
        for shape in m.get_shapes_for_camera(camera_id):
            if shape.get("shape_id") == zone_id:
                return shape.get("label")
        return None

    def is_staff_zone(self, zone_id: str, camera_id: str) -> bool:
        """Return True if the zone is a staff-only area."""
        m = self._m()
        if not m:
            return False
        for shape in m.get_shapes_for_camera(camera_id):
            if shape.get("shape_id") == zone_id:
                return shape.get("role") == "staff_area"
        return False

    def get_zone_role(self, zone_id: str, camera_id: str) -> Optional[str]:
        """Return the calibration role for a zone_id."""
        m = self._m()
        if not m:
            return None
        for shape in m.get_shapes_for_camera(camera_id):
            if shape.get("shape_id") == zone_id:
                return shape.get("role")
        return None

    def role_to_event_hints(self, role: str) -> List[str]:
        """Return list of event types associated with a calibration role."""
        return ROLE_EVENT_MAP.get(role, [])

    # ------------------------------------------------------------------
    # Bulk queries
    # ------------------------------------------------------------------

    def get_all_zone_ids(self, camera_id: str) -> List[str]:
        """Return all enabled shape_ids for a camera."""
        m = self._m()
        if not m:
            return []
        return [s["shape_id"] for s in m.get_shapes_for_camera(camera_id) if s.get("shape_id")]

    def get_queueing_zone_ids(self) -> frozenset:
        """
        Return a frozenset of zone_ids that trigger queue events.
        Used by config.py QUEUEING_ZONE_IDS equivalent.
        """
        ids = set()
        billing = self.get_billing_zone_id()
        queue   = self.get_queue_zone_id()
        if billing:
            ids.add(billing)
        if queue:
            ids.add(queue)
        # Always include legacy IDs as fallback
        ids.update({"ZONE_BILLING_QUEUE", "ZONE_CASH_COUNTER"})
        return frozenset(ids)

    def get_calibration_summary(self) -> dict:
        """Return a human-readable summary of current calibration wiring."""
        m = self._m()
        if not m:
            return {"error": "ZoneMapper not available"}

        cameras = m.get_all_cameras()
        summary = {
            "store_id": self._store_id,
            "cameras": {}
        }
        for cam in cameras:
            shapes = m.get_shapes_for_camera(cam)
            summary["cameras"][cam] = {
                "shape_count": len(shapes),
                "roles": list({s.get("role") for s in shapes}),
                "zone_ids": [s["shape_id"] for s in shapes],
            }
        summary["entry_config"] = self.get_entry_config()
        summary["billing_zone_id"] = self.get_billing_zone_id()
        summary["queue_zone_id"] = self.get_queue_zone_id()
        return summary


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_binding: Optional[EventBinding] = None


def get_event_binding(store_id: Optional[str] = None) -> EventBinding:
    """Return the module-level EventBinding singleton."""
    global _binding
    if _binding is None:
        resolved = store_id or os.environ.get("STORE_ID", "")
        _binding = EventBinding(resolved)
    return _binding
