"""
store_graph.py — Digital Twin of the store.

Maintains a directed weighted graph of zone and camera transitions.
Used by the ConsensusIdentityEngine to score zone/camera plausibility.

Nodes: logical store zones (matches zone_ids in zones.py)
Edges: (from_zone, to_zone) → transition_probability [0.0–1.0]

Camera transition probability accounts for:
  - Physical adjacency between cameras
  - Expected travel time between camera coverage areas
  - Whether a customer would plausibly take that path
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

from config import cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Zone node definitions (logical store areas)
# ---------------------------------------------------------------------------
STORE_ZONES = {
    "ENTRANCE",
    "ZONE_EB", "ZONE_TFS", "ZONE_FRAGRANCE", "ZONE_NAIL", "ZONE_FOH",
    "ZONE_FACES", "ZONE_SWISS_PLUS",
    "ZONE_MINIMALIST", "ZONE_AQUALOGICA", "ZONE_PILGRIM", "ZONE_DK",
    "ZONE_MAKEUP", "ZONE_MARS_NYBAE", "ZONE_LOREAL", "ZONE_BEAUTY",
    "ZONE_BILLING_QUEUE", "ZONE_CASH_COUNTER", "ZONE_ACCESSORIES",
    "ZONE_STAFF_AREA", "ZONE_ENTRANCE",
    "EXIT",
}

# ---------------------------------------------------------------------------
# Zone transition probability matrix
# (from_zone, to_zone) → probability [0.0–1.0]
# Missing pairs default to LOW_PROB (unlikely but not impossible)
# ---------------------------------------------------------------------------
_LOW   = 0.10
_MED   = 0.40
_HIGH  = 0.75
_VHIGH = 0.90

ZONE_TRANSITIONS: Dict[Tuple[str, str], float] = {
    # Entrance → main floor
    ("ZONE_ENTRANCE", "ZONE_EB"):          _VHIGH,
    ("ZONE_ENTRANCE", "ZONE_TFS"):         _HIGH,
    ("ZONE_ENTRANCE", "ZONE_FRAGRANCE"):   _HIGH,
    ("ZONE_ENTRANCE", "ZONE_MINIMALIST"):  _MED,
    ("ZONE_ENTRANCE", "ZONE_BILLING_QUEUE"): _LOW,
    ("ZONE_ENTRANCE", "ZONE_STAFF_AREA"):  _LOW,

    # Left-side shelf flow
    ("ZONE_EB",        "ZONE_TFS"):        _VHIGH,
    ("ZONE_EB",        "ZONE_FRAGRANCE"):  _HIGH,
    ("ZONE_TFS",       "ZONE_FRAGRANCE"):  _HIGH,
    ("ZONE_TFS",       "ZONE_NAIL"):       _MED,
    ("ZONE_FRAGRANCE", "ZONE_NAIL"):       _HIGH,
    ("ZONE_FRAGRANCE", "ZONE_FOH"):        _MED,
    ("ZONE_NAIL",      "ZONE_FOH"):        _HIGH,
    ("ZONE_NAIL",      "ZONE_FACES"):      _MED,
    ("ZONE_FOH",       "ZONE_MAKEUP"):     _HIGH,
    ("ZONE_FOH",       "ZONE_FACES"):      _MED,
    ("ZONE_FACES",     "ZONE_SWISS_PLUS"): _HIGH,
    ("ZONE_SWISS_PLUS","ZONE_MARS_NYBAE"): _MED,

    # Right-side shelf flow
    ("ZONE_MINIMALIST","ZONE_AQUALOGICA"): _VHIGH,
    ("ZONE_AQUALOGICA","ZONE_PILGRIM"):    _VHIGH,
    ("ZONE_PILGRIM",   "ZONE_DK"):         _VHIGH,
    ("ZONE_DK",        "ZONE_MAKEUP"):     _HIGH,
    ("ZONE_MAKEUP",    "ZONE_MARS_NYBAE"): _HIGH,
    ("ZONE_MARS_NYBAE","ZONE_LOREAL"):     _HIGH,
    ("ZONE_LOREAL",    "ZONE_BEAUTY"):     _HIGH,
    ("ZONE_BEAUTY",    "ZONE_BILLING_QUEUE"): _MED,

    # Billing area
    ("ZONE_BILLING_QUEUE","ZONE_CASH_COUNTER"): _HIGH,
    ("ZONE_CASH_COUNTER", "ZONE_BILLING_QUEUE"): _MED,
    ("ZONE_BILLING_QUEUE","ZONE_ACCESSORIES"):   _MED,
    ("ZONE_BILLING_QUEUE","ZONE_ENTRANCE"):      _HIGH,  # exit after billing
    ("ZONE_CASH_COUNTER", "ZONE_ENTRANCE"):      _HIGH,

    # Cross-floor movement (moderate — customers browse both sides)
    ("ZONE_FOH",       "ZONE_MAKEUP"):     _HIGH,
    ("ZONE_MAKEUP",    "ZONE_FOH"):        _HIGH,
    ("ZONE_NAIL",      "ZONE_MINIMALIST"): _MED,
    ("ZONE_TFS",       "ZONE_MINIMALIST"): _MED,

    # Staff-specific (high staff, very low customer probability)
    ("ZONE_STAFF_AREA","ZONE_BILLING_QUEUE"): 0.60,  # staff to billing is ok
    ("ZONE_BILLING_QUEUE","ZONE_STAFF_AREA"): 0.20,  # staff goes back
    ("ZONE_STAFF_AREA","ZONE_ENTRANCE"):     0.50,

    # Reverse entrance flow (exit)
    ("ZONE_FACES",     "ZONE_ENTRANCE"):   _MED,
    ("ZONE_FOH",       "ZONE_ENTRANCE"):   _MED,
    ("ZONE_EB",        "ZONE_ENTRANCE"):   _HIGH,
}

# ---------------------------------------------------------------------------
# Camera topology — camera-to-camera transition probability matrix
# Format: (cam_from, cam_to) → (base_prob, expected_transit_sec, tolerance_sec)
# ---------------------------------------------------------------------------
CAMERA_TRANSITIONS: Dict[Tuple[str, str], Tuple[float, float, float]] = {
    # Camera 1 ↔ Camera 2 (overlapping floor coverage)
    ("CAM_FLOOR_01", "CAM_FLOOR_02"): (0.80, 4.0, 5.0),
    ("CAM_FLOOR_02", "CAM_FLOOR_01"): (0.80, 4.0, 5.0),
    # Camera 3 (entrance) ↔ Camera 1 (main floor entry side)
    ("CAM_ENTRY_03", "CAM_FLOOR_01"): (0.85, 3.0, 4.0),
    ("CAM_FLOOR_01", "CAM_ENTRY_03"): (0.70, 3.0, 4.0),
    # Camera 2 ↔ Camera 5 (floor → billing)
    ("CAM_FLOOR_02", "CAM_BILLING_05"): (0.65, 4.0, 5.0),
    ("CAM_BILLING_05","CAM_FLOOR_02"):  (0.65, 4.0, 5.0),
    # Camera 2 ↔ Camera 4 (staff movement)
    ("CAM_FLOOR_02", "CAM_GODOWN_04"): (0.30, 6.0, 8.0),
    ("CAM_GODOWN_04","CAM_FLOOR_02"):  (0.30, 6.0, 8.0),
    # Low-probability long jumps (still possible)
    ("CAM_FLOOR_01", "CAM_BILLING_05"): (0.20, 8.0, 10.0),
    ("CAM_ENTRY_03", "CAM_BILLING_05"): (0.15, 10.0, 12.0),
}

# Camera "distance" in hops — used for simultaneous-presence anomaly detection
CAMERA_HOP_DISTANCE: Dict[Tuple[str, str], int] = {
    ("CAM_FLOOR_01", "CAM_FLOOR_02"): 1,
    ("CAM_FLOOR_02", "CAM_FLOOR_01"): 1,
    ("CAM_ENTRY_03", "CAM_FLOOR_01"): 1,
    ("CAM_FLOOR_01", "CAM_ENTRY_03"): 1,
    ("CAM_FLOOR_02", "CAM_BILLING_05"): 1,
    ("CAM_BILLING_05", "CAM_FLOOR_02"): 1,
    ("CAM_FLOOR_02", "CAM_GODOWN_04"): 1,
    ("CAM_GODOWN_04", "CAM_FLOOR_02"): 1,
    ("CAM_ENTRY_03", "CAM_FLOOR_02"): 2,
    ("CAM_FLOOR_02", "CAM_ENTRY_03"): 2,
    ("CAM_FLOOR_01", "CAM_BILLING_05"): 2,
    ("CAM_BILLING_05", "CAM_FLOOR_01"): 2,
    ("CAM_ENTRY_03", "CAM_BILLING_05"): 3,
    ("CAM_BILLING_05", "CAM_ENTRY_03"): 3,
    ("CAM_FLOOR_01", "CAM_GODOWN_04"): 2,
    ("CAM_GODOWN_04", "CAM_FLOOR_01"): 2,
}


class StoreGraph:
    """
    Digital twin of the store's spatial layout.

    Provides plausibility scores for zone and camera transitions,
    used as signals in the ConsensusIdentityEngine.
    """

    def transition_probability(
        self, zone_from: Optional[str], zone_to: Optional[str]
    ) -> float:
        """
        Probability that a visitor moves from zone_from to zone_to.
        Returns 0.5 (neutral) if either zone is None (unknown).
        """
        if zone_from is None or zone_to is None:
            return 0.5
        if zone_from == zone_to:
            return 1.0   # staying in same zone is always plausible
        # Check both orderings (undirected fallback)
        p = ZONE_TRANSITIONS.get((zone_from, zone_to))
        if p is not None:
            return p
        p = ZONE_TRANSITIONS.get((zone_to, zone_from))
        if p is not None:
            return p * 0.8   # reverse direction slightly less likely
        return _LOW   # unknown pair — unlikely but not impossible

    def camera_transition_probability(
        self,
        cam_from: Optional[str],
        cam_to: Optional[str],
        elapsed_sec: float,
    ) -> float:
        """
        Probability score for a person moving from cam_from to cam_to
        in elapsed_sec seconds. Peaks at expected_transit_sec.
        Returns 0.5 (neutral) if either camera is None.
        """
        if cam_from is None or cam_to is None:
            return 0.5
        if cam_from == cam_to:
            # Same camera re-appearance: plausible for short gaps
            if elapsed_sec < 5.0:
                return 0.90
            return max(0.40, 1.0 - elapsed_sec / 30.0)

        entry = CAMERA_TRANSITIONS.get((cam_from, cam_to))
        if entry is None:
            return 0.10  # unknown transition — very unlikely

        base_prob, expected_sec, tolerance_sec = entry
        # Gaussian-like score centred on expected transit time
        timing_error = abs(elapsed_sec - expected_sec)
        if timing_error <= tolerance_sec:
            timing_score = math.exp(-(timing_error ** 2) / (2 * tolerance_sec ** 2))
        else:
            timing_score = 0.05   # far off expected timing

        return round(base_prob * timing_score, 4)

    def is_transition_physically_possible(
        self,
        zone_from: Optional[str],
        zone_to: Optional[str],
        elapsed_sec: float,
    ) -> bool:
        """
        Hard gate: returns False if the transition is physically impossible
        (e.g., teleporting from Entrance to Billing in 0.5s).
        """
        if zone_from is None or zone_to is None or zone_from == zone_to:
            return True
        p = self.transition_probability(zone_from, zone_to)
        if p < 0.05:
            return False   # structurally impossible
        # Speed check: every store traversal takes at least ~2s
        if elapsed_sec < 1.5 and zone_from != zone_to:
            return False
        return True

    def camera_hop_distance(self, cam_from: str, cam_to: str) -> int:
        """Number of camera hops between two cameras (for anomaly detection)."""
        if cam_from == cam_to:
            return 0
        return CAMERA_HOP_DISTANCE.get((cam_from, cam_to), 99)

    def plausible_next_cameras(
        self, cam_id: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Return (camera_id, base_prob) sorted by probability, for handoff prediction."""
        results = []
        for (c_from, c_to), (base_prob, _, _) in CAMERA_TRANSITIONS.items():
            if c_from == cam_id:
                results.append((c_to, base_prob))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
