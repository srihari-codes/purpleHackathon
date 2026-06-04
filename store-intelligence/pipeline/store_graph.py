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
from typing import Dict, List, Optional, Set, Tuple

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
# Camera topology — populated at runtime by the wizard via set_camera_topology().
# Format: (cam_from, cam_to) → (base_prob, expected_transit_sec, tolerance_sec)
# ---------------------------------------------------------------------------
CAMERA_TRANSITIONS: Dict[Tuple[str, str], Tuple[float, float, float]] = {}

# Camera "distance" in hops — used for simultaneous-presence anomaly detection.
# Format: (cam_from, cam_to) → hop_count
CAMERA_HOP_DISTANCE: Dict[Tuple[str, str], int] = {}


def set_camera_topology(
    adjacency: Dict[str, Set[str]],
    role_map: Optional[Dict[str, str]] = None,
    transit_overrides: Optional[Dict[Tuple[str, str], Tuple[float, float, float]]] = None,
) -> None:
    """
    Build CAMERA_TRANSITIONS and CAMERA_HOP_DISTANCE from wizard-supplied topology.

    adjacency       : {camera_id: {neighbour_camera_id, ...}}
    role_map        : {camera_id: role}  — used to set transition probs by role pair
    transit_overrides: explicit (prob, expected_sec, tolerance_sec) per pair
    """
    global CAMERA_TRANSITIONS, CAMERA_HOP_DISTANCE

    # Role-based default transition probabilities
    _ROLE_PAIR_PROBS = {
        ("entry",   "floor"):   (0.85, 3.0, 4.0),
        ("floor",   "entry"):   (0.70, 3.0, 4.0),
        ("floor",   "floor"):   (0.80, 4.0, 5.0),
        ("floor",   "billing"): (0.65, 4.0, 5.0),
        ("billing", "floor"):   (0.65, 4.0, 5.0),
        ("floor",   "godown"):  (0.30, 6.0, 8.0),
        ("godown",  "floor"):   (0.30, 6.0, 8.0),
        ("entry",   "billing"): (0.15, 10.0, 12.0),
        ("billing", "entry"):   (0.15, 10.0, 12.0),
    }
    _DEFAULT_PROB = (0.40, 5.0, 6.0)  # fallback for unknown role pairs

    rmap = role_map or {}
    transitions: Dict[Tuple[str, str], Tuple[float, float, float]] = {}

    for cam, neighbours in adjacency.items():
        for nb in neighbours:
            for pair in [(cam, nb), (nb, cam)]:
                if pair in transitions:
                    continue
                r_from = rmap.get(pair[0], "floor")
                r_to   = rmap.get(pair[1], "floor")
                transitions[pair] = _ROLE_PAIR_PROBS.get(
                    (r_from, r_to), _DEFAULT_PROB
                )

    if transit_overrides:
        transitions.update(transit_overrides)

    # BFS-based hop distances
    all_cams = set(adjacency.keys()) | {nb for nbrs in adjacency.values() for nb in nbrs}
    hops: Dict[Tuple[str, str], int] = {}
    for start in all_cams:
        visited = {start: 0}
        queue = [start]
        while queue:
            cur = queue.pop(0)
            for nb in adjacency.get(cur, set()):
                if nb not in visited:
                    visited[nb] = visited[cur] + 1
                    queue.append(nb)
        for end, dist in visited.items():
            if end != start:
                hops[(start, end)] = dist

    CAMERA_TRANSITIONS = transitions
    CAMERA_HOP_DISTANCE = hops
    logger.info(
        "store_graph: topology loaded — %d transition pairs, %d hop distances",
        len(CAMERA_TRANSITIONS),
        len(CAMERA_HOP_DISTANCE),
    )


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


# ---------------------------------------------------------------------------
# Phase 4 Evolution: Retail Physics Engine
# ---------------------------------------------------------------------------
from dataclasses import dataclass

@dataclass
class PhysicsVerdict:
    is_valid:          bool
    plausibility_score:float
    violation_type:    Optional[str]  # None, "speed", "wall", "impossible"
    explanation:       str


class RetailPhysicsEngine(StoreGraph):
    """
    Validates identity transitions against physical reality.
    Subclasses StoreGraph for backward compatibility.
    """

    def validate_transition(
        self,
        cam_from: Optional[str],
        cam_to: Optional[str],
        zone_from: Optional[str],
        zone_to: Optional[str],
        elapsed_sec: float,
        walking_speed: float = 0.0
    ) -> PhysicsVerdict:
        """
        Evaluate if a transition is physically possible.
        """
        if not cam_from or not cam_to:
            return PhysicsVerdict(True, 0.5, None, "Missing camera context")

        # 1. Hard impossibility checks
        if not self.is_transition_physically_possible(zone_from, zone_to, elapsed_sec):
            return PhysicsVerdict(
                is_valid=False,
                plausibility_score=0.0,
                violation_type="impossible",
                explanation=f"Transition from {zone_from} to {zone_to} in {elapsed_sec:.1f}s is impossible"
            )

        # 2. Wall collision / Path check
        if self.wall_collision_check(zone_from, zone_to):
            return PhysicsVerdict(
                is_valid=False,
                plausibility_score=0.0,
                violation_type="wall",
                explanation=f"Direct path between {zone_from} and {zone_to} crosses a physical barrier"
            )

        # 3. Speed plausibility
        # For simplicity, we use the graph's transition prob mixed with speed expectation
        base_prob = self.camera_transition_probability(cam_from, cam_to, elapsed_sec)
        
        # Estimate physical distance loosely based on hops
        hops = self.camera_hop_distance(cam_from, cam_to)
        est_distance = hops * 4.0  # Approx 4 meters per camera coverage gap
        if zone_from != zone_to:
            est_distance += 2.0
            
        speed_plausibility = self.walking_speed_plausibility(est_distance, elapsed_sec)

        final_score = (base_prob * 0.7) + (speed_plausibility * 0.3)
        
        # Determine violation based on speed
        violation = None
        if speed_plausibility < 0.05 and elapsed_sec > 0:
            actual_speed = est_distance / elapsed_sec
            if actual_speed > cfg.PHYSICS_AVG_WALK_SPEED_MPS * 3:
                violation = "speed"
        
        is_valid = (final_score > 0.1) and (violation is None)
        
        expl = "Plausible" if is_valid else f"Implausible transition (score {final_score:.2f})"
        if violation == "speed":
            expl = "Speed violation: moving too fast for human walk"

        return PhysicsVerdict(
            is_valid=is_valid,
            plausibility_score=round(final_score, 4),
            violation_type=violation,
            explanation=expl
        )

    def walking_speed_plausibility(self, distance_m: float, elapsed_sec: float) -> float:
        """Gaussian plausibility around 1.4 m/s walking speed."""
        if elapsed_sec <= 0:
            return 0.0 if distance_m > 0 else 1.0
            
        speed = distance_m / elapsed_sec
        avg_speed = cfg.PHYSICS_AVG_WALK_SPEED_MPS
        std_dev = cfg.PHYSICS_WALK_SPEED_STD
        
        # Very slow is okay (browsing), very fast is penalized
        if speed <= avg_speed:
            return 1.0
            
        # Gaussian decay for speeds > avg_speed
        variance = std_dev ** 2
        score = math.exp(-((speed - avg_speed) ** 2) / (2 * variance))
        return max(0.01, score)

    def wall_collision_check(self, zone_from: Optional[str], zone_to: Optional[str]) -> bool:
        """
        Check if moving directly between zones crosses a physical barrier.
        Returns True if there is a collision (invalid path).
        """
        if not zone_from or not zone_to or zone_from == zone_to:
            return False
            
        # Hardcoded barriers for the store layout
        barriers = [
            ({"ZONE_EB", "ZONE_TFS", "ZONE_FRAGRANCE"}, {"ZONE_MINIMALIST", "ZONE_AQUALOGICA"}), # Center gondola barrier
            ({"ZONE_STAFF_AREA"}, {"ZONE_FRAGRANCE", "ZONE_NAIL", "ZONE_FACES"}) # Staff area wall
        ]
        
        for group_a, group_b in barriers:
            if (zone_from in group_a and zone_to in group_b) or \
               (zone_to in group_a and zone_from in group_b):
                return True # Barrier hit
                
        return False


# ---------------------------------------------------------------------------
# Phase 4 Evolution: Store Memory Graph
# ---------------------------------------------------------------------------
from collections import defaultdict

@dataclass
class GraphEdge:
    source_type: str
    source_id:   str
    target_type: str
    target_id:   str
    rel_type:    str
    wall_time:   float


class StoreMemoryGraph:
    """
    Long-term session memory as a relationship graph.
    In-memory implementation for high-speed tracking.
    """
    
    def __init__(self):
        # source_id -> list of edges
        self._edges: Dict[str, List[GraphEdge]] = defaultdict(list)
        self._edge_count = 0

    def add_event(self, visitor_id: str, camera_id: str, zone_id: Optional[str], event_type: str, wall_time: float):
        """Record relationships into the graph."""
        if self._edge_count > cfg.MEMORY_GRAPH_MAX_EDGES:
            # Prevent infinite growth in long-running processes; we would normally prune
            pass 
            
        edges_to_add = []
        
        if event_type in ("ENTRY", "REENTRY"):
            edges_to_add.append(GraphEdge("Visitor", visitor_id, "Store", "STORE", "ENTERED", wall_time))
        elif event_type == "EXIT":
            edges_to_add.append(GraphEdge("Visitor", visitor_id, "Store", "STORE", "EXITED", wall_time))
            
        if camera_id:
            edges_to_add.append(GraphEdge("Visitor", visitor_id, "Camera", camera_id, "SEEN_ON", wall_time))
            
        if zone_id:
            if event_type == "ZONE_ENTER":
                edges_to_add.append(GraphEdge("Visitor", visitor_id, "Zone", zone_id, "ENTERED", wall_time))
            elif event_type == "ZONE_EXIT":
                edges_to_add.append(GraphEdge("Visitor", visitor_id, "Zone", zone_id, "EXITED", wall_time))
            elif event_type == "ZONE_DWELL":
                edges_to_add.append(GraphEdge("Visitor", visitor_id, "Zone", zone_id, "DWELLED", wall_time))
            elif event_type == "BILLING_QUEUE_JOIN":
                edges_to_add.append(GraphEdge("Visitor", visitor_id, "Zone", zone_id, "QUEUED", wall_time))

        for edge in edges_to_add:
            self._edges[visitor_id].append(edge)
            self._edge_count += 1

    def visitor_journey(self, visitor_id: str) -> List[GraphEdge]:
        """Full journey edges for a visitor."""
        return sorted(self._edges.get(visitor_id, []), key=lambda e: e.wall_time)

    def check_continuity(self, visitor_id: str, expected_camera: str, expected_zone: Optional[str]) -> float:
        """Does this location match the visitor's established pattern?"""
        journey = self.visitor_journey(visitor_id)
        if not journey:
            return 0.5 # Neutral
            
        # Find last known camera and zone
        last_cam = None
        last_zone = None
        for edge in reversed(journey):
            if edge.target_type == "Camera" and not last_cam:
                last_cam = edge.target_id
            elif edge.target_type == "Zone" and not last_zone:
                last_zone = edge.target_id
            if last_cam and last_zone:
                break
                
        # Simple heuristic: if they were just here, high continuity.
        if last_cam == expected_camera and last_zone == expected_zone:
            return 0.9
            
        return 0.5
