"""
group.py — Group Awareness Module.

Detects visitors entering together and tracks their spatial proximity over time.
Group context becomes a consensus signal: if one group member is occluded,
nearby group members increase identity confidence for the missing member.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from config import cfg

logger = logging.getLogger(__name__)


class GroupTracker:
    """Tracks groups of visitors."""

    def __init__(self):
        # visitor_id -> wall_time of entry
        self._entries: Dict[str, float] = {}
        
        # visitor_id -> set of group member visitor_ids
        self._groups: Dict[str, Set[str]] = defaultdict(set)
        
        # (vid1, vid2) -> co-located frame count
        self._colocation_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    def on_entry(self, visitor_id: str, wall_time: float):
        """Record entry time to seed group formation."""
        self._entries[visitor_id] = wall_time
        
        # Check if anyone entered recently
        for other_vid, entry_time in self._entries.items():
            if other_vid == visitor_id:
                continue
                
            dt = abs(wall_time - entry_time)
            if dt <= cfg.GROUP_ENTRY_WINDOW_SEC:
                # Potential group member
                self._colocation_counts[tuple(sorted([visitor_id, other_vid]))] += 1
                
    def update(
        self,
        active_locations: Dict[str, Tuple[str, float, float, int, int]], # vid -> (camera_id, cx, cy, w, h)
        wall_time: float
    ):
        """
        Update group tracking based on spatial proximity in the current frame.
        active_locations maps visitor_id to (camera_id, cx, cy, frame_w, frame_h)
        """
        # Group locations by camera
        cam_locs: Dict[str, Dict[str, Tuple[float, float, int, int]]] = defaultdict(dict)
        for vid, (cam, cx, cy, fw, fh) in active_locations.items():
            cam_locs[cam][vid] = (cx, cy, fw, fh)
            
        # Check proximity for visitors on the same camera
        for cam, locs in cam_locs.items():
            vids = list(locs.keys())
            for i in range(len(vids)):
                for j in range(i + 1, len(vids)):
                    v1, v2 = vids[i], vids[j]
                    cx1, cy1, fw1, fh1 = locs[v1]
                    cx2, cy2, fw2, fh2 = locs[v2]
                    
                    # Calculate pixel distance
                    dx = (cx1 - cx2) * fw1
                    dy = (cy1 - cy2) * fh1
                    dist = (dx*dx + dy*dy)**0.5
                    
                    pair = tuple(sorted([v1, v2]))
                    
                    if dist <= cfg.GROUP_PROXIMITY_PX:
                        self._colocation_counts[pair] += 1
                        
                        # Confirm group membership if they stay together
                        if self._colocation_counts[pair] >= cfg.GROUP_MIN_COFRAMES:
                            self._merge_groups(v1, v2)
                    else:
                        # Slight decay if apart
                        if self._colocation_counts[pair] > 0:
                            self._colocation_counts[pair] -= 1
                            
    def _merge_groups(self, v1: str, v2: str):
        """Merge groups for v1 and v2, ensuring size limits."""
        g1 = self._groups[v1]
        g2 = self._groups[v2]
        
        combined = g1.union(g2).union({v1, v2})
        if len(combined) <= cfg.GROUP_MAX_SIZE:
            for v in combined:
                self._groups[v] = combined - {v}

    def get_group(self, visitor_id: str) -> Set[str]:
        """Return the set of visitor_ids in the same group."""
        return self._groups.get(visitor_id, set())

    def group_confidence_boost(self, visitor_id: str, nearby_visitor_ids: Set[str]) -> float:
        """
        Calculate confidence boost if group members are nearby.
        Used by the Consensus Engine.
        """
        group = self.get_group(visitor_id)
        if not group:
            return 0.0
            
        nearby_group_members = group.intersection(nearby_visitor_ids)
        if nearby_group_members:
            # Boost scales with number of nearby group members
            return cfg.GROUP_CONFIDENCE_BOOST * min(len(nearby_group_members), 3)
            
        return 0.0
        
    def summary(self) -> Dict[str, list]:
        """Summary for API/GUI."""
        return {vid: list(group) for vid, group in self._groups.items() if group}
