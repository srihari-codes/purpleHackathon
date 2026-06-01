"""
entry_exit.py — Line-crossing entry/exit logic for Camera 3 (entrance).

Logic:
  - A virtual horizontal line is defined at the doorway threshold.
  - We track each track's centroid y-position across frames.
  - When centroid crosses the line inbound → ENTRY
  - When centroid crosses the line outbound → EXIT
  - Brief disappearance inside the frame is NOT an exit (requires actual crossing).

The line position is configurable; defaults match zones.py ENTRY_LINE_NORM.
"""

import logging
import time
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Which direction means "entering"?
# For Camera 3 mounted at entrance, looking INTO the store:
#   - Person walking IN: moves from top-of-frame toward bottom (y increases)
#     OR from bottom toward top, depending on camera mounting.
# We default to: y > line_y = inside store.
# Override via config if your footage is different.
# ---------------------------------------------------------------------------


class EntryExitDetector:
    """
    Per-camera entry/exit detector using a virtual line segment crossing.
    Supports diagonal and arbitrary 2-point line segments.

    Args:
        camera_id:      which camera this instance handles
        p1:             [x1, y1] normalized start coordinate of the line segment
        p2:             [x2, y2] normalized end coordinate of the line segment
        inside_is:      "below" or "above"
        min_cross_frames: how many consecutive frames on new side before
                          we accept a crossing (debounce)
    """

    def __init__(
        self,
        camera_id:        str   = "CAM_ENTRY_03",
        p1:               list  = None,
        p2:               list  = None,
        line_y_norm:      float = 0.50,
        line_x_start:     float = 0.10,
        line_x_end:       float = 0.90,
        inside_is:        str   = "below",
        min_cross_frames: int   = 3,
    ):
        self.camera_id        = camera_id
        self.p1               = p1 if p1 is not None else [line_x_start, line_y_norm]
        self.p2               = p2 if p2 is not None else [line_x_end, line_y_norm]
        self.inside_is        = inside_is   # "below" or "above"
        self.min_cross_frames = min_cross_frames

        # track_id → deque of last N side-of-line values
        self._track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        # track_id → last committed side
        self._committed_side: Dict[int, str] = {}
        # track_id → consecutive frames on new side
        self._cross_counter:  Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------

    def _side(self, cy_norm: float, cx_norm: float) -> Optional[str]:
        """
        Return "inside", "outside", or None if centroid is outside the line segment span.
        """
        x1, y1 = self.p1
        x2, y2 = self.p2
        dx = x2 - x1
        dy = y2 - y1
        len_sq = dx*dx + dy*dy
        if len_sq == 0:
            return None

        # Project point onto the line segment: parameter t
        t = ((cx_norm - x1)*dx + (cy_norm - y1)*dy) / len_sq
        # Add a 5% buffer on the ends for tracking robustness
        if not (-0.05 <= t <= 1.05):
            return None

        # Signed 2D cross product of vector P1P2 and P1P
        d = dx * (cy_norm - y1) - dy * (cx_norm - x1)

        if self.inside_is == "below":
            return "inside" if d > 0 else "outside"
        else:
            return "inside" if d < 0 else "outside"

    def update(
        self,
        track_id: int,
        bbox_xyxy: Tuple[float, float, float, float],
        frame_w:   int,
        frame_h:   int,
    ) -> Optional[str]:
        """
        Update with a new detection for track_id.
        Returns:
          "ENTRY"  — person just crossed inbound
          "EXIT"   — person just crossed outbound
          None     — no crossing event
        """
        x1, y1, x2, y2 = bbox_xyxy
        cx = (x1 + x2) / 2 / frame_w
        cy = (y1 + y2) / 2 / frame_h

        current_side = self._side(cy, cx)
        if current_side is None:
            return None

        committed = self._committed_side.get(track_id)

        if committed is None:
            # First sighting — just record, no event
            self._committed_side[track_id] = current_side
            self._cross_counter[track_id] = 0
            return None

        if current_side == committed:
            # Still on same side — reset counter
            self._cross_counter[track_id] = 0
            return None

        # On opposite side — increment counter
        self._cross_counter[track_id] += 1
        if self._cross_counter[track_id] < self.min_cross_frames:
            return None

        # Crossing confirmed
        event = None
        if committed == "outside" and current_side == "inside":
            event = "ENTRY"
        elif committed == "inside" and current_side == "outside":
            event = "EXIT"

        self._committed_side[track_id] = current_side
        self._cross_counter[track_id]  = 0
        return event

    def remove_track(self, track_id: int):
        self._committed_side.pop(track_id, None)
        self._cross_counter.pop(track_id, None)
        self._track_history.pop(track_id, None)

    def get_line_pixels(self, frame_w: int, frame_h: int):
        """Return (x1, y1, x2, y2) pixel coords of the line for drawing."""
        x1 = int(self.p1[0] * frame_w)
        y1 = int(self.p1[1] * frame_h)
        x2 = int(self.p2[0] * frame_w)
        y2 = int(self.p2[1] * frame_h)
        return x1, y1, x2, y2
