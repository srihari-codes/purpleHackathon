"""
queue.py — Billing queue detection for Camera 5.

Logic:
  - Define a billing ROI (from zones.py ZONE_BILLING_QUEUE polygon).
  - Track which visitor_ids are inside the ROI each frame.
  - When a new visitor enters the ROI: emit BILLING_QUEUE_JOIN with
    queue_depth = number of others already waiting.
  - Track dwell time per visitor in the billing zone.
  - If a visitor leaves the ROI and no POS transaction is correlated
    within POS_WINDOW_SEC, flag as candidate for BILLING_QUEUE_ABANDON.
    (Final abandonment decision is POS-correlated downstream; we record
     the signal here.)
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, List, Tuple

logger = logging.getLogger(__name__)

# How long (seconds) a visitor must be in billing zone before they
# are considered "in queue" (filters walk-throughs)
MIN_QUEUE_DWELL_SEC = 5.0

# How long (seconds) to wait after billing zone exit before emitting
# a candidate abandon (POS correlation window)
POS_CORRELATION_WINDOW_SEC = 300.0   # 5 minutes


@dataclass
class BillingVisitor:
    visitor_id:   str
    enter_time:   float       # wall clock
    exit_time:    Optional[float] = None
    is_staff:     bool        = False
    confidence:   float       = 1.0
    session_seq:  int         = 0
    queue_depth_at_join: int  = 0
    abandon_emitted: bool     = False
    join_emitted:    bool     = False


class QueueTracker:
    """
    Tracks visitors in the billing zone.

    Usage:
        qt = QueueTracker()
        # each frame, for camera 5:
        events = qt.update(frame_visitors_in_billing_roi, wall_time)
        # events is a list of dicts with keys: type, visitor_id, queue_depth, dwell_ms
    """

    def __init__(self):
        # visitor_id → BillingVisitor (currently in zone)
        self._in_zone: Dict[str, BillingVisitor] = {}

        # visitor_id → BillingVisitor (recently left zone, awaiting POS)
        self._exited:  Dict[str, BillingVisitor] = {}

        # Set of visitor_ids confirmed purchased (from POS correlation)
        self._purchased: Set[str] = set()

    # ------------------------------------------------------------------

    def update(
        self,
        present_visitors: List[Tuple[str, bool, float, int]],
        # list of (visitor_id, is_staff, confidence, session_seq)
        wall_time: float,
    ) -> List[dict]:
        """
        Call every frame (or every N frames) with the current set of visitors
        detected inside the billing ROI.

        Returns list of pending events:
          {"type": "BILLING_QUEUE_JOIN",   "visitor_id": ..., "queue_depth": ...,
           "confidence": ..., "is_staff": ..., "session_seq": ...}
          {"type": "BILLING_QUEUE_ABANDON","visitor_id": ..., "dwell_ms": ...,
           "confidence": ..., "is_staff": ..., "session_seq": ...}
        """
        events = []
        present_ids = {v[0] for v in present_visitors}

        # --- New arrivals ---
        for visitor_id, is_staff, confidence, session_seq in present_visitors:
            if visitor_id not in self._in_zone:
                # Count existing non-staff visitors already in zone
                queue_depth = sum(
                    1 for vid, bv in self._in_zone.items()
                    if not bv.is_staff
                )
                bv = BillingVisitor(
                    visitor_id=visitor_id,
                    enter_time=wall_time,
                    is_staff=is_staff,
                    confidence=confidence,
                    session_seq=session_seq,
                    queue_depth_at_join=queue_depth,
                )
                self._in_zone[visitor_id] = bv

                # Emit join event (staff excluded from queue metrics but still
                # tracked for completeness — caller filters by is_staff)
                if not is_staff and queue_depth >= 0:
                    bv.join_emitted = True
                    events.append({
                        "type":        "BILLING_QUEUE_JOIN",
                        "visitor_id":  visitor_id,
                        "queue_depth": queue_depth,
                        "confidence":  confidence,
                        "is_staff":    is_staff,
                        "session_seq": session_seq,
                    })
                    logger.debug(f"BILLING_QUEUE_JOIN {visitor_id} depth={queue_depth}")

        # --- Departures ---
        departed = [vid for vid in self._in_zone if vid not in present_ids]
        for visitor_id in departed:
            bv = self._in_zone.pop(visitor_id)
            bv.exit_time = wall_time
            dwell_ms = int((wall_time - bv.enter_time) * 1000)

            if dwell_ms >= int(MIN_QUEUE_DWELL_SEC * 1000) and not bv.is_staff:
                # Move to exited list; real abandon decision comes from POS
                self._exited[visitor_id] = bv
                logger.debug(f"Billing zone exit {visitor_id} dwell={dwell_ms}ms "
                             f"— watching for POS")

        # --- Flush aged abandon candidates ---
        age_out = []
        for visitor_id, bv in self._exited.items():
            if bv.exit_time is None:
                continue
            age = wall_time - bv.exit_time
            if visitor_id in self._purchased:
                # Confirmed purchase — not an abandon
                age_out.append(visitor_id)
            elif age >= POS_CORRELATION_WINDOW_SEC and not bv.abandon_emitted:
                bv.abandon_emitted = True
                dwell_ms = int((bv.exit_time - bv.enter_time) * 1000)
                events.append({
                    "type":        "BILLING_QUEUE_ABANDON",
                    "visitor_id":  visitor_id,
                    "dwell_ms":    dwell_ms,
                    "confidence":  bv.confidence,
                    "is_staff":    bv.is_staff,
                    "session_seq": bv.session_seq,
                })
                logger.debug(f"BILLING_QUEUE_ABANDON {visitor_id} dwell={dwell_ms}ms")
                age_out.append(visitor_id)

        for vid in age_out:
            self._exited.pop(vid, None)

        return events

    def mark_purchased(self, visitor_id: str):
        """Called by POS correlation logic to prevent false abandon."""
        self._purchased.add(visitor_id)
        self._exited.pop(visitor_id, None)

    @property
    def current_queue_depth(self) -> int:
        return sum(1 for bv in self._in_zone.values() if not bv.is_staff)

    @property
    def in_zone_ids(self) -> Set[str]:
        return set(self._in_zone.keys())
