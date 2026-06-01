"""
behavior.py — Behavior State Machine for visitor journey tracking.

States (spec-mandated):
    UNKNOWN → ENTERED → BROWSING → DWELLING → QUEUEING → PURCHASE_CANDIDATE → EXITED

Design:
  - Events are generated FROM state transitions, not directly from detections.
  - The state machine runs alongside the event emitter; zone/entry/queue events
    fire as before, but each event also carries the visitor's current behavior_state
    in its metadata (GAP-2 and GAP-3 solution — "run alongside" approach).
  - State is persisted per visitor_id (survives track loss / re-association).

Transitions:
  UNKNOWN      → ENTERED          on first ENTRY event
  ENTERED      → BROWSING         after MIN_BROWSING_SEC in store, or first ZONE_ENTER
  BROWSING     → DWELLING         after BROWSING_MIN_DWELL_MS of continuous zone dwell
  BROWSING     → QUEUEING         on BILLING_QUEUE_JOIN
  DWELLING     → BROWSING         on ZONE_EXIT (left the dwelling zone)
  DWELLING     → QUEUEING         on BILLING_QUEUE_JOIN
  QUEUEING     → PURCHASE_CANDIDATE after extended queue time (> MIN_QUEUE_DWELL_SEC)
  QUEUEING     → BROWSING         on BILLING_QUEUE_ABANDON
  *            → EXITED           on EXIT event
  EXITED       → ENTERED          on REENTRY event (new session)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional

from config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------
class BehaviorState(str, Enum):
    """Visitor behavior state — stored as string for JSON serialisation."""
    UNKNOWN             = "UNKNOWN"
    ENTERED             = "ENTERED"
    BROWSING            = "BROWSING"
    DWELLING            = "DWELLING"
    QUEUEING            = "QUEUEING"
    PURCHASE_CANDIDATE  = "PURCHASE_CANDIDATE"
    EXITED              = "EXITED"


# Event types that drive state transitions
class _ET:
    ENTRY               = "ENTRY"
    EXIT                = "EXIT"
    REENTRY             = "REENTRY"
    ZONE_ENTER          = "ZONE_ENTER"
    ZONE_EXIT           = "ZONE_EXIT"
    ZONE_DWELL          = "ZONE_DWELL"
    BILLING_QUEUE_JOIN  = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"


# ---------------------------------------------------------------------------
# Per-visitor state record
# ---------------------------------------------------------------------------
@dataclass
class _VisitorBehavior:
    state:              BehaviorState   = BehaviorState.UNKNOWN
    state_entered_at:   float           = field(default_factory=time.time)
    zone_entered_at:    Optional[float] = None   # when current zone was entered
    current_zone:       Optional[str]   = None
    queue_joined_at:    Optional[float] = None
    total_dwell_ms:     int             = 0      # cumulative across all zones


# ---------------------------------------------------------------------------
# Behavior State Machine
# ---------------------------------------------------------------------------
class BehaviorStateMachine:
    """
    Tracks the behavioral state for every visitor_id.

    Usage:
        bsm = BehaviorStateMachine()
        state = bsm.update(visitor_id, event_type, zone_id, dwell_ms, wall_time)
        # state is the CURRENT BehaviorState AFTER applying the event.
        # Inject state.value into event metadata["behavior_state"].
    """

    def __init__(self):
        # visitor_id → _VisitorBehavior
        self._states: Dict[str, _VisitorBehavior] = {}

    # ------------------------------------------------------------------
    def get_state(self, visitor_id: str) -> BehaviorState:
        """Return current state without triggering a transition."""
        return self._states.get(visitor_id, _VisitorBehavior()).state

    def update(
        self,
        visitor_id: str,
        event_type: str,
        zone_id:    Optional[str],
        dwell_ms:   int,
        wall_time:  float,
    ) -> BehaviorState:
        """
        Apply an event to the visitor's state machine.
        Returns the new (possibly unchanged) BehaviorState.
        """
        if visitor_id not in self._states:
            self._states[visitor_id] = _VisitorBehavior(
                state_entered_at=wall_time
            )

        vb = self._states[visitor_id]
        prev = vb.state

        vb = self._transition(vb, event_type, zone_id, dwell_ms, wall_time)
        self._states[visitor_id] = vb

        if vb.state != prev:
            logger.debug(
                f"BSM {visitor_id}: {prev.value} → {vb.state.value} "
                f"(event={event_type}, zone={zone_id})"
            )

        return vb.state

    def tick(self, visitor_id: str, wall_time: float) -> BehaviorState:
        """
        Call periodically (every frame) to handle time-based transitions:
          ENTERED → BROWSING after being in store without hitting a zone.
          QUEUEING → PURCHASE_CANDIDATE after sustained queue presence.
        """
        if visitor_id not in self._states:
            return BehaviorState.UNKNOWN

        vb   = self._states[visitor_id]
        prev = vb.state

        # ENTERED → BROWSING if 5 s elapsed without hitting a zone
        if vb.state == BehaviorState.ENTERED:
            elapsed = (wall_time - vb.state_entered_at) * 1000
            if elapsed >= 5_000:
                vb.state          = BehaviorState.BROWSING
                vb.state_entered_at = wall_time

        # QUEUEING → PURCHASE_CANDIDATE if queue dwell long enough
        elif (vb.state == BehaviorState.QUEUEING
              and vb.queue_joined_at is not None):
            queue_ms = (wall_time - vb.queue_joined_at) * 1000
            if queue_ms >= cfg.MIN_QUEUE_DWELL_SEC * 1000 * 2:  # 2× min = candidate
                vb.state          = BehaviorState.PURCHASE_CANDIDATE
                vb.state_entered_at = wall_time

        if vb.state != prev:
            logger.debug(
                f"BSM tick {visitor_id}: {prev.value} → {vb.state.value}"
            )

        self._states[visitor_id] = vb
        return vb.state

    # ------------------------------------------------------------------
    def _transition(
        self,
        vb:         _VisitorBehavior,
        event_type: str,
        zone_id:    Optional[str],
        dwell_ms:   int,
        wall_time:  float,
    ) -> _VisitorBehavior:
        s = vb.state

        # ── ENTRY / REENTRY ───────────────────────────────────────────
        if event_type in (_ET.ENTRY, _ET.REENTRY):
            vb.state           = BehaviorState.ENTERED
            vb.state_entered_at = wall_time
            vb.zone_entered_at = None
            vb.current_zone    = None
            vb.queue_joined_at = None
            return vb

        # ── EXIT ──────────────────────────────────────────────────────
        if event_type == _ET.EXIT:
            vb.state            = BehaviorState.EXITED
            vb.state_entered_at = wall_time
            return vb

        # ── ZONE_ENTER ────────────────────────────────────────────────
        if event_type == _ET.ZONE_ENTER:
            vb.zone_entered_at = wall_time
            vb.current_zone    = zone_id
            if s in (BehaviorState.UNKNOWN, BehaviorState.ENTERED):
                vb.state           = BehaviorState.BROWSING
                vb.state_entered_at = wall_time
            return vb

        # ── ZONE_EXIT ─────────────────────────────────────────────────
        if event_type == _ET.ZONE_EXIT:
            if dwell_ms > 0:
                vb.total_dwell_ms += dwell_ms
            vb.zone_entered_at = None
            vb.current_zone    = None
            if s == BehaviorState.DWELLING:
                vb.state           = BehaviorState.BROWSING
                vb.state_entered_at = wall_time
            return vb

        # ── ZONE_DWELL ────────────────────────────────────────────────
        if event_type == _ET.ZONE_DWELL:
            if dwell_ms > 0:
                vb.total_dwell_ms = max(vb.total_dwell_ms, dwell_ms)
            if s == BehaviorState.BROWSING:
                if dwell_ms >= cfg.BROWSING_MIN_DWELL_MS:
                    vb.state           = BehaviorState.DWELLING
                    vb.state_entered_at = wall_time
            return vb

        # ── BILLING_QUEUE_JOIN ────────────────────────────────────────
        if event_type == _ET.BILLING_QUEUE_JOIN:
            if s not in (BehaviorState.EXITED,):
                vb.state           = BehaviorState.QUEUEING
                vb.state_entered_at = wall_time
                vb.queue_joined_at  = wall_time
            return vb

        # ── BILLING_QUEUE_ABANDON ─────────────────────────────────────
        if event_type == _ET.BILLING_QUEUE_ABANDON:
            if s == BehaviorState.QUEUEING:
                vb.state           = BehaviorState.BROWSING
                vb.state_entered_at = wall_time
                vb.queue_joined_at  = None
            return vb

        return vb

    # ------------------------------------------------------------------
    def summary(self, visitor_id: str) -> dict:
        vb = self._states.get(visitor_id)
        if vb is None:
            return {"visitor_id": visitor_id, "state": "UNKNOWN"}
        return {
            "visitor_id":    visitor_id,
            "state":         vb.state.value,
            "total_dwell_ms": vb.total_dwell_ms,
            "current_zone":  vb.current_zone,
        }

    def all_states(self) -> Dict[str, str]:
        return {vid: vb.state.value for vid, vb in self._states.items()}
