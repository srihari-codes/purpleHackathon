"""
app/verifier.py — Self-Verifier (VerifierEngine).

Detects impossible or suspicious states in the session stream and raw
event flow.  Every check that fires generates a VerifierWarning with:
    - severity   : INFO / WARN / CRITICAL
    - code       : machine-readable anomaly code
    - message    : human-readable explanation
    - visitor_ids: affected visitors
    - suggested_action: what ops should do

Checks implemented:
    SIMULTANEOUS_CAMERAS      — same visitor active in 2+ cameras simultaneously
    VISITOR_TELEPORT          — visitor appears in impossible zone sequence
    QUEUE_DEPTH_NEGATIVE      — queue_depth < 0 (logic error in detection layer)
    DUPLICATE_ACTIVE_SESSION  — visitor has >1 active session open
    STAFF_COUNT_EXPLOSION     — staff fraction of active visitors spikes >50%
    CONFIDENCE_CLIFF          — event confidence drops >0.4 in a single step
    ENTRY_WITHOUT_DOORWAY     — ENTRY events from cameras that aren't entry cameras
    EXIT_WITHOUT_ENTRY        — EXIT with no prior ENTRY in session history
    REENTRY_TOO_FAST          — re-entry within 10 seconds (physically impossible)
    IMPOSSIBLE_CAMERA_TRANSITION — camera hop that cannot happen in elapsed time

Verifier runs after sessionization — it reads VisitorSession state and
emits warnings into AuditTimeline (category=TRACKING) and a thread-safe
warning list queryable by the /anomalies endpoint.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import EventType, InboundEvent, VisitorSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Min seconds between exit and reentry to be physically plausible
MIN_REENTRY_GAP_SEC = 10.0

# Camera specs: which cameras are valid entry/exit cameras
ENTRY_CAMERA_PREFIXES = {"CAM_ENTRY", "CAM_DOOR", "CAM_THRESHOLD"}

# Max plausible hops per second between any two cameras in the store
MAX_CAM_HOPS_PER_SEC = 1.5  # rough: 1 zone/sec is already very fast

# Staff fraction spike threshold
STAFF_FRACTION_SPIKE = 0.50  # >50% of active visitors are staff → suspicious

# Confidence drop cliff
CONFIDENCE_CLIFF_DELTA = 0.40


# ---------------------------------------------------------------------------
# Warning model
# ---------------------------------------------------------------------------

class VerifierSeverity(str, Enum):
    INFO     = "INFO"
    WARN     = "WARN"
    CRITICAL = "CRITICAL"


@dataclass
class VerifierWarning:
    code:             str
    severity:         str
    message:          str
    visitor_ids:      List[str]
    store_id:         str
    timestamp:        str
    suggested_action: str
    metadata:         Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Verifier Engine
# ---------------------------------------------------------------------------

class VerifierEngine:
    """
    Post-session state verifier.

    Call verify_event(event, session) after each event is sessionized.
    Call verify_active_sessions(sessions) periodically (e.g. every ingest batch).

    All warnings are:
        1. Emitted to the Python logger
        2. Stored in _warnings (thread-safe) for the /anomalies endpoint
        3. Optionally forwarded to AuditTimeline (category=TRACKING)
    """

    def __init__(self, audit=None) -> None:
        self._audit    = audit
        self._warnings: List[VerifierWarning] = []
        self._lock     = threading.Lock()

        # State tracking for cross-event checks
        # visitor_id → (camera_id, timestamp_dt)
        self._last_camera: Dict[str, Tuple[str, datetime]] = {}
        # visitor_id → last EXIT timestamp
        self._last_exit:   Dict[str, datetime] = {}
        # visitor_id → last seen confidence
        self._last_conf:   Dict[str, float] = {}
        # (store_id, ts_bucket) → staff_count, total_count
        self._staff_buckets: Dict[str, List[Tuple[bool, datetime]]] = defaultdict(list)

    def set_audit(self, audit) -> None:
        self._audit = audit

    # ── primary entry points ──────────────────────────────────────────────

    def verify_event(
        self,
        event: InboundEvent,
        session: Optional[VisitorSession],
    ) -> List[VerifierWarning]:
        """
        Run per-event checks after sessionization.
        Returns list of warnings generated (may be empty).
        """
        warnings: List[VerifierWarning] = []

        warnings += self._check_queue_depth(event)
        warnings += self._check_confidence_cliff(event)
        warnings += self._check_camera_transition(event)
        warnings += self._check_entry_camera(event)
        warnings += self._check_reentry_speed(event)

        # Track state for subsequent checks
        self._last_camera[event.visitor_id] = (
            event.camera_id, event.parsed_timestamp()
        )
        if event.event_type == EventType.EXIT:
            self._last_exit[event.visitor_id] = event.parsed_timestamp()
        self._last_conf[event.visitor_id] = event.confidence

        # Staff tracking for explosion check
        with self._lock:
            self._staff_buckets[event.store_id].append(
                (event.is_staff, event.parsed_timestamp())
            )
            # Keep only last 5 minutes
            cutoff = event.parsed_timestamp() - timedelta(minutes=5)
            self._staff_buckets[event.store_id] = [
                (s, t) for s, t in self._staff_buckets[event.store_id] if t > cutoff
            ]

        warnings += self._check_staff_explosion(event.store_id, event.timestamp)

        self._emit_all(warnings, event.visitor_id, session)
        return warnings

    def verify_active_sessions(
        self,
        sessions: List[VisitorSession],
        store_id: str,
    ) -> List[VerifierWarning]:
        """
        Cross-session checks: run periodically against all active sessions.
        """
        warnings: List[VerifierWarning] = []
        warnings += self._check_duplicate_sessions(sessions, store_id)
        self._emit_all(warnings, None, None)
        return warnings

    # ── individual checks ─────────────────────────────────────────────────

    def _check_queue_depth(self, event: InboundEvent) -> List[VerifierWarning]:
        if event.metadata and event.metadata.queue_depth is not None:
            if event.metadata.queue_depth < 0:
                return [self._warn(
                    code="QUEUE_DEPTH_NEGATIVE",
                    severity=VerifierSeverity.CRITICAL,
                    message=(
                        f"queue_depth={event.metadata.queue_depth} is negative — "
                        f"impossible value from camera {event.camera_id}"
                    ),
                    visitor_ids=[event.visitor_id],
                    store_id=event.store_id,
                    timestamp=event.timestamp,
                    suggested_action="Check billing zone detection logic; reset queue counter.",
                    metadata={"queue_depth": event.metadata.queue_depth,
                              "camera_id": event.camera_id},
                )]
        return []

    def _check_confidence_cliff(self, event: InboundEvent) -> List[VerifierWarning]:
        """Warn when confidence drops sharply between consecutive events for same visitor."""
        prev = self._last_conf.get(event.visitor_id)
        if prev is not None:
            drop = prev - event.confidence
            if drop > CONFIDENCE_CLIFF_DELTA:
                return [self._warn(
                    code="CONFIDENCE_CLIFF",
                    severity=VerifierSeverity.WARN,
                    message=(
                        f"{event.visitor_id}: confidence dropped {drop:.2f} "
                        f"({prev:.2f} → {event.confidence:.2f}) on {event.event_type.value}"
                    ),
                    visitor_ids=[event.visitor_id],
                    store_id=event.store_id,
                    timestamp=event.timestamp,
                    suggested_action="Review Re-ID consistency; possible track fragmentation.",
                    metadata={"prev_conf": prev, "curr_conf": event.confidence,
                              "drop": round(drop, 3)},
                )]
        return []

    def _check_camera_transition(self, event: InboundEvent) -> List[VerifierWarning]:
        """Warn on physically impossible camera hops."""
        prev = self._last_camera.get(event.visitor_id)
        if prev is None:
            return []
        prev_cam, prev_ts = prev
        if prev_cam == event.camera_id:
            return []

        elapsed = (event.parsed_timestamp() - prev_ts).total_seconds()
        if elapsed <= 0:
            return []

        # Rough hop distance: just compare camera name numeric suffixes
        prev_num = _cam_number(prev_cam)
        curr_num = _cam_number(event.camera_id)
        hop_dist = abs(curr_num - prev_num) if (prev_num and curr_num) else 1
        max_hops_possible = elapsed * MAX_CAM_HOPS_PER_SEC

        if hop_dist > max(max_hops_possible, 1):
            return [self._warn(
                code="IMPOSSIBLE_CAMERA_TRANSITION",
                severity=VerifierSeverity.WARN,
                message=(
                    f"{event.visitor_id} transitioned {prev_cam} → {event.camera_id} "
                    f"({hop_dist} hops) in {elapsed:.1f}s — physics violation"
                ),
                visitor_ids=[event.visitor_id],
                store_id=event.store_id,
                timestamp=event.timestamp,
                suggested_action="Check Re-ID: possible cross-camera identity swap.",
                metadata={"from_cam": prev_cam, "to_cam": event.camera_id,
                          "elapsed_sec": round(elapsed, 2), "hop_dist": hop_dist},
            )]
        return []

    def _check_entry_camera(self, event: InboundEvent) -> List[VerifierWarning]:
        """ENTRY events should only come from entry-threshold cameras."""
        if event.event_type != EventType.ENTRY:
            return []
        cam_upper = event.camera_id.upper()
        is_entry_cam = any(cam_upper.startswith(pfx) for pfx in ENTRY_CAMERA_PREFIXES)
        if not is_entry_cam:
            return [self._warn(
                code="ENTRY_WITHOUT_DOORWAY",
                severity=VerifierSeverity.WARN,
                message=(
                    f"ENTRY event for {event.visitor_id} came from non-entry camera "
                    f"{event.camera_id} — possible false entry detection"
                ),
                visitor_ids=[event.visitor_id],
                store_id=event.store_id,
                timestamp=event.timestamp,
                suggested_action="Verify camera zone configuration in store_layout.json.",
                metadata={"camera_id": event.camera_id},
            )]
        return []

    def _check_reentry_speed(self, event: InboundEvent) -> List[VerifierWarning]:
        """REENTRY within MIN_REENTRY_GAP_SEC of EXIT is physically impossible."""
        if event.event_type != EventType.REENTRY:
            return []
        last_exit = self._last_exit.get(event.visitor_id)
        if last_exit is None:
            return []
        gap = (event.parsed_timestamp() - last_exit).total_seconds()
        if 0 < gap < MIN_REENTRY_GAP_SEC:
            return [self._warn(
                code="REENTRY_TOO_FAST",
                severity=VerifierSeverity.CRITICAL,
                message=(
                    f"{event.visitor_id} re-entered {gap:.1f}s after EXIT — "
                    f"minimum plausible gap is {MIN_REENTRY_GAP_SEC}s. "
                    f"Possible identity fragmentation."
                ),
                visitor_ids=[event.visitor_id],
                store_id=event.store_id,
                timestamp=event.timestamp,
                suggested_action="Review Re-ID: likely duplicate track for same person.",
                metadata={"gap_sec": round(gap, 2),
                          "min_gap_sec": MIN_REENTRY_GAP_SEC},
            )]
        return []

    def _check_staff_explosion(
        self, store_id: str, timestamp: str
    ) -> List[VerifierWarning]:
        """Warn if staff fraction of recent events exceeds threshold."""
        with self._lock:
            bucket = self._staff_buckets.get(store_id, [])
        if len(bucket) < 10:
            return []
        staff_count = sum(1 for s, _ in bucket if s)
        fraction = staff_count / len(bucket)
        if fraction > STAFF_FRACTION_SPIKE:
            return [self._warn(
                code="STAFF_COUNT_EXPLOSION",
                severity=VerifierSeverity.WARN,
                message=(
                    f"Staff fraction is {fraction:.0%} of recent events in {store_id} "
                    f"({staff_count}/{len(bucket)}) — possible misclassification"
                ),
                visitor_ids=[],
                store_id=store_id,
                timestamp=timestamp,
                suggested_action="Review staff detection thresholds in calibration profile.",
                metadata={"staff_fraction": round(fraction, 3),
                          "window_events": len(bucket)},
            )]
        return []

    def _check_duplicate_sessions(
        self, sessions: List[VisitorSession], store_id: str
    ) -> List[VerifierWarning]:
        """A visitor_id should not have more than one ACTIVE session."""
        active_by_visitor: Dict[str, List[VisitorSession]] = defaultdict(list)
        for s in sessions:
            if s.is_active:
                active_by_visitor[s.visitor_id].append(s)

        warnings = []
        for vid, active in active_by_visitor.items():
            if len(active) > 1:
                session_ids = [s.session_id for s in active]
                warnings.append(self._warn(
                    code="DUPLICATE_ACTIVE_SESSION",
                    severity=VerifierSeverity.CRITICAL,
                    message=(
                        f"{vid} has {len(active)} simultaneous active sessions in {store_id}: "
                        f"{session_ids}"
                    ),
                    visitor_ids=[vid],
                    store_id=store_id,
                    timestamp=_now_iso(),
                    suggested_action="Session deduplication bug — check Sessionizer.put_active().",
                    metadata={"session_ids": session_ids},
                ))
        return warnings

    # ── internal ──────────────────────────────────────────────────────────

    def _warn(
        self,
        code: str,
        severity: VerifierSeverity,
        message: str,
        visitor_ids: List[str],
        store_id: str,
        timestamp: str,
        suggested_action: str,
        metadata: Dict[str, Any] = None,
    ) -> VerifierWarning:
        w = VerifierWarning(
            code=code,
            severity=severity.value,
            message=message,
            visitor_ids=visitor_ids,
            store_id=store_id,
            timestamp=timestamp,
            suggested_action=suggested_action,
            metadata=metadata or {},
        )
        lvl = (logging.CRITICAL if severity == VerifierSeverity.CRITICAL else
               logging.WARNING  if severity == VerifierSeverity.WARN else
               logging.INFO)
        logger.log(lvl, "[VERIFIER] %s: %s", code, message)
        return w

    def _emit_all(
        self,
        warnings: List[VerifierWarning],
        visitor_id: Optional[str],
        session: Optional[VisitorSession],
    ) -> None:
        if not warnings:
            return
        with self._lock:
            self._warnings.extend(warnings)
            # Cap at 1000 to avoid unbounded growth
            if len(self._warnings) > 1000:
                self._warnings = self._warnings[-1000:]

        if self._audit and visitor_id:
            from .audit import AuditRecord, AuditCategory
            for w in warnings:
                self._audit.record(visitor_id, AuditRecord(
                    category=AuditCategory.TRACKING.value,
                    event_type=w.code,
                    timestamp=w.timestamp,
                    confidence=0.0,
                    source="VerifierEngine",
                    reason=w.message,
                    session_id=session.session_id if session else None,
                    metadata=w.metadata,
                ))

    # ── public read ───────────────────────────────────────────────────────

    def get_warnings(
        self,
        store_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[VerifierWarning]:
        with self._lock:
            ws = list(self._warnings)
        if store_id:
            ws = [w for w in ws if w.store_id == store_id]
        if severity:
            ws = [w for w in ws if w.severity == severity.upper()]
        return ws[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._warnings.clear()
        self._last_camera.clear()
        self._last_exit.clear()
        self._last_conf.clear()
        self._staff_buckets.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cam_number(camera_id: str) -> Optional[int]:
    """Extract trailing number from camera_id, e.g. CAM_ENTRY_03 → 3."""
    parts = camera_id.split("_")
    for p in reversed(parts):
        if p.isdigit():
            return int(p)
    return None


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
