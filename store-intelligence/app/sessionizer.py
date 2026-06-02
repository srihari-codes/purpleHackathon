"""
app/sessionizer.py — Converts raw detection events into canonical VisitorSessions.

The sessionizer is the bridge between the raw event stream and all API metrics.
The funnel, heatmap, metrics, and anomaly endpoints operate EXCLUSIVELY on
VisitorSession objects — never on raw events.

Session lifecycle (spec):
    ENTRY          → open a new session for visitor_id
    REENTRY        → continue visitor history; increment reentry_count
                     (must NOT double-count the visitor)
    EXIT           → close the active session (set end_time)
    ZONE_ENTER/
    ZONE_EXIT/
    ZONE_DWELL     → enrich session: zones_visited, dwell_per_zone
    BILLING_QUEUE_JOIN    → enrich queue_events; set purchase_candidate=True
    BILLING_QUEUE_ABANDON → enrich queue_events

Thread safety:
    All state mutations are protected by a single threading.Lock.

Visitor identity model:
    visitor_id is the stable token assigned by the Detection Layer Re-ID.
    One visitor_id → one VisitorSession (the current / most-recent one).
    On REENTRY: a NEW session is opened for the re-entering visit but the
    reentry_count is incremented and the visitor is tracked in
    _reentry_counts so we never double-count in unique-visitor metrics.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

from .models import (
    EventType,
    InboundEvent,
    QueueEvent,
    VisitorSession,
)

if TYPE_CHECKING:
    from .audit import AuditTimeline
    from .calibration import CalibrationEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session Store
# ---------------------------------------------------------------------------

class SessionStore:
    """
    Append-only store for completed and active VisitorSessions.

    Structure:
        _sessions: dict[store_id → list[VisitorSession]]
            Every closed session lives here.
        _active:   dict[visitor_id → VisitorSession]
            Currently open sessions (no end_time yet).

    Invariants:
        - A session is moved from _active to _sessions[store_id] on EXIT.
        - Active sessions also appear in _sessions as their "live" record.
          We don't duplicate them; callers must merge both lists when needed.
        - Sessions are never mutated once closed (end_time is set).
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, List[VisitorSession]] = defaultdict(list)
        self._active:   Dict[str, VisitorSession]        = {}   # visitor_id → session
        self._lock:     threading.Lock                   = threading.Lock()

    # ── write ─────────────────────────────────────────────────────────────

    def put_active(self, session: VisitorSession) -> None:
        """Register a new open session."""
        with self._lock:
            self._active[session.visitor_id] = session
            self._sessions[session.store_id].append(session)

    def close_session(self, visitor_id: str, end_time: str) -> Optional[VisitorSession]:
        """Mark the active session for visitor_id as closed."""
        with self._lock:
            session = self._active.pop(visitor_id, None)
            if session is not None:
                session.end_time = end_time
                logger.debug(
                    "session_closed visitor=%s session=%s duration_ms=%s",
                    visitor_id, session.session_id, session.duration_ms,
                )
            return session

    # ── read ──────────────────────────────────────────────────────────────

    def get_active(self, visitor_id: str) -> Optional[VisitorSession]:
        with self._lock:
            return self._active.get(visitor_id)

    def get_all_sessions(self, store_id: str) -> List[VisitorSession]:
        """Return all sessions (open + closed) for a store."""
        with self._lock:
            return list(self._sessions.get(store_id, []))

    def get_all_store_ids(self) -> List[str]:
        with self._lock:
            return list(self._sessions.keys())

    def get_active_count(self, store_id: Optional[str] = None) -> int:
        with self._lock:
            if store_id:
                return sum(
                    1 for s in self._active.values() if s.store_id == store_id
                )
            return len(self._active)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._active.clear()


# ---------------------------------------------------------------------------
# Sessionizer
# ---------------------------------------------------------------------------

class Sessionizer:
    """
    Stateful event processor that builds VisitorSessions from InboundEvents.

    One Sessionizer instance is shared across the API lifetime.
    It wraps a SessionStore and exposes process_event() for the ingestion
    pipeline to call after each accepted event.

    Optional integrations (injected after construction):
        audit       — AuditTimeline: records every state change with explanations
        calibration — CalibrationEngine: feeds observations for threshold tuning
    """

    def __init__(
        self,
        session_store: SessionStore,
        audit: Optional["AuditTimeline"] = None,
        calibration: Optional["CalibrationEngine"] = None,
        verifier = None,
    ) -> None:
        self._store         = session_store
        self._audit         = audit
        self._calibration   = calibration
        self._verifier      = verifier
        # visitor_id → total re-entry count across all time
        self._reentry_counts: Dict[str, int] = defaultdict(int)
        self._lock          = threading.RLock()

    def set_audit(self, audit: "AuditTimeline") -> None:
        self._audit = audit

    def set_calibration(self, calibration: "CalibrationEngine") -> None:
        self._calibration = calibration

    def set_verifier(self, verifier) -> None:
        self._verifier = verifier

    # ── public API ────────────────────────────────────────────────────────

    @property
    def store(self) -> SessionStore:
        return self._store

    def process_event(self, event: InboundEvent) -> None:
        """
        Route event to the appropriate handler based on event_type.
        This is the single entry point from the ingestion pipeline.
        """
        with self._lock:
            etype = event.event_type

            if etype == EventType.ENTRY:
                self._handle_entry(event)
            elif etype == EventType.REENTRY:
                self._handle_reentry(event)
            elif etype == EventType.EXIT:
                self._handle_exit(event)
            elif etype in (EventType.ZONE_ENTER, EventType.ZONE_EXIT, EventType.ZONE_DWELL):
                self._handle_zone_event(event)
            elif etype == EventType.BILLING_QUEUE_JOIN:
                self._handle_queue_join(event)
            elif etype == EventType.BILLING_QUEUE_ABANDON:
                self._handle_queue_abandon(event)
            else:
                logger.warning("sessionizer_unhandled_event_type type=%s", etype)

            if self._verifier is not None:
                session = self._store.get_active(event.visitor_id)
                try:
                    self._verifier.verify_event(event, session)
                except Exception as exc:
                    logger.error("verifier_error event_id=%s: %s", event.event_id, exc, exc_info=True)

    # ── event handlers ────────────────────────────────────────────────────

    def _handle_entry(self, event: InboundEvent) -> None:
        """
        ENTRY → open a new session.

        If a session is already active for this visitor (e.g. a missed EXIT),
        we close the old one before opening the new one to avoid orphaned state.
        """
        visitor_id = event.visitor_id

        # Close any orphaned active session for this visitor
        existing = self._store.get_active(visitor_id)
        if existing is not None:
            logger.warning(
                "orphaned_session_closed visitor=%s session=%s (new ENTRY without prior EXIT)",
                visitor_id, existing.session_id,
            )
            self._store.close_session(visitor_id, event.timestamp)

        session = VisitorSession(
            visitor_id   = visitor_id,
            store_id     = event.store_id,
            start_time   = event.timestamp,
            is_staff     = event.is_staff,
            event_count  = 1,
            avg_confidence = event.confidence,
        )
        self._store.put_active(session)
        logger.debug(
            "session_opened visitor=%s session=%s staff=%s",
            visitor_id, session.session_id, event.is_staff,
        )

        # Audit
        if self._audit:
            self._audit.record_detection(
                visitor_id, event.event_type.value, event.timestamp,
                event.confidence, event.camera_id, session_id=session.session_id,
            )
            self._audit.record_session_open(
                visitor_id, session.session_id, event.store_id,
                event.timestamp, event.confidence, event.is_staff,
            )
        # Calibration
        if self._calibration:
            self._calibration.observe_event(
                event.store_id, event.camera_id, event.confidence
            )

    def _handle_reentry(self, event: InboundEvent) -> None:
        """
        REENTRY → continue visitor history.

        Rules:
        - Increment reentry_count (never double-count the visitor in funnel).
        - If no active session exists, open one (robust to missed EXIT).
        - is_staff propagates from the event.
        """
        visitor_id = event.visitor_id
        reid_score = event.metadata.reid_score if event.metadata else None

        with self._lock:
            self._reentry_counts[visitor_id] += 1
            count = self._reentry_counts[visitor_id]

        session = self._store.get_active(visitor_id)
        if session is None:
            # Open a continuation session for this re-entry
            session = VisitorSession(
                visitor_id    = visitor_id,
                store_id      = event.store_id,
                start_time    = event.timestamp,
                is_staff      = event.is_staff,
                reentry_count = count,
                event_count   = 1,
                avg_confidence = event.confidence,
            )
            self._store.put_active(session)
            logger.debug(
                "reentry_new_session visitor=%s reentry_count=%d",
                visitor_id, count,
            )
        else:
            session.reentry_count = count
            self._update_avg_confidence(session, event.confidence)
            session.event_count += 1
            logger.debug(
                "reentry_continued visitor=%s reentry_count=%d session=%s",
                visitor_id, count, session.session_id,
            )

        # Audit
        if self._audit:
            self._audit.record_reentry(
                visitor_id, session.session_id, event.timestamp,
                event.confidence, count, reid_score=reid_score,
                camera_id=event.camera_id,
            )
        # Calibration: reid observations help tune reid_confidence_threshold
        if self._calibration and reid_score is not None:
            self._calibration.observe_reid(event.store_id, reid_score)

    def _handle_exit(self, event: InboundEvent) -> None:
        """EXIT → close the active session."""
        visitor_id = event.visitor_id
        session = self._store.get_active(visitor_id)

        if session is None:
            # EXIT without a known ENTRY — record a stub closed session
            logger.warning(
                "exit_without_entry visitor=%s — creating stub closed session",
                visitor_id,
            )
            session = VisitorSession(
                visitor_id   = visitor_id,
                store_id     = event.store_id,
                start_time   = event.timestamp,   # best-effort
                is_staff     = event.is_staff,
                event_count  = 1,
                avg_confidence = event.confidence,
            )
            self._store.put_active(session)

        self._update_avg_confidence(session, event.confidence)
        session.event_count += 1
        closed = self._store.close_session(visitor_id, event.timestamp)

        # Audit
        if self._audit and closed:
            self._audit.record_session_close(
                visitor_id, closed.session_id, event.timestamp,
                event.confidence, closed.duration_ms,
            )
        # Calibration
        if self._calibration:
            self._calibration.observe_event(
                event.store_id, event.camera_id, event.confidence
            )

    def _handle_zone_event(self, event: InboundEvent) -> None:
        """ZONE_ENTER / ZONE_EXIT / ZONE_DWELL → enrich zone data on session."""
        session = self._get_or_create_session(event)
        if session is None:
            return

        zone_id = event.zone_id
        if not zone_id:
            return

        if event.event_type == EventType.ZONE_ENTER:
            session.record_zone_visit(zone_id)

        if event.event_type in (EventType.ZONE_EXIT, EventType.ZONE_DWELL):
            session.add_zone_dwell(zone_id, event.dwell_ms)

        self._update_avg_confidence(session, event.confidence)
        session.event_count += 1

        # Audit
        if self._audit:
            self._audit.record_zone(
                event.visitor_id, session.session_id,
                event.event_type.value, event.timestamp,
                event.confidence, zone_id, event.dwell_ms,
            )
        # Calibration: feed zone dwell observations
        if self._calibration and event.dwell_ms > 0:
            self._calibration.observe_zone_dwell(event.store_id, float(event.dwell_ms))

    def _handle_queue_join(self, event: InboundEvent) -> None:
        """BILLING_QUEUE_JOIN → record queue event; flag purchase_candidate."""
        session = self._get_or_create_session(event)
        if session is None:
            return

        qe = QueueEvent(
            event_type  = EventType.BILLING_QUEUE_JOIN,
            timestamp   = event.timestamp,
            queue_depth = event.metadata.queue_depth,
        )
        session.queue_events.append(qe)
        session.purchase_candidate = True   # joined billing queue → candidate

        # Also track billing zone visit
        zone_id = event.zone_id or "ZONE_BILLING_QUEUE"
        session.record_zone_visit(zone_id)

        self._update_avg_confidence(session, event.confidence)
        session.event_count += 1
        logger.debug(
            "queue_join visitor=%s session=%s queue_depth=%s",
            event.visitor_id, session.session_id, event.metadata.queue_depth,
        )

        # Audit
        if self._audit:
            self._audit.record_queue(
                event.visitor_id, session.session_id,
                event.event_type.value, event.timestamp, event.confidence,
                queue_depth=event.metadata.queue_depth,
            )
        # Calibration: queue dwell observations
        if self._calibration and event.dwell_ms > 0:
            self._calibration.observe_queue_dwell(
                event.store_id, event.dwell_ms / 1000.0
            )

    def _handle_queue_abandon(self, event: InboundEvent) -> None:
        """BILLING_QUEUE_ABANDON → record abandon event."""
        session = self._get_or_create_session(event)
        if session is None:
            return

        wait_ms = event.metadata.wait_duration_ms or event.dwell_ms or None
        qe = QueueEvent(
            event_type = EventType.BILLING_QUEUE_ABANDON,
            timestamp  = event.timestamp,
            wait_ms    = wait_ms,
        )
        session.queue_events.append(qe)

        self._update_avg_confidence(session, event.confidence)
        session.event_count += 1
        logger.debug(
            "queue_abandon visitor=%s session=%s",
            event.visitor_id, session.session_id,
        )

        # Audit
        if self._audit:
            self._audit.record_queue(
                event.visitor_id, session.session_id,
                event.event_type.value, event.timestamp, event.confidence,
                wait_ms=wait_ms,
            )
        # Calibration: queue dwell observation
        if self._calibration and wait_ms:
            self._calibration.observe_queue_dwell(
                event.store_id, wait_ms / 1000.0
            )

    # ── internal helpers ──────────────────────────────────────────────────

    def _get_or_create_session(self, event: InboundEvent) -> Optional[VisitorSession]:
        """
        Return the active session for event.visitor_id.

        If no session exists (zone/queue event arrived before ENTRY — possible
        in out-of-order delivery or camera cut-in), create a best-effort open
        session so that enrichment is not lost.
        """
        session = self._store.get_active(event.visitor_id)
        if session is not None:
            return session

        # No active session — open one cautiously
        logger.warning(
            "event_before_entry visitor=%s type=%s — opening implicit session",
            event.visitor_id, event.event_type.value,
        )
        session = VisitorSession(
            visitor_id   = event.visitor_id,
            store_id     = event.store_id,
            start_time   = event.timestamp,
            is_staff     = event.is_staff,
            event_count  = 0,
            avg_confidence = event.confidence,
        )
        self._store.put_active(session)
        return session

    @staticmethod
    def _update_avg_confidence(session: VisitorSession, new_conf: float) -> None:
        """Incrementally update the rolling average confidence for a session."""
        n = session.event_count
        if n == 0:
            session.avg_confidence = new_conf
        else:
            session.avg_confidence = (session.avg_confidence * n + new_conf) / (n + 1)


# ---------------------------------------------------------------------------
# Factory helper (used by app/main.py to wire everything together)
# ---------------------------------------------------------------------------

def build_session_pipeline(
    audit: Optional["AuditTimeline"] = None,
    calibration: Optional["CalibrationEngine"] = None,
) -> tuple[SessionStore, Sessionizer]:
    """
    Create a ready-to-use SessionStore + Sessionizer pair.

    Args:
        audit       — AuditTimeline instance (optional; enables full audit recording)
        calibration — CalibrationEngine instance (optional; enables threshold tuning)

    Returns:
        (session_store, sessionizer)
    """
    store       = SessionStore()
    sessionizer = Sessionizer(store, audit=audit, calibration=calibration)
    return store, sessionizer
