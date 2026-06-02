"""
app/ingestion.py — Event validation, deduplication, and append-only storage.

Responsibilities:
  1. Parse raw event dicts from POST /events/ingest
  2. Validate each event against the InboundEvent schema
  3. Deduplicate by event_id (idempotency key)
  4. Append validated events to the EventStore
  5. Return a partial-success IngestResponse

Design principles:
  - A malformed event never aborts the rest of the batch (partial success).
  - A repeated event_id silently skips (idempotent).
  - The event store is append-only — no mutation after write.
  - Thread-safe via threading.Lock.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from pydantic import ValidationError

from .models import (
    InboundEvent,
    IngestResponse,
    RejectedEvent,
    StoredEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Append-only Event Store
# ---------------------------------------------------------------------------

class EventStore:
    """
    Append-only in-memory store for validated raw events.

    Invariants:
        - Events are never removed or mutated after insertion.
        - _seen_ids holds all event_ids ever written; used for dedup.
        - All public methods are thread-safe.
    """

    def __init__(self) -> None:
        self._events:   Dict[str, List[StoredEvent]] = {}
        self._seen_ids: Set[str]                     = set()
        self._lock:     threading.Lock               = threading.Lock()

    # ── read ──────────────────────────────────────────────────────────────

    def get_events(self, store_id: str) -> List[StoredEvent]:
        """Return all stored events for a store (snapshot copy)."""
        with self._lock:
            return list(self._events.get(store_id, []))

    def get_all_store_ids(self) -> List[str]:
        with self._lock:
            return list(self._events.keys())

    def has_seen(self, event_id: str) -> bool:
        with self._lock:
            return event_id in self._seen_ids

    def total_count(self, store_id: Optional[str] = None) -> int:
        with self._lock:
            if store_id:
                return len(self._events.get(store_id, []))
            return sum(len(v) for v in self._events.values())

    def last_event_timestamp(self, store_id: str) -> Optional[str]:
        """Timestamp of the most recently stored event for a store."""
        with self._lock:
            events = self._events.get(store_id, [])
            if not events:
                return None
            return events[-1].event.timestamp

    # ── write ─────────────────────────────────────────────────────────────

    def append(self, stored: StoredEvent) -> None:
        """Append a pre-validated, pre-deduped event."""
        store_id = stored.event.store_id
        with self._lock:
            if store_id not in self._events:
                self._events[store_id] = []
            self._events[store_id].append(stored)
            self._seen_ids.add(stored.event.event_id)

    def clear(self) -> None:
        """Wipe all state. For testing only."""
        with self._lock:
            self._events.clear()
            self._seen_ids.clear()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_event(raw: Any) -> InboundEvent:
    if not isinstance(raw, dict):
        raise ValueError(f"Event must be a JSON object, got {type(raw).__name__}")
    return InboundEvent.model_validate(raw)


def _extract_event_id(raw: Any) -> Optional[str]:
    """Best-effort extraction of event_id for error reporting."""
    if isinstance(raw, dict):
        eid = raw.get("event_id")
        if isinstance(eid, str) and eid.strip():
            return eid.strip()
    return None


def _format_validation_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors():
            loc = " → ".join(str(x) for x in err["loc"]) if err["loc"] else "root"
            parts.append(f"{loc}: {err['msg']}")
        return "; ".join(parts)
    return str(exc)


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """
    Orchestrates validation → deduplication → storage for a batch of events.

    The sessionizer is injected after construction to avoid circular imports.
    It is notified after each accepted event so session state stays consistent
    with the event store.
    """

    def __init__(
        self,
        event_store: EventStore,
        sessionizer=None,
    ) -> None:
        self._store       = event_store
        self._sessionizer = sessionizer

    def set_sessionizer(self, sessionizer) -> None:
        self._sessionizer = sessionizer

    def ingest_batch(self, raw_events: List[Any]) -> IngestResponse:
        """
        Validate, deduplicate, and store a batch of raw event dicts.

        For each event:
          1. Validate → InboundEvent (Pydantic)
          2. Deduplicate against seen event_ids
          3. Append to EventStore (append-only)
          4. Notify Sessionizer

        Returns IngestResponse with per-count and per-rejected error details.
        """
        response = IngestResponse()

        for raw in raw_events:
            event_id_hint = _extract_event_id(raw)

            # 1. Validate
            try:
                event = _validate_event(raw)
            except (ValidationError, ValueError) as exc:
                reason = _format_validation_error(exc)
                response.rejected += 1
                response.errors.append(RejectedEvent(event_id=event_id_hint, reason=reason))
                logger.warning("validation_failed event_id=%s reason=%r", event_id_hint, reason)
                continue

            # 2. Deduplicate
            if self._store.has_seen(event.event_id):
                response.duplicates += 1
                logger.debug("duplicate_skipped event_id=%s", event.event_id)
                continue

            # 3. Append
            stored = StoredEvent(event=event)
            self._store.append(stored)
            response.accepted += 1
            logger.debug(
                "event_accepted id=%s store=%s type=%s visitor=%s",
                event.event_id, event.store_id, event.event_type.value, event.visitor_id,
            )

            # 4. Sessionize
            if self._sessionizer is not None:
                try:
                    self._sessionizer.process_event(event)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error("sessionizer_error event_id=%s: %s", event.event_id, exc, exc_info=True)

        logger.info(
            "batch_complete accepted=%d duplicates=%d rejected=%d",
            response.accepted, response.duplicates, response.rejected,
        )
        return response
