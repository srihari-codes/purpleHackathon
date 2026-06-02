"""
app/audit.py — Visitor Audit Timeline for Layer 2 (Session Core).

Every visitor owns a complete, ordered audit chain.  Each record explains
exactly what happened, why, with what confidence, and from which source.

Audit record categories (AuditCategory):
    DETECTION   — raw detection event arrived
    TRACKING    — tracker associated / lost a track
    IDENTITY    — Re-ID decision (reuse, new, reentry)
    SESSION     — session opened / closed / continued
    ZONE        — visitor entered / exited / dwelled in a zone
    QUEUE       — joined or abandoned billing queue
    EXIT        — visitor left the store
    REENTRY     — visitor detected again after prior EXIT

Human-readable explanations are stored verbatim, e.g.:
    "VIS_001 reused because: appearance similarity 0.92,
     camera transition valid, time gap 3.2 sec"

Thread safety: all writes protected by threading.Lock.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit category
# ---------------------------------------------------------------------------

class AuditCategory(str, Enum):
    DETECTION  = "DETECTION"
    TRACKING   = "TRACKING"
    IDENTITY   = "IDENTITY"
    SESSION    = "SESSION"
    ZONE       = "ZONE"
    QUEUE      = "QUEUE"
    EXIT       = "EXIT"
    REENTRY    = "REENTRY"


# ---------------------------------------------------------------------------
# Single audit record
# ---------------------------------------------------------------------------

@dataclass
class AuditRecord:
    """
    One entry in a visitor's audit chain.

    Fields:
        category    — which layer produced this record
        event_type  — the triggering event type (ENTRY, ZONE_DWELL, etc.)
        timestamp   — ISO-8601 UTC
        confidence  — confidence value at decision time (0–1)
        source      — component that created the record (e.g. "Sessionizer", "ReplayEngine")
        reason      — human-readable explanation of the decision
        metadata    — optional structured payload (e.g. reid scores, zone ids)
        session_id  — associated session_id at time of record (may be None for early events)
    """
    category:   str
    event_type: str
    timestamp:  str
    confidence: float
    source:     str
    reason:     str
    metadata:   Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str]  = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ---------------------------------------------------------------------------
# Per-visitor audit timeline
# ---------------------------------------------------------------------------

class VisitorAuditTimeline:
    """
    Ordered audit chain for a single visitor_id.

    Usage:
        timeline = VisitorAuditTimeline(visitor_id="VIS_aaa111")
        timeline.record(AuditRecord(...))
        timeline.explain()          # human-readable summary
    """

    def __init__(self, visitor_id: str) -> None:
        self.visitor_id = visitor_id
        self._records:  List[AuditRecord] = []

    def record(self, entry: AuditRecord) -> None:
        self._records.append(entry)

    @property
    def records(self) -> List[AuditRecord]:
        return list(self._records)

    def records_by_category(self, category: AuditCategory) -> List[AuditRecord]:
        return [r for r in self._records if r.category == category.value]

    def explain(self) -> str:
        """
        Return a multi-line human-readable audit trail for this visitor.

        Example output:
            [VIS_001] Audit Trail (7 records)
            ─────────────────────────────────
            2026-03-03T14:00:00Z  SESSION   ENTRY       conf=0.91  Sessionizer
              Session opened for VIS_001 at STORE_BLR_002
            2026-03-03T14:01:05Z  IDENTITY  REENTRY     conf=0.88  Sessionizer
              VIS_001 reused because: appearance similarity 0.92, camera transition valid, time gap 3.2 sec
        """
        lines = [
            f"[{self.visitor_id}] Audit Trail ({len(self._records)} records)",
            "─" * 56,
        ]
        for r in self._records:
            header = (
                f"{r.timestamp}  {r.category:<10} {r.event_type:<22} "
                f"conf={r.confidence:.2f}  {r.source}"
            )
            lines.append(header)
            lines.append(f"  {r.reason}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "visitor_id": self.visitor_id,
            "record_count": len(self._records),
            "records": [r.to_dict() for r in self._records],
        }


# ---------------------------------------------------------------------------
# Audit Timeline Store
# ---------------------------------------------------------------------------

class AuditTimeline:
    """
    Global store of per-visitor AuditTimelines, indexed by visitor_id.

    The AuditTimeline is the single source-of-truth for audit data.
    The Sessionizer calls record_*() methods after every state change.

    Thread safety: all mutations hold _lock.
    """

    def __init__(self) -> None:
        self._timelines: Dict[str, VisitorAuditTimeline] = {}
        self._lock = threading.Lock()

    # ── write helpers ──────────────────────────────────────────────────────

    def _get_or_create(self, visitor_id: str) -> VisitorAuditTimeline:
        """Must be called with _lock held."""
        if visitor_id not in self._timelines:
            self._timelines[visitor_id] = VisitorAuditTimeline(visitor_id)
        return self._timelines[visitor_id]

    def record(self, visitor_id: str, entry: AuditRecord) -> None:
        with self._lock:
            self._get_or_create(visitor_id).record(entry)

    # ── convenience record methods ─────────────────────────────────────────

    def record_session_open(
        self,
        visitor_id: str,
        session_id: str,
        store_id: str,
        timestamp: str,
        confidence: float,
        is_staff: bool,
        source: str = "Sessionizer",
    ) -> None:
        staff_note = " [STAFF]" if is_staff else ""
        reason = (
            f"Session opened for {visitor_id} at {store_id}{staff_note}. "
            f"session_id={session_id}"
        )
        self.record(visitor_id, AuditRecord(
            category=AuditCategory.SESSION.value,
            event_type="ENTRY",
            timestamp=timestamp,
            confidence=confidence,
            source=source,
            reason=reason,
            session_id=session_id,
            metadata={"store_id": store_id, "is_staff": is_staff},
        ))

    def record_session_close(
        self,
        visitor_id: str,
        session_id: str,
        timestamp: str,
        confidence: float,
        duration_ms: Optional[int],
        source: str = "Sessionizer",
    ) -> None:
        duration_s = f"{duration_ms/1000:.1f}s" if duration_ms is not None else "unknown"
        reason = (
            f"Session closed for {visitor_id}. "
            f"Duration: {duration_s}. session_id={session_id}"
        )
        self.record(visitor_id, AuditRecord(
            category=AuditCategory.EXIT.value,
            event_type="EXIT",
            timestamp=timestamp,
            confidence=confidence,
            source=source,
            reason=reason,
            session_id=session_id,
            metadata={"duration_ms": duration_ms},
        ))

    def record_reentry(
        self,
        visitor_id: str,
        session_id: str,
        timestamp: str,
        confidence: float,
        reentry_count: int,
        reid_score: Optional[float] = None,
        camera_id: Optional[str] = None,
        time_gap_sec: Optional[float] = None,
        source: str = "Sessionizer",
    ) -> None:
        parts = [f"reentry #{reentry_count} for {visitor_id}"]
        if reid_score is not None:
            parts.append(f"appearance similarity {reid_score:.2f}")
        if camera_id:
            parts.append(f"camera={camera_id}")
        if time_gap_sec is not None:
            parts.append(f"time gap {time_gap_sec:.1f} sec")
        reason = "; ".join(parts)
        self.record(visitor_id, AuditRecord(
            category=AuditCategory.REENTRY.value,
            event_type="REENTRY",
            timestamp=timestamp,
            confidence=confidence,
            source=source,
            reason=reason,
            session_id=session_id,
            metadata={
                "reentry_count": reentry_count,
                "reid_score": reid_score,
                "camera_id": camera_id,
                "time_gap_sec": time_gap_sec,
            },
        ))

    def record_zone(
        self,
        visitor_id: str,
        session_id: str,
        event_type: str,
        timestamp: str,
        confidence: float,
        zone_id: str,
        dwell_ms: int = 0,
        source: str = "Sessionizer",
    ) -> None:
        if event_type == "ZONE_ENTER":
            reason = f"Entered zone {zone_id}"
        elif event_type == "ZONE_EXIT":
            reason = f"Left zone {zone_id} after {dwell_ms/1000:.1f}s"
        else:  # ZONE_DWELL
            reason = f"Dwelled in zone {zone_id} for {dwell_ms/1000:.1f}s (periodic checkpoint)"
        self.record(visitor_id, AuditRecord(
            category=AuditCategory.ZONE.value,
            event_type=event_type,
            timestamp=timestamp,
            confidence=confidence,
            source=source,
            reason=reason,
            session_id=session_id,
            metadata={"zone_id": zone_id, "dwell_ms": dwell_ms},
        ))

    def record_queue(
        self,
        visitor_id: str,
        session_id: str,
        event_type: str,
        timestamp: str,
        confidence: float,
        queue_depth: Optional[int] = None,
        wait_ms: Optional[int] = None,
        source: str = "Sessionizer",
    ) -> None:
        if event_type == "BILLING_QUEUE_JOIN":
            reason = f"Joined billing queue (depth={queue_depth}). Marked as purchase_candidate."
        else:
            wait_s = f"{wait_ms/1000:.1f}s" if wait_ms else "unknown"
            reason = f"Abandoned billing queue after waiting {wait_s}"
        self.record(visitor_id, AuditRecord(
            category=AuditCategory.QUEUE.value,
            event_type=event_type,
            timestamp=timestamp,
            confidence=confidence,
            source=source,
            reason=reason,
            session_id=session_id,
            metadata={"queue_depth": queue_depth, "wait_ms": wait_ms},
        ))

    def record_identity_decision(
        self,
        visitor_id: str,
        session_id: Optional[str],
        timestamp: str,
        confidence: float,
        decision: str,        # e.g. "REUSED", "NEW", "STAFF_CLASSIFIED"
        explanation: str,     # free-form human-readable explanation
        source: str = "Sessionizer",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.record(visitor_id, AuditRecord(
            category=AuditCategory.IDENTITY.value,
            event_type=decision,
            timestamp=timestamp,
            confidence=confidence,
            source=source,
            reason=explanation,
            session_id=session_id,
            metadata=extra or {},
        ))

    def record_detection(
        self,
        visitor_id: str,
        event_type: str,
        timestamp: str,
        confidence: float,
        camera_id: str,
        source: str = "IngestionPipeline",
        session_id: Optional[str] = None,
    ) -> None:
        reason = f"{event_type} detected on {camera_id} with confidence {confidence:.2f}"
        self.record(visitor_id, AuditRecord(
            category=AuditCategory.DETECTION.value,
            event_type=event_type,
            timestamp=timestamp,
            confidence=confidence,
            source=source,
            reason=reason,
            session_id=session_id,
            metadata={"camera_id": camera_id},
        ))

    # ── read ───────────────────────────────────────────────────────────────

    def get_timeline(self, visitor_id: str) -> Optional[VisitorAuditTimeline]:
        with self._lock:
            return self._timelines.get(visitor_id)

    def get_all_visitor_ids(self) -> List[str]:
        with self._lock:
            return list(self._timelines.keys())

    def explain(self, visitor_id: str) -> str:
        """Return human-readable audit trail string for a visitor."""
        with self._lock:
            tl = self._timelines.get(visitor_id)
        if tl is None:
            return f"No audit timeline found for visitor {visitor_id}"
        return tl.explain()

    def to_dict(self, visitor_id: str) -> Optional[dict]:
        with self._lock:
            tl = self._timelines.get(visitor_id)
        return tl.to_dict() if tl else None

    def store_summaries(self, store_id: str, session_store) -> List[dict]:
        """
        Return audit summaries for all visitors in a store.
        Requires a SessionStore to resolve visitor→store mapping.
        """
        with self._lock:
            all_ids = list(self._timelines.keys())
        results = []
        for vid in all_ids:
            sessions = session_store.get_all_sessions(store_id)
            if any(s.visitor_id == vid for s in sessions):
                tl = self._timelines.get(vid)
                if tl:
                    results.append({
                        "visitor_id": vid,
                        "record_count": len(tl.records),
                        "categories": list({r.category for r in tl.records}),
                    })
        return results

    def clear(self) -> None:
        with self._lock:
            self._timelines.clear()
