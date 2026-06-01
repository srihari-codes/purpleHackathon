"""
events.py — Event schema definition and emitter.

All detection layer output must conform to this schema.
Events are written to a .jsonl file AND broadcast over WebSocket.

Confidence propagation (spec):
    final_confidence = detection_confidence
                     × tracking_confidence
                     × reid_confidence
                     × zone_confidence
"""

import uuid
import json
import asyncio
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event type catalogue
# ---------------------------------------------------------------------------
class EventType:
    ENTRY                = "ENTRY"
    EXIT                 = "EXIT"
    ZONE_ENTER           = "ZONE_ENTER"
    ZONE_EXIT            = "ZONE_EXIT"
    ZONE_DWELL           = "ZONE_DWELL"
    BILLING_QUEUE_JOIN   = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON= "BILLING_QUEUE_ABANDON"
    REENTRY              = "REENTRY"


# ---------------------------------------------------------------------------
# Event dataclass (spec-compliant schema)
# ---------------------------------------------------------------------------
@dataclass
class StoreEvent:
    store_id:   str
    camera_id:  str
    visitor_id: str
    event_type: str
    timestamp:  str                         # ISO-8601 UTC
    zone_id:    Optional[str]  = None
    dwell_ms:   int            = 0
    is_staff:   bool           = False
    confidence: float          = 1.0        # final_confidence (propagated)
    metadata: Dict[str, Any]   = field(default_factory=lambda: {
        # Required schema fields
        "queue_depth":      None,
        "sku_zone":         None,
        "session_seq":      None,
        # Enrichment fields (spec + gap closures)
        "behavior_state":   None,    # GAP-2: current state from BehaviorStateMachine
        "reid_score":       None,    # GAP-4: composite ReID score on re-association
        "reentry_count":    None,    # GAP-1: total re-entries for this visitor
        "session_duration_ms": None, # GAP-12: total time since first_seen
        "wait_duration_ms": None,    # queue abandon: time spent waiting
        "det_conf":         None,    # confidence pipeline components
        "track_conf":       None,
        "reid_conf":        None,
        "zone_conf":        None,
    })
    event_id:   str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        d = asdict(self)
        # Ensure all required metadata keys are present
        for k in ("queue_depth", "sku_zone", "session_seq",
                  "behavior_state", "reid_score", "reentry_count",
                  "session_duration_ms", "wait_duration_ms",
                  "det_conf", "track_conf", "reid_conf", "zone_conf"):
            d["metadata"].setdefault(k, None)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def make_timestamp(camera_start_time: datetime, frame_index: int, fps: float = 15.0) -> str:
    """
    Compute ISO-8601 UTC timestamp for a given frame.
    camera_start_time must be a timezone-aware datetime (UTC).
    """
    offset_sec = frame_index / fps
    ts = camera_start_time.timestamp() + offset_sec
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Event emitter
# ---------------------------------------------------------------------------
class EventEmitter:
    """
    Thread-safe event emitter.
    Writes to a JSONL file and queues events for WebSocket broadcast.
    """

    def __init__(self, output_path: str, store_id: str):
        self.output_path  = Path(output_path)
        self.store_id     = store_id
        self._file        = None
        self._broadcast_queue: asyncio.Queue = None
        self._seen_ids    = set()
        self._event_count = 0

    def open(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "a", buffering=1)
        logger.info(f"EventEmitter writing to {self.output_path}")

    def close(self):
        if self._file:
            self._file.close()

    def set_broadcast_queue(self, q: asyncio.Queue):
        self._broadcast_queue = q

    def emit(self, event: StoreEvent) -> bool:
        """Emit one event. Returns True if written, False if duplicate."""
        if event.event_id in self._seen_ids:
            logger.warning(f"Duplicate event_id suppressed: {event.event_id}")
            return False

        event.store_id = self.store_id
        self._seen_ids.add(event.event_id)
        self._event_count += 1

        line = event.to_json()
        if self._file:
            self._file.write(line + "\n")

        if self._broadcast_queue is not None:
            try:
                self._broadcast_queue.put_nowait(event.to_dict())
            except asyncio.QueueFull:
                pass

        logger.debug(
            f"EVENT {event.event_type} vis={event.visitor_id} "
            f"cam={event.camera_id} zone={event.zone_id} "
            f"conf={event.confidence:.3f}"
        )
        return True

    # ------------------------------------------------------------------
    # Convenience emitters (include all enrichment fields)
    # ------------------------------------------------------------------

    def emit_entry(
        self, visitor_id: str, camera_id: str, timestamp: str,
        is_staff: bool, confidence: float, session_seq: int,
        is_reentry: bool = False,
        reentry_count: int = 0,
        behavior_state: str = "ENTERED",
        session_duration_ms: Optional[int] = None,
        det_conf: float = 1.0, track_conf: float = 1.0,
        reid_conf: float = 1.0, zone_conf: float = 1.0,
    ) -> StoreEvent:
        etype = EventType.REENTRY if is_reentry else EventType.ENTRY
        ev = StoreEvent(
            store_id=self.store_id, camera_id=camera_id,
            visitor_id=visitor_id, event_type=etype,
            timestamp=timestamp, zone_id=None, dwell_ms=0,
            is_staff=is_staff, confidence=confidence,
            metadata={
                "queue_depth": None, "sku_zone": None, "session_seq": session_seq,
                "behavior_state": behavior_state,
                "reid_score": None,
                "reentry_count": reentry_count,
                "session_duration_ms": session_duration_ms,
                "wait_duration_ms": None,
                "det_conf": round(det_conf, 4),
                "track_conf": round(track_conf, 4),
                "reid_conf": round(reid_conf, 4),
                "zone_conf": round(zone_conf, 4),
            },
        )
        self.emit(ev)
        return ev

    def emit_exit(
        self, visitor_id: str, camera_id: str, timestamp: str,
        is_staff: bool, confidence: float, session_seq: int,
        session_duration_ms: Optional[int] = None,
        behavior_state: str = "EXITED",
        det_conf: float = 1.0, track_conf: float = 1.0,
        reid_conf: float = 1.0, zone_conf: float = 1.0,
    ) -> StoreEvent:
        ev = StoreEvent(
            store_id=self.store_id, camera_id=camera_id,
            visitor_id=visitor_id, event_type=EventType.EXIT,
            timestamp=timestamp, zone_id=None, dwell_ms=0,
            is_staff=is_staff, confidence=confidence,
            metadata={
                "queue_depth": None, "sku_zone": None, "session_seq": session_seq,
                "behavior_state": behavior_state,
                "reid_score": None,
                "reentry_count": None,
                "session_duration_ms": session_duration_ms,
                "wait_duration_ms": None,
                "det_conf": round(det_conf, 4),
                "track_conf": round(track_conf, 4),
                "reid_conf": round(reid_conf, 4),
                "zone_conf": round(zone_conf, 4),
            },
        )
        self.emit(ev)
        return ev

    def emit_zone_enter(
        self, visitor_id: str, camera_id: str, timestamp: str,
        zone_id: str, sku_zone: str, is_staff: bool,
        confidence: float, session_seq: int,
        behavior_state: str = "BROWSING",
        reid_score: Optional[float] = None,
        det_conf: float = 1.0, track_conf: float = 1.0,
        reid_conf: float = 1.0, zone_conf: float = 1.0,
    ) -> StoreEvent:
        ev = StoreEvent(
            store_id=self.store_id, camera_id=camera_id,
            visitor_id=visitor_id, event_type=EventType.ZONE_ENTER,
            timestamp=timestamp, zone_id=zone_id, dwell_ms=0,
            is_staff=is_staff, confidence=confidence,
            metadata={
                "queue_depth": None, "sku_zone": sku_zone, "session_seq": session_seq,
                "behavior_state": behavior_state,
                "reid_score": round(reid_score, 4) if reid_score else None,
                "reentry_count": None, "session_duration_ms": None,
                "wait_duration_ms": None,
                "det_conf": round(det_conf, 4),
                "track_conf": round(track_conf, 4),
                "reid_conf": round(reid_conf, 4),
                "zone_conf": round(zone_conf, 4),
            },
        )
        self.emit(ev)
        return ev

    def emit_zone_exit(
        self, visitor_id: str, camera_id: str, timestamp: str,
        zone_id: str, sku_zone: str, dwell_ms: int,
        is_staff: bool, confidence: float, session_seq: int,
        behavior_state: str = "BROWSING",
        det_conf: float = 1.0, track_conf: float = 1.0,
        reid_conf: float = 1.0, zone_conf: float = 1.0,
    ) -> StoreEvent:
        ev = StoreEvent(
            store_id=self.store_id, camera_id=camera_id,
            visitor_id=visitor_id, event_type=EventType.ZONE_EXIT,
            timestamp=timestamp, zone_id=zone_id, dwell_ms=dwell_ms,
            is_staff=is_staff, confidence=confidence,
            metadata={
                "queue_depth": None, "sku_zone": sku_zone, "session_seq": session_seq,
                "behavior_state": behavior_state,
                "reid_score": None, "reentry_count": None,
                "session_duration_ms": None, "wait_duration_ms": None,
                "det_conf": round(det_conf, 4),
                "track_conf": round(track_conf, 4),
                "reid_conf": round(reid_conf, 4),
                "zone_conf": round(zone_conf, 4),
            },
        )
        self.emit(ev)
        return ev

    def emit_zone_dwell(
        self, visitor_id: str, camera_id: str, timestamp: str,
        zone_id: str, sku_zone: str, dwell_ms: int,
        is_staff: bool, confidence: float, session_seq: int,
        behavior_state: str = "DWELLING",
        det_conf: float = 1.0, track_conf: float = 1.0,
        reid_conf: float = 1.0, zone_conf: float = 1.0,
    ) -> StoreEvent:
        ev = StoreEvent(
            store_id=self.store_id, camera_id=camera_id,
            visitor_id=visitor_id, event_type=EventType.ZONE_DWELL,
            timestamp=timestamp, zone_id=zone_id, dwell_ms=dwell_ms,
            is_staff=is_staff, confidence=confidence,
            metadata={
                "queue_depth": None, "sku_zone": sku_zone, "session_seq": session_seq,
                "behavior_state": behavior_state,
                "reid_score": None, "reentry_count": None,
                "session_duration_ms": None, "wait_duration_ms": None,
                "det_conf": round(det_conf, 4),
                "track_conf": round(track_conf, 4),
                "reid_conf": round(reid_conf, 4),
                "zone_conf": round(zone_conf, 4),
            },
        )
        self.emit(ev)
        return ev

    def emit_billing_queue_join(
        self, visitor_id: str, camera_id: str,
        timestamp: str, queue_depth: int,
        is_staff: bool, confidence: float, session_seq: int,
        behavior_state: str = "QUEUEING",
        det_conf: float = 1.0, track_conf: float = 1.0,
        reid_conf: float = 1.0, zone_conf: float = 1.0,
    ) -> StoreEvent:
        ev = StoreEvent(
            store_id=self.store_id, camera_id=camera_id,
            visitor_id=visitor_id, event_type=EventType.BILLING_QUEUE_JOIN,
            timestamp=timestamp, zone_id="ZONE_BILLING_QUEUE", dwell_ms=0,
            is_staff=is_staff, confidence=confidence,
            metadata={
                "queue_depth": queue_depth,
                "sku_zone": "BILLING_QUEUE",
                "session_seq": session_seq,
                "behavior_state": behavior_state,
                "reid_score": None, "reentry_count": None,
                "session_duration_ms": None, "wait_duration_ms": None,
                "det_conf": round(det_conf, 4),
                "track_conf": round(track_conf, 4),
                "reid_conf": round(reid_conf, 4),
                "zone_conf": round(zone_conf, 4),
            },
        )
        self.emit(ev)
        return ev

    def emit_billing_queue_abandon(
        self, visitor_id: str, camera_id: str,
        timestamp: str, dwell_ms: int,
        is_staff: bool, confidence: float, session_seq: int,
        behavior_state: str = "BROWSING",
        wait_duration_ms: Optional[int] = None,
        det_conf: float = 1.0, track_conf: float = 1.0,
        reid_conf: float = 1.0, zone_conf: float = 1.0,
    ) -> StoreEvent:
        ev = StoreEvent(
            store_id=self.store_id, camera_id=camera_id,
            visitor_id=visitor_id, event_type=EventType.BILLING_QUEUE_ABANDON,
            timestamp=timestamp, zone_id="ZONE_BILLING_QUEUE", dwell_ms=dwell_ms,
            is_staff=is_staff, confidence=confidence,
            metadata={
                "queue_depth": None,
                "sku_zone": "BILLING_QUEUE",
                "session_seq": session_seq,
                "behavior_state": behavior_state,
                "reid_score": None, "reentry_count": None,
                "session_duration_ms": None,
                "wait_duration_ms": wait_duration_ms if wait_duration_ms else dwell_ms,
                "det_conf": round(det_conf, 4),
                "track_conf": round(track_conf, 4),
                "reid_conf": round(reid_conf, 4),
                "zone_conf": round(zone_conf, 4),
            },
        )
        self.emit(ev)
        return ev

    @property
    def count(self) -> int:
        return self._event_count
