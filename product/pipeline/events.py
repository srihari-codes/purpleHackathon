"""
events.py — Event schema definition and emitter.

All detection layer output must conform to this schema.
Events are written to a .jsonl file AND broadcast over WebSocket.
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
# Event dataclass
# ---------------------------------------------------------------------------
@dataclass
class StoreEvent:
    store_id:    str
    camera_id:   str
    visitor_id:  str
    event_type:  str
    timestamp:   str                        # ISO-8601 UTC
    zone_id:     Optional[str]   = None
    dwell_ms:    int             = 0
    is_staff:    bool            = False
    confidence:  float           = 1.0
    metadata: Dict[str, Any]     = field(default_factory=lambda: {
        "queue_depth": None,
        "sku_zone":    None,
        "session_seq": None,
    })
    event_id:    str             = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        d = asdict(self)
        # Ensure metadata keys always present
        d["metadata"].setdefault("queue_depth", None)
        d["metadata"].setdefault("sku_zone",    None)
        d["metadata"].setdefault("session_seq", None)
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
        self.output_path = Path(output_path)
        self.store_id    = store_id
        self._file       = None
        self._broadcast_queue: asyncio.Queue = None   # set by GUI server
        self._seen_ids   = set()
        self._event_count = 0

    def open(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "a", buffering=1)   # line-buffered
        logger.info(f"EventEmitter writing to {self.output_path}")

    def close(self):
        if self._file:
            self._file.close()

    def set_broadcast_queue(self, q: asyncio.Queue):
        """Called by GUI server so events are also pushed over WebSocket."""
        self._broadcast_queue = q

    def emit(self, event: StoreEvent) -> bool:
        """
        Emit one event. Returns True if written, False if duplicate.
        Duplicates are checked by event_id.
        """
        if event.event_id in self._seen_ids:
            logger.warning(f"Duplicate event_id suppressed: {event.event_id}")
            return False

        # Enforce store_id
        event.store_id = self.store_id

        self._seen_ids.add(event.event_id)
        self._event_count += 1

        line = event.to_json()
        if self._file:
            self._file.write(line + "\n")

        # Non-blocking push to WebSocket queue
        if self._broadcast_queue is not None:
            try:
                self._broadcast_queue.put_nowait(event.to_dict())
            except asyncio.QueueFull:
                pass   # drop from WS if consumer is slow; file always gets it

        logger.debug(f"EVENT {event.event_type} vis={event.visitor_id} "
                     f"cam={event.camera_id} zone={event.zone_id}")
        return True

    def emit_entry(self, visitor_id: str, camera_id: str, timestamp: str,
                   is_staff: bool, confidence: float, session_seq: int,
                   is_reentry: bool = False) -> StoreEvent:
        etype = EventType.REENTRY if is_reentry else EventType.ENTRY
        ev = StoreEvent(
            store_id=self.store_id, camera_id=camera_id,
            visitor_id=visitor_id, event_type=etype,
            timestamp=timestamp, zone_id=None, dwell_ms=0,
            is_staff=is_staff, confidence=confidence,
            metadata={"queue_depth": None, "sku_zone": None, "session_seq": session_seq},
        )
        self.emit(ev)
        return ev

    def emit_exit(self, visitor_id: str, camera_id: str, timestamp: str,
                  is_staff: bool, confidence: float, session_seq: int) -> StoreEvent:
        ev = StoreEvent(
            store_id=self.store_id, camera_id=camera_id,
            visitor_id=visitor_id, event_type=EventType.EXIT,
            timestamp=timestamp, zone_id=None, dwell_ms=0,
            is_staff=is_staff, confidence=confidence,
            metadata={"queue_depth": None, "sku_zone": None, "session_seq": session_seq},
        )
        self.emit(ev)
        return ev

    def emit_zone_enter(self, visitor_id: str, camera_id: str, timestamp: str,
                        zone_id: str, sku_zone: str, is_staff: bool,
                        confidence: float, session_seq: int) -> StoreEvent:
        ev = StoreEvent(
            store_id=self.store_id, camera_id=camera_id,
            visitor_id=visitor_id, event_type=EventType.ZONE_ENTER,
            timestamp=timestamp, zone_id=zone_id, dwell_ms=0,
            is_staff=is_staff, confidence=confidence,
            metadata={"queue_depth": None, "sku_zone": sku_zone, "session_seq": session_seq},
        )
        self.emit(ev)
        return ev

    def emit_zone_exit(self, visitor_id: str, camera_id: str, timestamp: str,
                       zone_id: str, sku_zone: str, dwell_ms: int,
                       is_staff: bool, confidence: float, session_seq: int) -> StoreEvent:
        ev = StoreEvent(
            store_id=self.store_id, camera_id=camera_id,
            visitor_id=visitor_id, event_type=EventType.ZONE_EXIT,
            timestamp=timestamp, zone_id=zone_id, dwell_ms=dwell_ms,
            is_staff=is_staff, confidence=confidence,
            metadata={"queue_depth": None, "sku_zone": sku_zone, "session_seq": session_seq},
        )
        self.emit(ev)
        return ev

    def emit_zone_dwell(self, visitor_id: str, camera_id: str, timestamp: str,
                        zone_id: str, sku_zone: str, dwell_ms: int,
                        is_staff: bool, confidence: float, session_seq: int) -> StoreEvent:
        ev = StoreEvent(
            store_id=self.store_id, camera_id=camera_id,
            visitor_id=visitor_id, event_type=EventType.ZONE_DWELL,
            timestamp=timestamp, zone_id=zone_id, dwell_ms=dwell_ms,
            is_staff=is_staff, confidence=confidence,
            metadata={"queue_depth": None, "sku_zone": sku_zone, "session_seq": session_seq},
        )
        self.emit(ev)
        return ev

    def emit_billing_queue_join(self, visitor_id: str, camera_id: str,
                                timestamp: str, queue_depth: int,
                                is_staff: bool, confidence: float,
                                session_seq: int) -> StoreEvent:
        ev = StoreEvent(
            store_id=self.store_id, camera_id=camera_id,
            visitor_id=visitor_id, event_type=EventType.BILLING_QUEUE_JOIN,
            timestamp=timestamp, zone_id="ZONE_BILLING_QUEUE", dwell_ms=0,
            is_staff=is_staff, confidence=confidence,
            metadata={"queue_depth": queue_depth, "sku_zone": "BILLING_QUEUE",
                      "session_seq": session_seq},
        )
        self.emit(ev)
        return ev

    def emit_billing_queue_abandon(self, visitor_id: str, camera_id: str,
                                   timestamp: str, dwell_ms: int,
                                   is_staff: bool, confidence: float,
                                   session_seq: int) -> StoreEvent:
        ev = StoreEvent(
            store_id=self.store_id, camera_id=camera_id,
            visitor_id=visitor_id, event_type=EventType.BILLING_QUEUE_ABANDON,
            timestamp=timestamp, zone_id="ZONE_BILLING_QUEUE", dwell_ms=dwell_ms,
            is_staff=is_staff, confidence=confidence,
            metadata={"queue_depth": None, "sku_zone": "BILLING_QUEUE",
                      "session_seq": session_seq},
        )
        self.emit(ev)
        return ev

    @property
    def count(self) -> int:
        return self._event_count
