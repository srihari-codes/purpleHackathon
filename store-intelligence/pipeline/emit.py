"""
emit.py — Event schema builder and emitter.
Constructs StoreEvent objects from raw detection data and POSTs them to the API.
Also writes a local JSONL file alongside (for auditing / replay).
"""

from __future__ import annotations
import json
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import requests

from pipeline.tracker import VisitorTracker

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
INGEST_ENDPOINT = f"{API_BASE_URL}/events/ingest"
BATCH_SIZE = 200   # flush when buffer reaches this size
DWELL_INTERVAL_SECONDS = 30   # emit ZONE_DWELL every 30s of continuous presence


class EventEmitter:
    """
    Buffers events and flushes to the API in batches.
    Also writes to a local JSONL file for traceability.
    """

    def __init__(
        self,
        store_id: str,
        clip_start_time: datetime,
        output_jsonl: Optional[Path] = None,
    ):
        self.store_id = store_id
        self.clip_start_time = clip_start_time
        self._buffer: List[dict] = []
        self._output_file = output_jsonl
        self._session_seqs: dict = {}  # visitor_id → seq counter
        self._zone_dwell_start: dict = {}  # visitor_id → (zone_id, entry_datetime)

    # ─────────────────────────────────────────
    # Frame timestamp helper
    # ─────────────────────────────────────────

    def frame_to_ts(self, frame_number: int, fps: float = 15.0) -> datetime:
        """Convert frame index to ISO-8601 UTC timestamp."""
        offset = timedelta(seconds=frame_number / fps)
        return (self.clip_start_time + offset).replace(tzinfo=timezone.utc)

    # ─────────────────────────────────────────
    # Event builders
    # ─────────────────────────────────────────

    def _build_event(
        self,
        camera_id: str,
        visitor_id: str,
        event_type: str,
        timestamp: datetime,
        zone_id: Optional[str] = None,
        dwell_ms: int = 0,
        is_staff: bool = False,
        confidence: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> dict:
        seq = self._session_seqs.get(visitor_id, 0) + 1
        self._session_seqs[visitor_id] = seq

        evt_meta = {"session_seq": seq}
        if metadata:
            evt_meta.update(metadata)

        return {
            "event_id": str(uuid.uuid4()),
            "store_id": self.store_id,
            "camera_id": camera_id,
            "visitor_id": visitor_id,
            "event_type": event_type,
            "timestamp": timestamp.isoformat(),
            "zone_id": zone_id,
            "dwell_ms": dwell_ms,
            "is_staff": is_staff,
            "confidence": round(confidence, 4),
            "metadata": evt_meta,
        }

    def emit_entry(self, camera_id: str, visitor_id: str, timestamp: datetime,
                   is_staff: bool = False, confidence: float = 1.0, is_reentry: bool = False):
        event_type = "REENTRY" if is_reentry else "ENTRY"
        evt = self._build_event(camera_id, visitor_id, event_type, timestamp,
                                 is_staff=is_staff, confidence=confidence)
        self._enqueue(evt)

    def emit_exit(self, camera_id: str, visitor_id: str, timestamp: datetime,
                  is_staff: bool = False, confidence: float = 1.0):
        evt = self._build_event(camera_id, visitor_id, "EXIT", timestamp,
                                 is_staff=is_staff, confidence=confidence)
        self._enqueue(evt)

    def emit_zone_enter(self, camera_id: str, visitor_id: str, zone_id: str,
                        timestamp: datetime, is_staff: bool = False, confidence: float = 1.0):
        self._zone_dwell_start[visitor_id] = (zone_id, timestamp)
        evt = self._build_event(camera_id, visitor_id, "ZONE_ENTER", timestamp,
                                 zone_id=zone_id, is_staff=is_staff, confidence=confidence)
        self._enqueue(evt)

    def emit_zone_exit(self, camera_id: str, visitor_id: str, zone_id: str,
                       timestamp: datetime, dwell_ms: int = 0,
                       is_staff: bool = False, confidence: float = 1.0):
        self._zone_dwell_start.pop(visitor_id, None)
        evt = self._build_event(camera_id, visitor_id, "ZONE_EXIT", timestamp,
                                 zone_id=zone_id, dwell_ms=dwell_ms,
                                 is_staff=is_staff, confidence=confidence)
        self._enqueue(evt)

    def emit_zone_dwell(self, camera_id: str, visitor_id: str, zone_id: str,
                        timestamp: datetime, dwell_ms: int,
                        is_staff: bool = False, confidence: float = 1.0):
        evt = self._build_event(camera_id, visitor_id, "ZONE_DWELL", timestamp,
                                 zone_id=zone_id, dwell_ms=dwell_ms,
                                 is_staff=is_staff, confidence=confidence)
        self._enqueue(evt)

    def emit_billing_queue_join(self, camera_id: str, visitor_id: str,
                                timestamp: datetime, queue_depth: int,
                                is_staff: bool = False, confidence: float = 1.0):
        evt = self._build_event(
            camera_id, visitor_id, "BILLING_QUEUE_JOIN", timestamp,
            zone_id="BILLING_QUEUE", is_staff=is_staff, confidence=confidence,
            metadata={"queue_depth": queue_depth},
        )
        self._enqueue(evt)

    def emit_billing_queue_abandon(self, camera_id: str, visitor_id: str,
                                   timestamp: datetime, is_staff: bool = False,
                                   confidence: float = 1.0):
        evt = self._build_event(
            camera_id, visitor_id, "BILLING_QUEUE_ABANDON", timestamp,
            zone_id="BILLING_QUEUE", is_staff=is_staff, confidence=confidence,
        )
        self._enqueue(evt)

    # ─────────────────────────────────────────
    # Dwell ticker (call every 30s of dwell)
    # ─────────────────────────────────────────

    def tick_dwell(self, camera_id: str, visitor_id: str, current_ts: datetime,
                   is_staff: bool = False, confidence: float = 1.0):
        """Emit ZONE_DWELL if visitor has been in a zone for 30+ seconds."""
        entry = self._zone_dwell_start.get(visitor_id)
        if not entry:
            return
        zone_id, entry_ts = entry
        dwell_ms = int((current_ts - entry_ts).total_seconds() * 1000)
        if dwell_ms >= DWELL_INTERVAL_SECONDS * 1000:
            self.emit_zone_dwell(camera_id, visitor_id, zone_id, current_ts,
                                 dwell_ms=dwell_ms, is_staff=is_staff, confidence=confidence)

    # ─────────────────────────────────────────
    # Buffer + flush
    # ─────────────────────────────────────────

    def _enqueue(self, evt: dict):
        self._buffer.append(evt)
        if len(self._buffer) >= BATCH_SIZE:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer = []

        # Write to JSONL
        if self._output_file:
            with open(self._output_file, "a") as f:
                for evt in batch:
                    f.write(json.dumps(evt) + "\n")

        # POST to API
        try:
            resp = requests.post(
                INGEST_ENDPOINT,
                json={"events": batch},
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
            logger.info(
                "Flushed %d events → API: accepted=%d rejected=%d duplicate=%d",
                len(batch), result.get("accepted", 0),
                result.get("rejected", 0), result.get("duplicate", 0),
            )
        except Exception as exc:
            logger.error("Failed to flush %d events to API: %s", len(batch), exc)
            # Re-queue for next flush attempt
            self._buffer = batch + self._buffer

    def close(self):
        """Flush remaining events at pipeline end."""
        self.flush()
