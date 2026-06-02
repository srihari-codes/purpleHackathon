"""
app/replay.py — Deterministic Replay Engine for the Store Intelligence API.

Two operating modes:

  LIVE mode  — events arrive in real time via POST /events/ingest.
               No special replay logic; the engine is transparent.

  REPLAY mode — events are loaded from a historical .jsonl file (or from
                the in-memory EventStore), sorted by timestamp, and fed
                through the ingestion pipeline in strict chronological order.
                The SessionStore and AuditTimeline are reset first so the
                replay produces a clean, reproducible result.

Goal:
    Allow deterministic debugging of session construction.  Any sequence of
    raw events can be replayed to reproduce the exact sessions, metrics, and
    audit trail that the live system would have produced.

Integration:
    ReplayEngine wraps IngestionPipeline.  It does NOT bypass validation or
    deduplication — every event still goes through the full Layer 2 stack.
    This guarantees that replayed metrics are identical to live metrics for
    the same event log.

Replay speed control:
    - speed=0   → replay as fast as possible (batch, no sleep)
    - speed=1.0 → real-time (sleep proportional to event timestamp gaps)
    - speed=2.0 → 2× real-time, etc.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional

from .ingestion import EventStore, IngestionPipeline
from .sessionizer import SessionStore, Sessionizer
from .audit import AuditTimeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode enum
# ---------------------------------------------------------------------------

class ReplayMode(str, Enum):
    LIVE   = "LIVE"
    REPLAY = "REPLAY"


# ---------------------------------------------------------------------------
# Replay progress callback type
# ---------------------------------------------------------------------------
# Signature: callback(processed: int, total: int, current_event_ts: str)
ReplayProgressCallback = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# Replay Engine
# ---------------------------------------------------------------------------

class ReplayEngine:
    """
    Deterministic replay engine with live/replay mode switching.

    Constructor args:
        event_store    — shared EventStore (Layer 2)
        session_store  — shared SessionStore
        sessionizer    — shared Sessionizer
        audit          — shared AuditTimeline
        mode           — LIVE (default) or REPLAY
        speed          — replay speed multiplier (0 = max speed)

    Usage (REPLAY mode):
        engine = ReplayEngine(event_store, session_store, sessionizer, audit,
                              mode=ReplayMode.REPLAY)
        engine.replay_file("/data/events.jsonl")

    Usage (LIVE mode):
        engine = ReplayEngine(event_store, session_store, sessionizer, audit)
        # Events arrive via POST /events/ingest; engine is a no-op wrapper.
    """

    def __init__(
        self,
        event_store:   EventStore,
        session_store: SessionStore,
        sessionizer:   Sessionizer,
        audit:         AuditTimeline,
        mode:          ReplayMode = ReplayMode.LIVE,
        speed:         float      = 0.0,
    ) -> None:
        self._event_store   = event_store
        self._session_store = session_store
        self._sessionizer   = sessionizer
        self._audit         = audit
        self._mode          = mode
        self._speed         = speed
        self._lock          = threading.Lock()

        # Build a pipeline that routes through the same sessionizer
        self._pipeline = IngestionPipeline(event_store, sessionizer)

        # Replay state
        self._is_replaying  = False
        self._replayed      = 0
        self._replay_total  = 0
        self._replay_thread: Optional[threading.Thread] = None

    # ── mode control ──────────────────────────────────────────────────────

    @property
    def mode(self) -> ReplayMode:
        return self._mode

    def set_mode(self, mode: ReplayMode) -> None:
        with self._lock:
            self._mode = mode
        logger.info("replay_engine_mode_set mode=%s", mode.value)

    @property
    def is_replaying(self) -> bool:
        return self._is_replaying

    # ── replay from file ──────────────────────────────────────────────────

    def replay_file(
        self,
        path: str,
        reset_state: bool = True,
        progress_cb: Optional[ReplayProgressCallback] = None,
        store_id_filter: Optional[str] = None,
    ) -> "ReplayResult":
        """
        Replay events from a JSONL file in timestamp order.

        Args:
            path             — path to a .jsonl events file
            reset_state      — if True, clears EventStore / SessionStore / AuditTimeline first
            progress_cb      — optional callback(processed, total, current_ts)
            store_id_filter  — if set, only replay events for this store

        Returns:
            ReplayResult with summary statistics.
        """
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Replay file not found: {path}")

        raw_events = self._load_jsonl(p, store_id_filter)
        return self._run_replay(raw_events, reset_state, progress_cb, source=str(p))

    def replay_store(
        self,
        store_id: str,
        reset_state: bool = True,
        progress_cb: Optional[ReplayProgressCallback] = None,
    ) -> "ReplayResult":
        """
        Replay events already held in the EventStore (in-memory replay).

        Events are re-ingested in timestamp order.  The EventStore is reset
        first to avoid duplicate-key collisions if reset_state=True.

        Note: re-ingest means the dedup set is also cleared, so the same
        event_ids will be accepted again — this is intentional for replay.
        """
        stored = self._event_store.get_events(store_id)
        raw_events = [s.event.model_dump() for s in stored]
        return self._run_replay(raw_events, reset_state, progress_cb, source=f"store:{store_id}")

    # ── internal replay runner ─────────────────────────────────────────────

    def _run_replay(
        self,
        raw_events: List[Dict[str, Any]],
        reset_state: bool,
        progress_cb: Optional[ReplayProgressCallback],
        source: str,
    ) -> "ReplayResult":
        with self._lock:
            if self._is_replaying:
                raise RuntimeError("A replay is already in progress")
            self._is_replaying  = True
            self._replayed      = 0
            self._replay_total  = len(raw_events)

        logger.info(
            "replay_started source=%s events=%d reset=%s speed=%s",
            source, len(raw_events), reset_state, self._speed,
        )

        try:
            if reset_state:
                self._reset_all_state()

            # Sort by timestamp (deterministic order)
            sorted_events = sorted(
                raw_events,
                key=lambda e: _parse_ts(e.get("timestamp", "")),
            )

            result = ReplayResult(source=source, total=len(sorted_events))
            prev_ts: Optional[datetime] = None

            for idx, raw in enumerate(sorted_events):
                # Optional real-time pacing
                current_ts = _parse_ts(raw.get("timestamp", ""))
                if self._speed > 0 and prev_ts is not None and current_ts > prev_ts:
                    gap_real = (current_ts - prev_ts).total_seconds()
                    sleep_sec = gap_real / self._speed
                    if sleep_sec > 0.001:
                        time.sleep(min(sleep_sec, 5.0))  # cap at 5s to avoid hangs
                prev_ts = current_ts

                # Ingest single event through the full pipeline
                ingest_resp = self._pipeline.ingest_batch([raw])
                result.accepted   += ingest_resp.accepted
                result.duplicates += ingest_resp.duplicates
                result.rejected   += ingest_resp.rejected

                with self._lock:
                    self._replayed = idx + 1

                if progress_cb and (idx % 50 == 0 or idx == len(sorted_events) - 1):
                    progress_cb(idx + 1, len(sorted_events), raw.get("timestamp", ""))

            result.mark_done()
            logger.info(
                "replay_complete source=%s accepted=%d duplicates=%d rejected=%d elapsed=%.2fs",
                source, result.accepted, result.duplicates, result.rejected,
                result.elapsed_sec,
            )
            return result

        finally:
            with self._lock:
                self._is_replaying = False

    # ── state management ──────────────────────────────────────────────────

    def _reset_all_state(self) -> None:
        """
        Clear EventStore, SessionStore, and AuditTimeline for a clean replay.
        The Sessionizer's internal reentry_counts are also reset.
        """
        self._event_store.clear()
        self._session_store.clear()
        self._audit.clear()
        # Reset sessionizer reentry counts
        if hasattr(self._sessionizer, "_reentry_counts"):
            self._sessionizer._reentry_counts.clear()
        logger.info("replay_state_reset")

    # ── JSONL loader ──────────────────────────────────────────────────────

    @staticmethod
    def _load_jsonl(
        path: pathlib.Path,
        store_id_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        events = []
        skipped = 0
        errors  = 0
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("replay_jsonl_parse_error line=%d: %s", lineno, exc)
                    errors += 1
                    continue
                if store_id_filter and obj.get("store_id") != store_id_filter:
                    skipped += 1
                    continue
                events.append(obj)
        logger.info(
            "replay_file_loaded path=%s events=%d skipped=%d errors=%d",
            path, len(events), skipped, errors,
        )
        return events

    # ── progress query ─────────────────────────────────────────────────────

    def progress(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "mode":         self._mode.value,
                "is_replaying": self._is_replaying,
                "replayed":     self._replayed,
                "total":        self._replay_total,
                "pct":          round(100 * self._replayed / max(1, self._replay_total), 1),
            }


# ---------------------------------------------------------------------------
# Replay Result
# ---------------------------------------------------------------------------

class ReplayResult:
    """Summary of a completed replay run."""

    def __init__(self, source: str, total: int) -> None:
        self.source    = source
        self.total     = total
        self.accepted  = 0
        self.duplicates = 0
        self.rejected  = 0
        self._started  = time.monotonic()
        self.elapsed_sec = 0.0

    def mark_done(self) -> None:
        self.elapsed_sec = time.monotonic() - self._started

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source":      self.source,
            "total":       self.total,
            "accepted":    self.accepted,
            "duplicates":  self.duplicates,
            "rejected":    self.rejected,
            "elapsed_sec": round(self.elapsed_sec, 3),
        }

    def __repr__(self) -> str:
        return (
            f"ReplayResult(source={self.source!r}, total={self.total}, "
            f"accepted={self.accepted}, rejected={self.rejected}, "
            f"elapsed={self.elapsed_sec:.2f}s)"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp, returning epoch 0 on failure (sort-safe)."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
