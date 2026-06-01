"""
audit.py — System Auditor + Self-Monitoring.

Runs continuously alongside the pipeline to detect anomalies that
indicate identity fragmentation, data quality issues, or logic bugs.

Anomaly types:
  SIMULTANEOUS_PRESENCE  — same VIS_ID active in two distant cameras
  STAFF_COUNT_SPIKE      — staff count jumped by > N in 10 seconds
  ZONE_TELEPORT          — visitor appeared in implausible zone sequence
  NEGATIVE_QUEUE         — queue depth < 0 (logic error)
  IDENTITY_EXPLOSION     — new visitor IDs created at > 3× baseline rate
  TRACK_HEALTH_COLLAPSE  — multiple visitors health dropped to 0 simultaneously

Every warning is:
  1. Logged to the Python logger
  2. Pushed to SHARED for GUI display
  3. Written to warnings.jsonl for offline audit
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

from config import cfg

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    SIMULTANEOUS_PRESENCE  = "SIMULTANEOUS_PRESENCE"
    STAFF_COUNT_SPIKE      = "STAFF_COUNT_SPIKE"
    ZONE_TELEPORT          = "ZONE_TELEPORT"
    NEGATIVE_QUEUE         = "NEGATIVE_QUEUE"
    IDENTITY_EXPLOSION     = "IDENTITY_EXPLOSION"
    TRACK_HEALTH_COLLAPSE  = "TRACK_HEALTH_COLLAPSE"


@dataclass
class AuditWarning:
    anomaly_type:  str
    severity:      str           # "INFO" / "WARNING" / "CRITICAL"
    visitor_ids:   List[str]
    description:   str
    timestamp:     str
    camera_ids:    List[str]
    metadata:      dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_type"] = "warning"
        return d


class SystemAuditor:
    """
    Periodically checks pipeline state for anomalies.
    Designed to run every AUDIT_INTERVAL_FRAMES frames from detect.py.
    """

    def __init__(self, output_dir: str = "."):
        self._output_path = Path(output_dir) / "warnings.jsonl"
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._output_path, "a", buffering=1)

        # Rolling history for spike detection
        self._staff_count_history: Deque[Tuple[float, int]] = deque(maxlen=60)
        self._new_id_timestamps:   Deque[float]             = deque(maxlen=200)
        self._baseline_id_rate:    float                    = 0.0
        self._warned_pairs:        Set[Tuple[str, str]]     = set()
        self._warned_teleports:    Dict[str, float]         = {}   # vid → last warn time
        self._broadcast_fn  = None   # set externally to push to GUI

    def set_broadcast(self, fn):
        """Provide a callable(dict) to push warnings to the GUI."""
        self._broadcast_fn = fn

    def close(self):
        if self._file:
            self._file.close()

    # ------------------------------------------------------------------
    # Primary entry point: call once per audit cycle from detect.py
    # ------------------------------------------------------------------
    def check(
        self,
        active_passports:   dict,    # visitor_id → VisitorPassport
        active_cameras:     dict,    # visitor_id → camera_id (current)
        staff_ids:          Set[str],
        queue_depth:        int,
        health_scores:      Dict[str, float],
        new_ids_this_cycle: int,
        wall_time:          float,
    ) -> List[AuditWarning]:
        warnings = []

        # 1. Simultaneous presence in distant cameras
        warnings += self._check_simultaneous_presence(active_cameras, wall_time)

        # 2. Staff count spike
        staff_count = len(staff_ids)
        self._staff_count_history.append((wall_time, staff_count))
        warnings += self._check_staff_spike(wall_time)

        # 3. Negative queue
        if queue_depth < 0:
            warnings.append(self._warn(
                AnomalyType.NEGATIVE_QUEUE, "CRITICAL",
                [], f"Queue depth = {queue_depth} (impossible value)",
                [], wall_time,
            ))

        # 4. Identity explosion
        now = wall_time
        for _ in range(new_ids_this_cycle):
            self._new_id_timestamps.append(now)
        warnings += self._check_identity_rate(wall_time)

        # 5. Track health collapse
        collapsed = [
            vid for vid, score in health_scores.items() if score <= 0.0
        ]
        if len(collapsed) >= 3:
            warnings.append(self._warn(
                AnomalyType.TRACK_HEALTH_COLLAPSE, "WARNING",
                collapsed,
                f"{len(collapsed)} visitors simultaneously reached health=0",
                [], wall_time,
            ))

        # Emit all warnings
        for w in warnings:
            self._emit(w)

        return warnings

    def report_zone_teleport(
        self,
        visitor_id: str,
        zone_from:  str,
        zone_to:    str,
        elapsed_sec: float,
        camera_id:  str,
        wall_time:  float,
    ) -> Optional[AuditWarning]:
        """
        Called by detect.py when StoreGraph says a transition is impossible.
        Rate-limited per visitor (max 1 warning per 30s per visitor).
        """
        last = self._warned_teleports.get(visitor_id, 0.0)
        if wall_time - last < 30.0:
            return None
        self._warned_teleports[visitor_id] = wall_time
        w = self._warn(
            AnomalyType.ZONE_TELEPORT, "WARNING",
            [visitor_id],
            f"Impossible zone jump: {zone_from} → {zone_to} in {elapsed_sec:.1f}s",
            [camera_id], wall_time,
            metadata={"zone_from": zone_from, "zone_to": zone_to,
                      "elapsed_sec": round(elapsed_sec, 2)},
        )
        self._emit(w)
        return w

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_simultaneous_presence(
        self, active_cameras: dict, wall_time: float
    ) -> List[AuditWarning]:
        """Flag same visitor_id active in two distant cameras simultaneously."""
        warnings = []
        # Group by visitor_id
        from store_graph import StoreGraph
        sg = StoreGraph()
        # Build: visitor_id → list of camera_ids it's currently seen in
        vid_cameras: Dict[str, List[str]] = {}
        for vid, cam in active_cameras.items():
            vid_cameras.setdefault(vid, []).append(cam)

        for vid, cams in vid_cameras.items():
            if len(cams) < 2:
                continue
            for i, ca in enumerate(cams):
                for cb in cams[i+1:]:
                    pair = tuple(sorted([ca, cb]))
                    hop_dist = sg.camera_hop_distance(ca, cb)
                    if (hop_dist >= cfg.AUDIT_SIMULTANEOUS_CAM_MIN_DIST
                            and pair not in self._warned_pairs):
                        self._warned_pairs.add(pair)
                        warnings.append(self._warn(
                            AnomalyType.SIMULTANEOUS_PRESENCE, "CRITICAL",
                            [vid],
                            f"{vid} simultaneously active in {ca} and {cb} "
                            f"({hop_dist} hops apart)",
                            list(cams), wall_time,
                            metadata={"hop_distance": hop_dist},
                        ))
        return warnings

    def _check_staff_spike(self, wall_time: float) -> List[AuditWarning]:
        """Detect sudden jump in staff count within a 10-second window."""
        if len(self._staff_count_history) < 2:
            return []
        recent = [(t, c) for t, c in self._staff_count_history
                  if wall_time - t <= 10.0]
        if len(recent) < 2:
            return []
        counts = [c for _, c in recent]
        jump = max(counts) - min(counts)
        if jump > cfg.AUDIT_STAFF_SPIKE_THRESHOLD:
            return [self._warn(
                AnomalyType.STAFF_COUNT_SPIKE, "WARNING",
                [],
                f"Staff count jumped by {jump} in 10s "
                f"(min={min(counts)}, max={max(counts)})",
                [], wall_time,
                metadata={"jump": jump, "counts": counts[-5:]},
            )]
        return []

    def _check_identity_rate(self, wall_time: float) -> List[AuditWarning]:
        """Detect burst of new visitor_id creation (possible fragmentation)."""
        recent = [t for t in self._new_id_timestamps
                  if wall_time - t <= 60.0]
        rate = len(recent) / 60.0  # IDs per second
        if self._baseline_id_rate == 0.0 and len(recent) >= 10:
            self._baseline_id_rate = rate
        if (self._baseline_id_rate > 0
                and rate > self._baseline_id_rate * cfg.AUDIT_IDENTITY_RATE_THRESHOLD):
            return [self._warn(
                AnomalyType.IDENTITY_EXPLOSION, "WARNING",
                [],
                f"New visitor ID rate {rate:.2f}/s is {rate/self._baseline_id_rate:.1f}× "
                f"the baseline {self._baseline_id_rate:.2f}/s — possible fragmentation",
                [], wall_time,
                metadata={"rate": round(rate, 3),
                          "baseline": round(self._baseline_id_rate, 3)},
            )]
        return []

    def _warn(
        self,
        anomaly_type: AnomalyType,
        severity: str,
        visitor_ids: List[str],
        description: str,
        camera_ids: List[str],
        wall_time: float,
        metadata: dict = None,
    ) -> AuditWarning:
        from datetime import datetime, timezone
        ts = datetime.fromtimestamp(wall_time, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        w = AuditWarning(
            anomaly_type = anomaly_type.value,
            severity     = severity,
            visitor_ids  = visitor_ids,
            description  = description,
            timestamp    = ts,
            camera_ids   = camera_ids,
            metadata     = metadata or {},
        )
        lvl = logging.CRITICAL if severity == "CRITICAL" else (
              logging.WARNING  if severity == "WARNING"  else logging.INFO)
        logger.log(lvl, f"[AUDIT] {anomaly_type.value}: {description}")
        return w

    def _emit(self, w: AuditWarning):
        """Write to JSONL and push to GUI."""
        line = json.dumps(w.to_dict())
        if self._file:
            self._file.write(line + "\n")
        if self._broadcast_fn:
            try:
                self._broadcast_fn(w.to_dict())
            except Exception:
                pass
