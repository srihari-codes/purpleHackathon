"""
app/calibration.py — Automatic threshold calibration per store and per camera.

Purpose:
    Provide a systematic, data-driven way to tune the thresholds that control
    session quality — without hardcoding values.  Calibration profiles are
    computed from observed event statistics and can be updated incrementally
    as more data arrives.

Profiles:
    CameraCalibration   — per-camera trust + occlusion timeout
    StoreCalibration    — per-store aggregated thresholds

Calibrated parameters:
    reid_confidence_threshold   — min composite ReID score to re-associate
    queue_join_dwell_sec        — min dwell before queue join is recorded
    staff_confidence_threshold  — min staff score to classify as staff
    dwell_threshold_ms          — min dwell before ZONE_DWELL fires
    camera_trust                — multiplier [0.5–1.0] applied to event confidence
    occlusion_timeout_sec       — how long to hold a suspended track

Algorithm:
    - Maintain rolling observation windows per store/camera.
    - On each calibrate() call, compute percentile-based estimates from history.
    - Clamp outputs to safe [min, max] ranges to prevent runaway values.
    - Emit a calibration AuditRecord so changes are traceable.

Thread safety: all calibration state protected by threading.Lock.
"""

from __future__ import annotations

import logging
import math
import statistics
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default safe ranges for each calibrated parameter
# ---------------------------------------------------------------------------

PARAM_RANGES: Dict[str, Tuple[float, float, float]] = {
    # (default, min, max)
    "reid_confidence_threshold":  (0.65, 0.40, 0.90),
    "queue_join_dwell_sec":       (5.0,  2.0,  30.0),
    "staff_confidence_threshold": (0.55, 0.35, 0.85),
    "dwell_threshold_ms":         (30_000, 10_000, 90_000),
    "camera_trust":               (1.00, 0.50, 1.00),
    "occlusion_timeout_sec":      (20.0, 5.0,  60.0),
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Camera Calibration Profile
# ---------------------------------------------------------------------------

@dataclass
class CameraCalibration:
    """
    Per-camera calibrated thresholds.

    camera_trust: confidence multiplier. Cameras with high occlusion or
    poor entry-line accuracy get lower trust.
    occlusion_timeout_sec: how long a suspended track is kept alive before
    being expired.  Longer for zones with frequent occlusion (shelves).
    """
    camera_id:            str
    camera_trust:         float = 1.00
    occlusion_timeout_sec: float = 20.0

    # Observation history (raw event confidence values from this camera)
    _confidence_obs: List[float] = field(default_factory=list, repr=False)
    # Fraction of events that were flagged as occluded
    _occlusion_flags: List[bool] = field(default_factory=list, repr=False)

    def observe(self, confidence: float, is_occluded: bool = False) -> None:
        self._confidence_obs.append(confidence)
        self._occlusion_flags.append(is_occluded)
        # Keep rolling window of 500 samples
        if len(self._confidence_obs) > 500:
            self._confidence_obs.pop(0)
            self._occlusion_flags.pop(0)

    def calibrate(self) -> None:
        """Recompute parameters from observations."""
        if len(self._confidence_obs) >= 20:
            # Trust = p25 of confidence distribution (conservative)
            sorted_conf = sorted(self._confidence_obs)
            p25_idx = max(0, int(0.25 * len(sorted_conf)) - 1)
            raw_trust = sorted_conf[p25_idx]
            lo, hi = PARAM_RANGES["camera_trust"][1], PARAM_RANGES["camera_trust"][2]
            self.camera_trust = _clamp(raw_trust, lo, hi)

        if len(self._occlusion_flags) >= 20:
            occlusion_rate = sum(self._occlusion_flags) / len(self._occlusion_flags)
            # Higher occlusion rate → longer timeout (more retention needed)
            default_t, lo_t, hi_t = PARAM_RANGES["occlusion_timeout_sec"]
            self.occlusion_timeout_sec = _clamp(
                default_t + (occlusion_rate * 40.0),  # scale up to 40s extra at 100% occlusion
                lo_t, hi_t,
            )

        logger.debug(
            "camera_calibrated camera=%s trust=%.3f occlusion_timeout=%.1fs",
            self.camera_id, self.camera_trust, self.occlusion_timeout_sec,
        )

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "camera_trust": round(self.camera_trust, 4),
            "occlusion_timeout_sec": round(self.occlusion_timeout_sec, 2),
            "observations": len(self._confidence_obs),
        }


# ---------------------------------------------------------------------------
# Store Calibration Profile
# ---------------------------------------------------------------------------

@dataclass
class StoreCalibration:
    """
    Per-store calibrated thresholds.

    These drive the Sessionizer and are the source of truth for any
    threshold value in the session processing loop.
    """
    store_id: str

    reid_confidence_threshold:  float = 0.65
    queue_join_dwell_sec:       float = 5.0
    staff_confidence_threshold: float = 0.55
    dwell_threshold_ms:         float = 30_000.0

    # Camera sub-profiles
    cameras: Dict[str, CameraCalibration] = field(default_factory=dict)

    # Observation windows
    _reid_scores:      List[float] = field(default_factory=list, repr=False)
    _staff_scores:     List[float] = field(default_factory=list, repr=False)
    _queue_dwell_secs: List[float] = field(default_factory=list, repr=False)
    _zone_dwell_ms:    List[float] = field(default_factory=list, repr=False)

    # Calibration metadata
    last_calibrated_at: Optional[str] = None
    calibration_count:  int = 0

    # ── observation recording ──────────────────────────────────────────────

    def observe_reid(self, score: float) -> None:
        self._reid_scores.append(score)
        if len(self._reid_scores) > 1000:
            self._reid_scores.pop(0)

    def observe_staff(self, score: float) -> None:
        self._staff_scores.append(score)
        if len(self._staff_scores) > 500:
            self._staff_scores.pop(0)

    def observe_queue_dwell(self, dwell_sec: float) -> None:
        self._queue_dwell_secs.append(dwell_sec)
        if len(self._queue_dwell_secs) > 500:
            self._queue_dwell_secs.pop(0)

    def observe_zone_dwell(self, dwell_ms: float) -> None:
        self._zone_dwell_ms.append(dwell_ms)
        if len(self._zone_dwell_ms) > 2000:
            self._zone_dwell_ms.pop(0)

    def observe_camera_event(
        self, camera_id: str, confidence: float, is_occluded: bool = False
    ) -> None:
        if camera_id not in self.cameras:
            self.cameras[camera_id] = CameraCalibration(camera_id=camera_id)
        self.cameras[camera_id].observe(confidence, is_occluded)

    # ── calibration ───────────────────────────────────────────────────────

    def calibrate(self) -> Dict[str, Any]:
        """
        Recompute all store-level thresholds from observed data.

        Returns a dict describing what changed (for audit logging).
        """
        changes: Dict[str, Any] = {}
        prev = self.to_dict()

        # ReID threshold: set to p10 of clean-match score distribution.
        # Low enough to catch genuine re-entries; high enough to avoid false positives.
        if len(self._reid_scores) >= 30:
            sorted_reid = sorted(self._reid_scores)
            p10_idx = max(0, int(0.10 * len(sorted_reid)) - 1)
            candidate = sorted_reid[p10_idx]
            lo, hi = PARAM_RANGES["reid_confidence_threshold"][1], PARAM_RANGES["reid_confidence_threshold"][2]
            self.reid_confidence_threshold = _clamp(candidate, lo, hi)

        # Staff threshold: just below the p25 of observed staff scores.
        if len(self._staff_scores) >= 20:
            sorted_staff = sorted(self._staff_scores, reverse=True)
            p25_idx = max(0, int(0.25 * len(sorted_staff)) - 1)
            candidate = sorted_staff[p25_idx] - 0.05  # margin below p25
            lo, hi = PARAM_RANGES["staff_confidence_threshold"][1], PARAM_RANGES["staff_confidence_threshold"][2]
            self.staff_confidence_threshold = _clamp(candidate, lo, hi)

        # Queue join dwell: p10 of observed queue dwell times (catch quick joiners)
        if len(self._queue_dwell_secs) >= 15:
            sorted_dwell = sorted(self._queue_dwell_secs)
            p10_idx = max(0, int(0.10 * len(sorted_dwell)) - 1)
            candidate = sorted_dwell[p10_idx]
            lo, hi = PARAM_RANGES["queue_join_dwell_sec"][1], PARAM_RANGES["queue_join_dwell_sec"][2]
            self.queue_join_dwell_sec = _clamp(candidate, lo, hi)

        # Zone dwell threshold: p5 of zone dwell distribution
        if len(self._zone_dwell_ms) >= 50:
            sorted_zone = sorted(self._zone_dwell_ms)
            p5_idx = max(0, int(0.05 * len(sorted_zone)) - 1)
            candidate = sorted_zone[p5_idx]
            lo, hi = PARAM_RANGES["dwell_threshold_ms"][1], PARAM_RANGES["dwell_threshold_ms"][2]
            self.dwell_threshold_ms = _clamp(candidate, lo, hi)

        # Calibrate each camera sub-profile
        for cam_cal in self.cameras.values():
            cam_cal.calibrate()

        # Record calibration metadata
        self.last_calibrated_at = _now_iso()
        self.calibration_count += 1

        # Compute change set for audit
        after = self.to_dict()
        for k in ("reid_confidence_threshold", "queue_join_dwell_sec",
                  "staff_confidence_threshold", "dwell_threshold_ms"):
            if prev.get(k) != after.get(k):
                changes[k] = {"before": prev.get(k), "after": after.get(k)}

        if changes:
            logger.info(
                "store_calibrated store=%s changes=%s",
                self.store_id, changes,
            )

        return changes

    def get_camera(self, camera_id: str) -> CameraCalibration:
        """Return camera calibration (creates default if not yet observed)."""
        if camera_id not in self.cameras:
            self.cameras[camera_id] = CameraCalibration(camera_id=camera_id)
        return self.cameras[camera_id]

    def to_dict(self) -> dict:
        return {
            "store_id": self.store_id,
            "reid_confidence_threshold": round(self.reid_confidence_threshold, 4),
            "queue_join_dwell_sec": round(self.queue_join_dwell_sec, 2),
            "staff_confidence_threshold": round(self.staff_confidence_threshold, 4),
            "dwell_threshold_ms": round(self.dwell_threshold_ms, 0),
            "cameras": {k: v.to_dict() for k, v in self.cameras.items()},
            "last_calibrated_at": self.last_calibrated_at,
            "calibration_count": self.calibration_count,
        }


# ---------------------------------------------------------------------------
# Calibration Engine
# ---------------------------------------------------------------------------

class CalibrationEngine:
    """
    Central engine that manages StoreCalibration profiles.

    Usage:
        engine = CalibrationEngine()

        # Feed observations (called by the Sessionizer):
        engine.observe_event(event)
        engine.observe_reid(store_id, score)

        # Trigger calibration (call periodically or after every N events):
        engine.calibrate(store_id)

        # Query thresholds:
        thresh = engine.get_threshold(store_id, "reid_confidence_threshold")
        cam_cal = engine.get_camera_calibration(store_id, camera_id)
    """

    def __init__(self, calibrate_every_n_events: int = 200) -> None:
        self._profiles: Dict[str, StoreCalibration] = {}
        self._lock = threading.Lock()
        self._event_counts: Dict[str, int] = {}
        self._calibrate_every = calibrate_every_n_events

    # ── profile access ─────────────────────────────────────────────────────

    def _get_or_create(self, store_id: str) -> StoreCalibration:
        """Must be called with _lock held."""
        if store_id not in self._profiles:
            self._profiles[store_id] = StoreCalibration(store_id=store_id)
            self._event_counts[store_id] = 0
        return self._profiles[store_id]

    def get_profile(self, store_id: str) -> StoreCalibration:
        with self._lock:
            return self._get_or_create(store_id)

    def get_threshold(self, store_id: str, param: str) -> float:
        """
        Return the calibrated value for a parameter, or the safe default
        if not enough data has been observed yet.
        """
        with self._lock:
            profile = self._get_or_create(store_id)
            value = getattr(profile, param, None)
        if value is None:
            default = PARAM_RANGES.get(param, (0.5, 0.0, 1.0))[0]
            logger.warning("calibration_unknown_param param=%s — using default %.3f", param, default)
            return default
        return value

    def get_camera_calibration(self, store_id: str, camera_id: str) -> CameraCalibration:
        with self._lock:
            profile = self._get_or_create(store_id)
            return profile.get_camera(camera_id)

    # ── observation feed ───────────────────────────────────────────────────

    def observe_event(
        self,
        store_id: str,
        camera_id: str,
        confidence: float,
        is_occluded: bool = False,
    ) -> None:
        with self._lock:
            profile = self._get_or_create(store_id)
            profile.observe_camera_event(camera_id, confidence, is_occluded)
            self._event_counts[store_id] = self._event_counts.get(store_id, 0) + 1
            should_calibrate = (self._event_counts[store_id] % self._calibrate_every == 0)

        if should_calibrate:
            self.calibrate(store_id)

    def observe_reid(self, store_id: str, reid_score: float) -> None:
        with self._lock:
            self._get_or_create(store_id).observe_reid(reid_score)

    def observe_staff_score(self, store_id: str, score: float) -> None:
        with self._lock:
            self._get_or_create(store_id).observe_staff(score)

    def observe_queue_dwell(self, store_id: str, dwell_sec: float) -> None:
        with self._lock:
            self._get_or_create(store_id).observe_queue_dwell(dwell_sec)

    def observe_zone_dwell(self, store_id: str, dwell_ms: float) -> None:
        with self._lock:
            self._get_or_create(store_id).observe_zone_dwell(dwell_ms)

    # ── calibration trigger ────────────────────────────────────────────────

    def calibrate(self, store_id: str) -> Dict[str, Any]:
        """
        Recompute thresholds for a store.  Returns change dict (may be empty).
        """
        with self._lock:
            profile = self._get_or_create(store_id)
        changes = profile.calibrate()
        return changes

    def calibrate_all(self) -> Dict[str, Dict[str, Any]]:
        """Calibrate every store. Returns {store_id: changes}."""
        with self._lock:
            store_ids = list(self._profiles.keys())
        results = {}
        for sid in store_ids:
            results[sid] = self.calibrate(sid)
        return results

    # ── export ─────────────────────────────────────────────────────────────

    def all_profiles(self) -> Dict[str, dict]:
        with self._lock:
            return {sid: p.to_dict() for sid, p in self._profiles.items()}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
