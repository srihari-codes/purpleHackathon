"""
camera_reputation.py — Camera Reputation System.

Not all cameras are equally trustworthy. Cameras in occluded areas produce
noisier detections; entry cameras are highly reliable. This module provides
per-camera confidence modifiers that flow into the confidence pipeline.

Static priors are defined in config.py (CAMERA_REPUTATION_PRIORS_BY_ROLE).
Dynamic adjustments learn from observed detection quality over time.

Camera specializations are determined by the role assigned in the wizard:
  "entry"   — high reliability for entry/exit events
  "billing" — high reliability for queue events
  "floor"   — moderate reliability, moderate occlusion
  "godown"  — high occlusion, low customer visibility
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera reputation profile
# ---------------------------------------------------------------------------
@dataclass
class CameraReputationProfile:
    """Per-camera reputation metrics."""
    camera_id:            str
    occlusion_risk:       float    # 0–1: probability of occlusion events
    detection_reliability:float    # 0–1: how often detections are correct
    specialization:       str      # "entry" / "billing" / "floor" / "staff"

    # Dynamic statistics (updated at runtime)
    total_detections:     int   = 0
    successful_reids:     int   = 0
    failed_reids:         int   = 0
    track_losses:         int   = 0
    false_positive_est:   int   = 0   # estimated false positives (track < 3 frames)

    # Effective modifier (cached)
    _confidence_modifier: float = 1.0

    def compute_modifier(self) -> float:
        """
        Compute the confidence modifier for detections from this camera.
        Range: [CAMERA_REP_MIN_MODIFIER, 1.0]

        Higher reliability → closer to 1.0
        Higher occlusion_risk → reduces the modifier
        """
        base = self.detection_reliability
        # Occlusion penalty: high occlusion cameras get a small reduction
        occlusion_penalty = self.occlusion_risk * 0.15
        # Dynamic adjustment from observed performance
        if self.total_detections > 50:
            success_rate = self.successful_reids / max(1, self.successful_reids + self.failed_reids)
            dynamic_adj = 0.1 * (success_rate - 0.5)  # ±0.05 adjustment
        else:
            dynamic_adj = 0.0

        modifier = base - occlusion_penalty + dynamic_adj
        modifier = max(cfg.CAMERA_REP_MIN_MODIFIER, min(1.0, modifier))
        self._confidence_modifier = round(modifier, 4)
        return self._confidence_modifier

    @property
    def trust_level(self) -> str:
        """Human-readable trust classification."""
        m = self._confidence_modifier
        if m >= 0.90:
            return "HIGH"
        elif m >= 0.80:
            return "MEDIUM"
        else:
            return "LOW"

    def to_dict(self) -> dict:
        return {
            "camera_id":             self.camera_id,
            "occlusion_risk":        round(self.occlusion_risk, 3),
            "detection_reliability": round(self.detection_reliability, 3),
            "specialization":        self.specialization,
            "confidence_modifier":   self._confidence_modifier,
            "trust_level":           self.trust_level,
            "total_detections":      self.total_detections,
            "successful_reids":      self.successful_reids,
            "failed_reids":          self.failed_reids,
            "track_losses":          self.track_losses,
        }


# ---------------------------------------------------------------------------
# Role inference helper
# ---------------------------------------------------------------------------

def _role_from_camera_id(camera_id: str) -> str:
    """
    Infer role from camera ID using naming conventions.
    e.g. CAM_ENTRY_03 → entry, CAM_BILLING_05 → billing
    Falls back to 'floor' for unknown patterns.
    """
    cam_upper = camera_id.upper()
    if "ENTRY" in cam_upper:
        return "entry"
    if "BILLING" in cam_upper:
        return "billing"
    if "GODOWN" in cam_upper or "STAFF" in cam_upper:
        return "godown"
    return "floor"


# ---------------------------------------------------------------------------
# Camera Reputation Manager
# ---------------------------------------------------------------------------
class CameraReputation:
    """
    Manages reputation profiles for all cameras.

    Usage:
        rep = CameraReputation()
        modifier = rep.confidence_modifier("CAM_FLOOR_01")  # → 0.83
        rep.on_successful_reid("CAM_FLOOR_01")   # improves reputation
        rep.on_track_loss("CAM_GODOWN_04")        # degrades reputation
    """

    def __init__(self):
        self._profiles: Dict[str, CameraReputationProfile] = {}
        self._init_from_priors()

    def _init_from_priors(self):
        """Initialize profiles from static priors in config."""
        for cam_id, priors in cfg.CAMERA_REPUTATION_PRIORS.items():
            profile = CameraReputationProfile(
                camera_id             = cam_id,
                occlusion_risk        = priors.get("occlusion_risk", 0.30),
                detection_reliability = priors.get("reliability", 0.80),
                specialization        = priors.get("spec", "floor"),
            )
            profile.compute_modifier()
            self._profiles[cam_id] = profile

    def seed_from_role_map(self, role_map: dict) -> None:
        """
        Seed reputation priors from a wizard-supplied role map.

        role_map format: {camera_id: role}  e.g. {"CAM_ENTRY_03": "entry"}
        Called once at pipeline startup after the wizard session is loaded.
        """
        role_priors = cfg.CAMERA_REPUTATION_PRIORS_BY_ROLE
        for cam_id, role in role_map.items():
            priors = role_priors.get(role, role_priors.get("floor", {}))
            profile = CameraReputationProfile(
                camera_id             = cam_id,
                occlusion_risk        = priors.get("occlusion_risk", 0.30),
                detection_reliability = priors.get("reliability", 0.80),
                specialization        = priors.get("spec", "floor"),
            )
            profile.compute_modifier()
            self._profiles[cam_id] = profile
            logger.info(
                f"CameraReputation seeded: {cam_id} role={role} "
                f"reliability={profile.detection_reliability:.2f} "
                f"modifier={profile._confidence_modifier:.3f}"
            )

    def _get(self, camera_id: str) -> CameraReputationProfile:
        """Get or create profile for a camera, using role-based priors where available."""
        if camera_id not in self._profiles:
            # Try to infer role from camera_id suffix (e.g. "CAM_ENTRY_03" -> "entry")
            role = _role_from_camera_id(camera_id)
            priors = cfg.CAMERA_REPUTATION_PRIORS_BY_ROLE.get(
                role, cfg.CAMERA_REPUTATION_PRIORS_BY_ROLE.get("floor", {})
            )
            self._profiles[camera_id] = CameraReputationProfile(
                camera_id=camera_id,
                occlusion_risk=priors.get("occlusion_risk", 0.30),
                detection_reliability=priors.get("reliability", 0.80),
                specialization=priors.get("spec", "floor"),
            )
            self._profiles[camera_id].compute_modifier()
        return self._profiles[camera_id]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def confidence_modifier(self, camera_id: str) -> float:
        """
        Returns a multiplier [0.7, 1.0] to apply to detection confidence
        from this camera. Low-reliability cameras reduce confidence.
        """
        return self._get(camera_id)._confidence_modifier

    def get_profile(self, camera_id: str) -> CameraReputationProfile:
        """Full reputation profile for a camera."""
        return self._get(camera_id)

    def occlusion_risk(self, camera_id: str) -> float:
        """Probability of occlusion events [0, 1]."""
        return self._get(camera_id).occlusion_risk

    def specialization(self, camera_id: str) -> str:
        """Camera specialization: 'entry', 'billing', 'floor', 'staff'."""
        return self._get(camera_id).specialization

    # ------------------------------------------------------------------
    # Dynamic updates (called from detect.py pipeline)
    # ------------------------------------------------------------------

    def on_detection(self, camera_id: str):
        """Record a detection from this camera."""
        self._get(camera_id).total_detections += 1

    def on_successful_reid(self, camera_id: str):
        """Record a successful re-identification from this camera."""
        profile = self._get(camera_id)
        profile.successful_reids += 1
        # Slowly improve reliability
        profile.detection_reliability = min(1.0,
            profile.detection_reliability + cfg.CAMERA_REP_LEARNING_RATE
        )
        profile.compute_modifier()

    def on_failed_reid(self, camera_id: str):
        """Record a failed re-identification attempt from this camera."""
        profile = self._get(camera_id)
        profile.failed_reids += 1
        # Slowly degrade reliability
        profile.detection_reliability = max(0.50,
            profile.detection_reliability - cfg.CAMERA_REP_LEARNING_RATE * 0.5
        )
        profile.compute_modifier()

    def on_track_loss(self, camera_id: str):
        """Record a track loss from this camera."""
        profile = self._get(camera_id)
        profile.track_losses += 1
        # Increase occlusion risk estimate
        profile.occlusion_risk = min(0.90,
            profile.occlusion_risk + cfg.CAMERA_REP_LEARNING_RATE * 0.3
        )
        profile.compute_modifier()

    def on_short_track(self, camera_id: str):
        """Record a very short track (< 3 frames) — likely false positive."""
        profile = self._get(camera_id)
        profile.false_positive_est += 1

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def all_profiles(self) -> Dict[str, dict]:
        """All camera reputation profiles as dicts."""
        return {cam_id: p.to_dict() for cam_id, p in self._profiles.items()}

    def summary(self, camera_id: str) -> dict:
        """Single camera summary."""
        return self._get(camera_id).to_dict()
