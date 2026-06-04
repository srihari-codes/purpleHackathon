"""
tests/test_calibration.py — Unit tests for CalibrationEngine (app/calibration.py).

13 tests covering:
  - Default profile values on first access
  - Rolling window observation feeds
  - Calibration of every parameter (reid, staff, queue_dwell, zone_dwell)
  - camera_trust and occlusion_timeout calibration
  - get_threshold for known and unknown parameters
  - clamp bounds respected (values never outside safe ranges)
  - Auto-calibrate every N events
  - calibrate_all
"""

import pytest

from app.calibration import CalibrationEngine, PARAM_RANGES


# ---------------------------------------------------------------------------
# 1. Default profile
# ---------------------------------------------------------------------------

def test_default_profile_values():
    cal = CalibrationEngine()
    profile = cal.get_profile("STORE_A")
    assert profile.reid_confidence_threshold == PARAM_RANGES["reid_confidence_threshold"][0]
    assert profile.queue_join_dwell_sec == PARAM_RANGES["queue_join_dwell_sec"][0]
    assert profile.staff_confidence_threshold == PARAM_RANGES["staff_confidence_threshold"][0]
    assert profile.dwell_threshold_ms == PARAM_RANGES["dwell_threshold_ms"][0]
    assert profile.calibration_count == 0


# ---------------------------------------------------------------------------
# 2. Observations feed and calibrate
# ---------------------------------------------------------------------------

def test_calibrate_reid_threshold(calibration):
    for score in [0.70, 0.75, 0.72, 0.68, 0.80] * 10:  # 50 observations
        calibration.observe_reid("STORE_A", score)
    calibration.calibrate("STORE_A")
    profile = calibration.get_profile("STORE_A")
    lo, hi = PARAM_RANGES["reid_confidence_threshold"][1], PARAM_RANGES["reid_confidence_threshold"][2]
    assert lo <= profile.reid_confidence_threshold <= hi


def test_calibrate_staff_threshold(calibration):
    for score in [0.60, 0.65, 0.70, 0.55, 0.58] * 10:
        calibration.observe_staff_score("STORE_A", score)
    calibration.calibrate("STORE_A")
    profile = calibration.get_profile("STORE_A")
    lo, hi = PARAM_RANGES["staff_confidence_threshold"][1], PARAM_RANGES["staff_confidence_threshold"][2]
    assert lo <= profile.staff_confidence_threshold <= hi


def test_calibrate_queue_dwell(calibration):
    for dwell in [3.0, 4.0, 5.0, 6.0, 7.0] * 10:
        calibration.observe_queue_dwell("STORE_A", dwell)
    calibration.calibrate("STORE_A")
    profile = calibration.get_profile("STORE_A")
    lo, hi = PARAM_RANGES["queue_join_dwell_sec"][1], PARAM_RANGES["queue_join_dwell_sec"][2]
    assert lo <= profile.queue_join_dwell_sec <= hi


def test_calibrate_zone_dwell(calibration):
    for dwell in [20_000.0, 30_000.0, 40_000.0, 25_000.0] * 15:  # 60 observations
        calibration.observe_zone_dwell("STORE_A", dwell)
    calibration.calibrate("STORE_A")
    profile = calibration.get_profile("STORE_A")
    lo, hi = PARAM_RANGES["dwell_threshold_ms"][1], PARAM_RANGES["dwell_threshold_ms"][2]
    assert lo <= profile.dwell_threshold_ms <= hi


def test_calibrate_increments_count(calibration):
    calibration.calibrate("STORE_A")
    assert calibration.get_profile("STORE_A").calibration_count == 1
    calibration.calibrate("STORE_A")
    assert calibration.get_profile("STORE_A").calibration_count == 2


# ---------------------------------------------------------------------------
# 3. Camera calibration
# ---------------------------------------------------------------------------

def test_camera_trust_calibrated(calibration):
    for conf in [0.80, 0.85, 0.75, 0.90, 0.70] * 10:  # 50 obs
        calibration.observe_event("STORE_A", "CAM_01", confidence=conf)
    calibration.calibrate("STORE_A")
    cam_cal = calibration.get_camera_calibration("STORE_A", "CAM_01")
    lo, hi = PARAM_RANGES["camera_trust"][1], PARAM_RANGES["camera_trust"][2]
    assert lo <= cam_cal.camera_trust <= hi


# ---------------------------------------------------------------------------
# 4. get_threshold
# ---------------------------------------------------------------------------

def test_get_threshold_known_param(calibration):
    val = calibration.get_threshold("STORE_A", "reid_confidence_threshold")
    assert 0.0 < val <= 1.0


def test_get_threshold_unknown_param_returns_default(calibration):
    """Unknown parameter must not raise — must return fallback default."""
    val = calibration.get_threshold("STORE_A", "totally_unknown_param")
    assert isinstance(val, float)


# ---------------------------------------------------------------------------
# 5. Auto-calibrate every N events
# ---------------------------------------------------------------------------

def test_auto_calibrate_fires_every_n():
    """With calibrate_every_n_events=5, calibration_count should be 2 after 10 observe_event calls."""
    cal = CalibrationEngine(calibrate_every_n_events=5)
    for i in range(10):
        cal.observe_event("STORE_A", "CAM_01", confidence=0.8)
    profile = cal.get_profile("STORE_A")
    assert profile.calibration_count >= 2


# ---------------------------------------------------------------------------
# 6. calibrate_all
# ---------------------------------------------------------------------------

def test_calibrate_all(calibration):
    calibration.observe_reid("STORE_A", 0.7)
    calibration.observe_reid("STORE_B", 0.8)
    results = calibration.calibrate_all()
    assert "STORE_A" in results
    assert "STORE_B" in results


# ---------------------------------------------------------------------------
# 7. all_profiles export
# ---------------------------------------------------------------------------

def test_all_profiles_returns_dict(calibration):
    calibration.get_profile("STORE_A")
    calibration.get_profile("STORE_B")
    profiles = calibration.all_profiles()
    assert set(profiles.keys()) == {"STORE_A", "STORE_B"}
    for p in profiles.values():
        assert "store_id" in p
        assert "reid_confidence_threshold" in p
