"""
detect.py — Main detection pipeline orchestrator.

Reads all camera clips in sync, runs YOLO + ByteTrack + Re-ID,
classifies staff, detects zone visits, entry/exit, and billing queue events.
Emits structured events to JSONL + WebSocket GUI.

Usage:
    python detect.py --store_id STORE_BLR_002 --clips_dir /data/clips
                     [--output /data/events.jsonl]
                     [--gui_port 8080]
                     [--speed 1.0]          # playback speed multiplier
                     [--cam3_start_iso ...]  # override entry cam start time

Camera filename convention (case-insensitive):
    CAM 1.mp4 or cam1.mp4 → CAM_FLOOR_01
    CAM 2.mp4 or cam2.mp4 → CAM_FLOOR_02
    CAM 3.mp4 or cam3.mp4 → CAM_ENTRY_03
    CAM 4.mp4 or cam4.mp4 → CAM_GODOWN_04
    CAM 5.mp4 or cam5.mp4 → CAM_BILLING_05
    (also accepts .mp3 extension which ffmpeg can decode)
"""

import argparse
import logging
import os
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import cv2
import numpy as np

# Local modules
from config     import cfg
from events     import EventEmitter, make_timestamp
from tracker    import VisitorIdentityManager
from staff      import StaffBehaviourTracker
from zones      import get_zones_for_camera, zone_for_point, ENTRY_LINE_NORM, get_entry_line_for_camera
from entry_exit import EntryExitDetector
from billing_queue import QueueTracker
from behavior   import BehaviorStateMachine
from gui_server import run_server, SHARED
from memory     import VisitorMemoryManager
from audit      import SystemAuditor
from camera_reputation import CameraReputation

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
)
logger = logging.getLogger("detect")

# Camera ID convention note:
#   Camera IDs are now fully dynamic and assigned by the wizard onboarding flow.
#   Roles (entry / billing / floor / godown) determine pipeline behaviour.
#   The mapping from role → camera_id is passed in via camera_role_map.

# ---------------------------------------------------------------------------
# YOLO + ByteTrack wrapper
# ---------------------------------------------------------------------------

class YOLODetector:
    """
    Wraps ultralytics YOLOv8 for person detection + ByteTrack tracking.
    Falls back to MOG2 background subtraction if ultralytics unavailable.
    """

    def __init__(self, device: str = "auto"):
        self._model  = None
        self._use_yolo = False
        self._device   = device
        self._try_load_yolo(device)

    def _try_load_yolo(self, device: str):
        try:
            import torch
            from ultralytics import YOLO

            # ── Device resolution ─────────────────────────────────────────
            if device == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                dev = device

            # Rich diagnostic log so the operator can confirm GPU is active
            cuda_avail = torch.cuda.is_available()
            gpu_name   = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
            logger.info(
                f"YOLODetector: torch={torch.__version__} | "
                f"CUDA available={cuda_avail} | GPU={gpu_name} | "
                f"resolved device='{dev}'"
            )
            if device == "auto" and not cuda_avail:
                logger.warning(
                    "YOLODetector: CUDA not available — running on CPU. "
                    "If you expected GPU, check that the container was built "
                    "with USE_CUDA=1 and is started with --gpus all / nvidia runtime."
                )

            model_path = cfg.YOLO_MODEL
            self._model = YOLO(model_path)
            self._model.to(dev)
            self._resolved_device = dev
            self._use_yolo = True
            logger.info(f"YOLODetector: YOLO model loaded → running on '{dev}'")
        except Exception as e:
            logger.warning(f"YOLODetector: YOLO unavailable ({e}). "
                           f"Falling back to MOG2 background subtraction.")
            self._use_yolo = False
            self._resolved_device = "cpu"
            self._bg_sub = {}    # camera_id → MOG2 subtractor


    def detect_and_track(
        self, frame_bgr: np.ndarray, camera_id: str
    ) -> List[Tuple[int, Tuple[float, float, float, float], float]]:
        """
        Returns list of (track_id, bbox_xyxy, confidence).
        bbox_xyxy is in pixel coordinates.
        Only returns person-class detections.
        """
        if self._use_yolo:
            return self._yolo_track(frame_bgr, camera_id)
        else:
            return self._mog2_detect(frame_bgr, camera_id)

    def _yolo_track(self, frame_bgr, camera_id):
        try:
            results = self._model.track(
                frame_bgr,
                persist=True,
                classes=[0],        # class 0 = person
                conf=0.35,
                iou=0.45,
                verbose=False,
                device=self._resolved_device,
                tracker="bytetrack.yaml",
            )
            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    tid  = int(boxes.id[i]) if boxes.id is not None else (i + 1)
                    xyxy = tuple(float(v) for v in boxes.xyxy[i])
                    detections.append((tid, xyxy, conf))
            return detections
        except Exception as e:
            logger.debug(f"YOLO track error: {e}")
            return []

    def _mog2_detect(self, frame_bgr, camera_id):
        """
        Simple fallback: MOG2 foreground mask → connected components → bboxes.
        track_id = component label (not stable across frames, but functional).
        """
        if camera_id not in self._bg_sub:
            self._bg_sub[camera_id] = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=40, detectShadows=False
            )
        fg = self._bg_sub[camera_id].apply(frame_bgr)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.dilate(fg, kernel, iterations=2)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
        detections = []
        MIN_AREA = 2000
        for i in range(1, n_labels):
            x, y, w, h, area = stats[i]
            if area < MIN_AREA:
                continue
            # Filter by aspect ratio (person is taller than wide)
            if h < w:
                continue
            xyxy = (float(x), float(y), float(x+w), float(y+h))
            detections.append((i, xyxy, 0.60))   # fixed conf for fallback
        return detections


# ---------------------------------------------------------------------------
# Per-camera zone dwell tracker
# ---------------------------------------------------------------------------

class ZoneDwellTracker:
    """Tracks per-(visitor, zone) dwell time and emits ZONE_DWELL every 30s."""

    DWELL_INTERVAL_MS = 30_000   # 30 seconds

    def __init__(self):
        # (visitor_id, zone_id) → (enter_wall_time, last_dwell_emit_wall_time)
        self._state: Dict[Tuple[str,str], Tuple[float, float]] = {}

    def enter(self, visitor_id: str, zone_id: str, wall_time: float):
        key = (visitor_id, zone_id)
        if key not in self._state:
            self._state[key] = (wall_time, wall_time)

    def exit(self, visitor_id: str, zone_id: str, wall_time: float
             ) -> Optional[int]:
        """Returns dwell_ms since zone enter, or None."""
        key = (visitor_id, zone_id)
        if key in self._state:
            enter_time, _ = self._state.pop(key)
            return int((wall_time - enter_time) * 1000)
        return None

    def tick(self, visitor_id: str, zone_id: str, wall_time: float
             ) -> Optional[int]:
        """
        Call every frame for visitors in a zone.
        Returns dwell_ms if a 30s interval has elapsed, else None.
        """
        key = (visitor_id, zone_id)
        if key not in self._state:
            return None
        enter_time, last_emit = self._state[key]
        elapsed_since_emit = wall_time - last_emit
        if elapsed_since_emit * 1000 >= self.DWELL_INTERVAL_MS:
            self._state[key] = (enter_time, wall_time)
            return int((wall_time - enter_time) * 1000)
        return None

    def current_dwell_ms(self, visitor_id: str, zone_id: str, wall_time: float) -> int:
        key = (visitor_id, zone_id)
        if key not in self._state:
            return 0
        enter_time, _ = self._state[key]
        return int((wall_time - enter_time) * 1000)

    def clear_visitor(self, visitor_id: str, wall_time: float
                      ) -> List[Tuple[str, int]]:
        """Remove all zone state for a visitor. Returns [(zone_id, dwell_ms)]."""
        keys = [k for k in self._state if k[0] == visitor_id]
        result = []
        for k in keys:
            enter_time, _ = self._state.pop(k)
            dwell_ms = int((wall_time - enter_time) * 1000)
            result.append((k[1], dwell_ms))
        return result


# ---------------------------------------------------------------------------
# Frame annotator
# ---------------------------------------------------------------------------

def annotate_frame(
    frame_bgr:    np.ndarray,
    camera_id:    str,
    detections:   List[Tuple[int, Tuple, float]],
    identity_mgr: VisitorIdentityManager,
    staff_tracker: StaffBehaviourTracker,
    zones:        list,
    frame_w:      int,
    frame_h:      int,
    entry_detector: Optional[EntryExitDetector] = None,
    queue_depth:  int = 0,
    is_billing:   bool = False,
) -> np.ndarray:
    annotated = frame_bgr.copy()

    # Draw zone polygons
    for zone in zones:
        poly = zone.pixel_polygon(frame_w, frame_h)
        cv2.polylines(annotated, [poly], True, zone.color_bgr, 1)
        cx = int(poly[:, 0].mean())
        cy = int(poly[:, 1].mean())
        cv2.putText(annotated, zone.sku_zone, (cx-20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, zone.color_bgr, 1)

    # Draw entry line (any camera with the entry role)
    if entry_detector and entry_detector.camera_id == camera_id:
        x1, y1, x2, y2 = entry_detector.get_line_pixels(frame_w, frame_h)
        cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(annotated, "ENTRY LINE", (x1+4, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


    # Draw detections
    for track_id, bbox_xyxy, conf in detections:
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        state = identity_mgr.get_active_state_by_track(track_id, camera_id)
        if state is None:
            continue
        vid = state.visitor_id
        is_staff, _ = staff_tracker.is_staff(vid)
        color = (0, 0, 220) if is_staff else (0, 200, 50)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        health = identity_mgr.health_score(vid)
        label = f"{'STAFF' if is_staff else vid[-6:]} C:{state.final_confidence:.2f} H:{health:.0f}"
        if state.zone_id:
            label += f" [{state.zone_id.replace('ZONE_','')}]"
        cv2.putText(annotated, label, (x1, max(y1-4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # Draw Shadow Tracks
    if hasattr(identity_mgr, "_shadows"):
        for shadow in identity_mgr._shadows.active_shadows(camera_id):
            sx1, sy1, sx2, sy2 = [int(v) for v in shadow.predicted_bbox]
            # Draw dashed/dim rectangle for shadow
            cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), (150, 150, 150), 1)
            cv2.putText(annotated, f"SHADOW {shadow.visitor_id[-6:]}", (sx1, max(sy1-4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # HUD
    hud_text = [
        f"CAM: {camera_id}",
        f"ACTIVE: {identity_mgr.active_count}",
    ]
    if is_billing or camera_id == "CAM_BILLING_05":
        hud_text.append(f"QUEUE: {queue_depth}")
    for i, t in enumerate(hud_text):
        cv2.putText(annotated, t, (8, 20 + i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return annotated


# ---------------------------------------------------------------------------
# Camera processor (one per camera)
# ---------------------------------------------------------------------------

class CameraProcessor:
    def __init__(
        self,
        camera_id:      str,
        video_path:     str,
        detector:       YOLODetector,
        identity_mgr:   VisitorIdentityManager,
        staff_tracker:  StaffBehaviourTracker,
        emitter:        EventEmitter,
        start_time_utc: datetime,
        fps:            float = 15.0,
        behavior_sm:    Optional["BehaviorStateMachine"] = None,
        memory_mgr=None,
        camera_rep=None,
        camera_role:    str = "floor",
    ):
        self.camera_id     = camera_id
        self.camera_role   = camera_role
        self.video_path    = video_path
        self.detector      = detector
        self.identity_mgr  = identity_mgr
        self.staff_tracker = staff_tracker
        self.emitter       = emitter
        self.start_time    = start_time_utc
        self.fps           = fps
        self.behavior_sm   = behavior_sm   # shared across cameras

        self.memory_mgr    = memory_mgr
        self.camera_rep    = camera_rep

        self.zones         = get_zones_for_camera(camera_id)
        self._zones_refresh_counter = 0   # refresh zones every N frames
        self.dwell_tracker = ZoneDwellTracker()

        # Camera-specific modules based on role
        self.entry_detector: Optional[EntryExitDetector] = None
        self.queue_tracker:  Optional[QueueTracker]      = None
        if camera_role == "entry":
            cam_cfg = get_entry_line_for_camera(camera_id)
            self.entry_detector = EntryExitDetector(
                camera_id=camera_id,
                p1=cam_cfg.get("p1"),
                p2=cam_cfg.get("p2"),
                inside_is=cam_cfg.get("inside_is", "below"),
            )
        if camera_role == "billing":
            self.queue_tracker = QueueTracker()

        # Track which visitors we've seen zone-enter for (per zone)
        self._visitor_zones: Dict[str, Optional[str]] = {}   # visitor_id → current_zone_id

        self.frame_index = 0
        self.cap         = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logger.error(f"Cannot open {self.video_path}")
            return False
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if actual_fps > 1:
            self.fps = actual_fps
        logger.info(f"{self.camera_id}: opened {self.video_path} @ {self.fps:.1f}fps")
        return True

    def release(self):
        if self.cap:
            self.cap.release()

    def process_frame(self, frame_bgr: np.ndarray, wall_time: float) -> Tuple[Optional[np.ndarray], str]:
        """
        Process one frame. Returns (annotated frame, ISO timestamp string).
        """
        h, w = frame_bgr.shape[:2]
        ts   = make_timestamp(self.start_time, self.frame_index, self.fps)

        # Hot-reload zone geometry every 150 frames (~10s @15fps)
        self._zones_refresh_counter += 1
        if self._zones_refresh_counter >= 150:
            self._zones_refresh_counter = 0
            self.zones = get_zones_for_camera(self.camera_id)

        detections = self.detector.detect_and_track(frame_bgr, self.camera_id)

        # Track IDs seen this frame
        seen_track_ids = set()

        billing_present = []   # for queue tracker

        for track_id, bbox_xyxy, conf in detections:
            seen_track_ids.add(track_id)

            # Extract embedding for new tracks or periodically
            key = (track_id, self.camera_id)
            is_active = key in self.identity_mgr._active
            if not is_active or (self.frame_index % cfg.EMBEDDING_REFRESH_FRAMES == 0):
                embedding = self.identity_mgr.extract_embedding(frame_bgr, bbox_xyxy)
            else:
                embedding = None

            # Compute zone BEFORE resolve() so it can be passed as current_zone signal
            cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
            cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
            current_zone_obj = zone_for_point(cx, cy, w, h, self.camera_id)
            zone_id = current_zone_obj.zone_id if current_zone_obj else None

            # Resolve visitor identity — returns (visitor_id, is_new, reid_conf)
            # fingerprint requires visitor_id, so we resolve first with zone_id only
            visitor_id, is_new, reid_conf = self.identity_mgr.resolve(
                track_id, self.camera_id, bbox_xyxy, embedding, wall_time,
                det_conf=conf,
                current_zone=zone_id,
                fingerprint=None,       # updated below after visitor_id is known
                trajectory_score=0.5,
            )

            # Now compute fingerprint with the resolved visitor_id and update memory
            fp = self.memory_mgr.fingerprint(visitor_id) if self.memory_mgr else None
            traj_score = 0.5
            if self.memory_mgr and fp:
                traj_score = min(1.0, 0.5 + fp.motion_speed_mean / 20.0)

            if self.memory_mgr:
                self.memory_mgr.update(
                    visitor_id=visitor_id,
                    frame_bgr=frame_bgr,
                    bbox_xyxy=bbox_xyxy,
                    embedding=embedding,
                    cx=cx,
                    cy=cy,
                    camera_id=self.camera_id,
                    zone_id=zone_id,
                )

            # Health update
            self.identity_mgr.on_frame_observed(visitor_id)

            # Get passport
            state = self.identity_mgr.get_active_passport_by_track(track_id, self.camera_id)
            if state is None:
                continue

            # ── Confidence pipeline (spec: det × track × reid × zone) ──────
            zone_conf  = cfg.DEFAULT_ZONE_CONF
            rep_mod    = self.camera_rep.confidence_modifier(self.camera_id) if self.camera_rep else 1.0
            final_conf = round(conf * state.last_track_conf * reid_conf * zone_conf * rep_mod, 4)
            
            # Optimization: Only compute black clothing score for new tracks or periodically (every 15 frames)
            skip_clothing = not (is_new or (self.frame_index % 15 == 0))
            self.staff_tracker.update(visitor_id, frame_bgr, bbox_xyxy,
                                      self.camera_id, zone_id, skip_clothing=skip_clothing)
            is_staff, staff_conf = self.staff_tracker.is_staff(visitor_id)
            state.is_staff  = is_staff
            state.staff_conf = staff_conf

            seq = self.identity_mgr.get_session_seq(visitor_id)

            # -------------------------------------------------------
            # ENTRY / EXIT logic (Camera 3 only)
            # -------------------------------------------------------
            if self.entry_detector is not None:
                crossing = self.entry_detector.update(track_id, bbox_xyxy, w, h)
                if crossing == "ENTRY":
                    already_exited = self.identity_mgr.exit_count(visitor_id) > 0
                    seq = self.identity_mgr.increment_session_seq(visitor_id)
                    passport = self.identity_mgr.get_passport(visitor_id)
                    reentry_count = passport.reentry_count if passport else 0
                    bstate = self.behavior_sm.update(
                        visitor_id, "ENTRY", None, 0, wall_time
                    ) if self.behavior_sm else None
                    self.emitter.emit_entry(
                        visitor_id=visitor_id, camera_id=self.camera_id,
                        timestamp=ts, is_staff=is_staff, confidence=final_conf,
                        session_seq=seq, is_reentry=already_exited,
                        reentry_count=reentry_count,
                        behavior_state=bstate.value if bstate else "ENTERED",
                        session_duration_ms=passport.session_duration_ms if passport else None,
                        det_conf=conf, track_conf=state.last_track_conf,
                        reid_conf=reid_conf, zone_conf=zone_conf,
                    )
                    state.has_entered = True
                elif crossing == "EXIT":
                    seq = self.identity_mgr.increment_session_seq(visitor_id)
                    zone_exits = self.dwell_tracker.clear_visitor(visitor_id, wall_time)
                    passport = self.identity_mgr.get_passport(visitor_id)
                    bstate = self.behavior_sm.update(
                        visitor_id, "EXIT", None, 0, wall_time
                    ) if self.behavior_sm else None
                    for zid, dwell_ms in zone_exits:
                        zone_obj = next((z for z in self.zones if z.zone_id == zid), None)
                        sku = zone_obj.sku_zone if zone_obj else zid
                        zb = self.behavior_sm.update(
                            visitor_id, "ZONE_EXIT", zid, dwell_ms, wall_time
                        ) if self.behavior_sm else None
                        self.emitter.emit_zone_exit(
                            visitor_id, self.camera_id, ts, zid, sku,
                            dwell_ms, is_staff, final_conf, seq,
                            behavior_state=zb.value if zb else "BROWSING",
                            det_conf=conf, track_conf=state.last_track_conf,
                            reid_conf=reid_conf, zone_conf=zone_conf,
                        )
                    self.emitter.emit_exit(
                        visitor_id=visitor_id, camera_id=self.camera_id,
                        timestamp=ts, is_staff=is_staff, confidence=final_conf,
                        session_seq=seq,
                        session_duration_ms=passport.session_duration_ms if passport else None,
                        behavior_state=bstate.value if bstate else "EXITED",
                        det_conf=conf, track_conf=state.last_track_conf,
                        reid_conf=reid_conf, zone_conf=zone_conf,
                    )
                    self.identity_mgr.mark_exited(visitor_id)
                    state.has_exited = True

            # -------------------------------------------------------
            # ZONE logic (Cameras 1, 2, 4, 5)
            # -------------------------------------------------------
            prev_zone = self._visitor_zones.get(visitor_id)

            if zone_id != prev_zone:
                if prev_zone is not None:
                    dwell_ms = self.dwell_tracker.exit(visitor_id, prev_zone, wall_time)
                    if dwell_ms is None:
                        dwell_ms = 0
                    zone_obj = next((z for z in self.zones if z.zone_id == prev_zone), None)
                    sku = zone_obj.sku_zone if zone_obj else prev_zone
                    seq = self.identity_mgr.increment_session_seq(visitor_id)
                    zb = self.behavior_sm.update(
                        visitor_id, "ZONE_EXIT", prev_zone, dwell_ms, wall_time
                    ) if self.behavior_sm else None
                    self.emitter.emit_zone_exit(
                        visitor_id, self.camera_id, ts, prev_zone, sku,
                        dwell_ms, is_staff, final_conf, seq,
                        behavior_state=zb.value if zb else "BROWSING",
                        det_conf=conf, track_conf=state.last_track_conf,
                        reid_conf=reid_conf, zone_conf=zone_conf,
                    )
                if zone_id is not None:
                    self.dwell_tracker.enter(visitor_id, zone_id, wall_time)
                    zone_obj = current_zone_obj
                    sku = zone_obj.sku_zone if zone_obj else zone_id
                    seq = self.identity_mgr.increment_session_seq(visitor_id)
                    zb = self.behavior_sm.update(
                        visitor_id, "ZONE_ENTER", zone_id, 0, wall_time
                    ) if self.behavior_sm else None
                    self.emitter.emit_zone_enter(
                        visitor_id, self.camera_id, ts, zone_id, sku,
                        is_staff, final_conf, seq,
                        behavior_state=zb.value if zb else "BROWSING",
                        reid_score=reid_conf if not is_new else None,
                        det_conf=conf, track_conf=state.last_track_conf,
                        reid_conf=reid_conf, zone_conf=zone_conf,
                    )
                self._visitor_zones[visitor_id] = zone_id
                state.zone_id = zone_id
            else:
                if zone_id is not None:
                    dwell_ms = self.dwell_tracker.tick(visitor_id, zone_id, wall_time)
                    if dwell_ms is not None:
                        zone_obj = current_zone_obj
                        sku = zone_obj.sku_zone if zone_obj else zone_id
                        seq = self.identity_mgr.get_session_seq(visitor_id)
                        zb = self.behavior_sm.update(
                            visitor_id, "ZONE_DWELL", zone_id, dwell_ms, wall_time
                        ) if self.behavior_sm else None
                        self.emitter.emit_zone_dwell(
                            visitor_id, self.camera_id, ts, zone_id, sku,
                            dwell_ms, is_staff, final_conf, seq,
                            behavior_state=zb.value if zb else "DWELLING",
                            det_conf=conf, track_conf=state.last_track_conf,
                            reid_conf=reid_conf, zone_conf=zone_conf,
                        )

            # -------------------------------------------------------
            # BILLING QUEUE (Camera 5 only)
            # -------------------------------------------------------
            # IMPORTANT: Only count visitors in ZONE_BILLING_QUEUE, QUEUE_AREA, or dynamic queue zone ID.
            # ZONE_CASH_COUNTER is the cashier's side — staff stand there.
            # Anyone in ZONE_CASH_COUNTER is NOT a customer in queue.
            is_queue_zone = False
            if zone_id in ("ZONE_BILLING_QUEUE", "QUEUE_AREA"):
                is_queue_zone = True
            else:
                try:
                    from zone_mapper import get_mapper
                    mapped_q_id = get_mapper(self.emitter.store_id).get_queue_zone_id(self.camera_id)
                    if mapped_q_id and zone_id == mapped_q_id:
                        is_queue_zone = True
                except Exception:
                    pass

            if (self.queue_tracker is not None and is_queue_zone and not is_staff):
                seq = self.identity_mgr.get_session_seq(visitor_id)
                billing_present.append((visitor_id, is_staff, conf, seq))

        # -------------------------------------------------------
        # Mark lost tracks (track IDs the tracker dropped this frame)
        # -------------------------------------------------------
        # We check which tracks were active for this camera but not seen
        # We do this lazily — if a track was active last frame and is not
        # in seen_track_ids, give it 3 frames grace before marking lost.
        # (Simplified: mark immediately; identity manager's memory window handles re-assoc)
        for (tid, cam) in list(self.identity_mgr._active.keys()):
            if cam == self.camera_id and tid not in seen_track_ids:
                # Check if it was recently active (within 5 frames)
                state = self.identity_mgr._active.get((tid, cam))
                if state and (wall_time - state.last_seen) > (5 / self.fps):
                    self.identity_mgr.mark_lost(tid, cam)

        # -------------------------------------------------------
        # Queue tracker update
        # -------------------------------------------------------
        if self.queue_tracker is not None:
            q_events = self.queue_tracker.update(billing_present, wall_time)
            for qev in q_events:
                seq = qev["session_seq"]
                qc  = qev["confidence"]
                if qev["type"] == "BILLING_QUEUE_JOIN":
                    qb = self.behavior_sm.update(
                        qev["visitor_id"], "BILLING_QUEUE_JOIN", None, 0, wall_time
                    ) if self.behavior_sm else None
                    self.emitter.emit_billing_queue_join(
                        qev["visitor_id"], self.camera_id, ts,
                        qev["queue_depth"], qev["is_staff"], qc, seq,
                        behavior_state=qb.value if qb else "QUEUEING",
                    )
                elif qev["type"] == "BILLING_QUEUE_ABANDON":
                    qb = self.behavior_sm.update(
                        qev["visitor_id"], "BILLING_QUEUE_ABANDON", None,
                        qev["dwell_ms"], wall_time
                    ) if self.behavior_sm else None
                    self.emitter.emit_billing_queue_abandon(
                        qev["visitor_id"], self.camera_id, ts,
                        qev["dwell_ms"], qev["is_staff"], qc, seq,
                        behavior_state=qb.value if qb else "BROWSING",
                        wait_duration_ms=qev["dwell_ms"],
                    )

        # Purge stale lost tracks periodically
        if self.frame_index % 150 == 0:
            self.identity_mgr.purge_stale_lost(wall_time)

        self.frame_index += 1

        # Annotate and push frame every 3rd frame.
        # Always encode (don't gate on _frame_queues) so late-connecting
        # browsers get frames immediately the moment they open the WS.
        if self.frame_index % 3 == 0:
            queue_depth = self.queue_tracker.current_queue_depth if self.queue_tracker else 0
            annotated = annotate_frame(
                frame_bgr, self.camera_id, detections, self.identity_mgr,
                self.staff_tracker, self.zones, w, h,
                self.entry_detector, queue_depth,
                is_billing=(self.camera_role == "billing"),
            )
        else:
            annotated = None
        return annotated, ts


# ---------------------------------------------------------------------------
# Timestamp extractor (OCR for visible watermarks)
# ---------------------------------------------------------------------------

def ocr_timestamp_from_frame(frame_bgr: np.ndarray) -> Optional[datetime]:
    """
    Attempt to OCR the timestamp watermark from the frame.
    Scans all four corners with a robust multi-threshold and prioritized ROI approach.
    Returns a UTC-aware datetime or None.
    """
    try:
        import pytesseract
        import re
        
        h, w = frame_bgr.shape[:2]
        
        # Prioritized ROIs (upper_right is highly likely for our store camera format)
        rois = [
            ("upper_right", frame_bgr[0:150, w-600:w]),
            ("upper_left",  frame_bgr[0:150, 0:600]),
            ("lower_right", frame_bgr[h-150:h, w-600:w]),
            ("lower_left",  frame_bgr[h-150:h, 0:600]),
        ]
        
        def parse_dt(text):
            # Correct common OCR character substitutions
            text = text.replace("o", "0").replace("O", "0")
            text = text.replace("I", "1").replace("l", "1").replace("S", "5")
            cleaned = re.sub(r'[^0-9\s:/|-]', '', text).strip()
            
            # Match DD/MM/YYYY HH:MM:SS
            pattern = r'\b\d{2}[/|-]\d{2}[/|-]\d{4}\s+\d{2}:\d{2}:\d{2}\b'
            for match in re.findall(pattern, cleaned):
                for fmt in ("%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
                            "%d-%m-%Y %H:%M:%S", "%m-%d-%Y %H:%M:%S"):
                    try:
                        dt = datetime.strptime(match, fmt)
                        if dt.year < 2000:
                            dt = dt.replace(year=2026)
                        if 2020 <= dt.year <= 2030:
                            return dt.replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
            
            # Match DD/MM/YYYYHH:MM:SS (no space)
            pattern_nospace = r'\b\d{2}[/|-]\d{2}[/|-]\d{4}\d{2}:\d{2}:\d{2}\b'
            for match in re.findall(pattern_nospace, cleaned):
                match_with_space = match[:10] + " " + match[10:]
                for fmt in ("%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
                            "%d-%m-%Y %H:%M:%S", "%m-%d-%Y %H:%M:%S"):
                    try:
                        dt = datetime.strptime(match_with_space, fmt)
                        if dt.year < 2000:
                            dt = dt.replace(year=2026)
                        if 2020 <= dt.year <= 2030:
                            return dt.replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
            return None

        thresholds = [200, 160, 120, 220, 140]
        for name, roi in rois:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Try multiple thresholds to isolate text
            for thresh_val in thresholds:
                _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                text = pytesseract.image_to_string(
                    thresh,
                    config="--psm 6 -c tessedit_char_whitelist=0123456789:-T/ ",
                ).strip()
                dt = parse_dt(text)
                if dt:
                    logger.debug(f"Successfully OCR'd timestamp from {name} using thresh={thresh_val}: '{text}' → {dt.isoformat()}")
                    return dt
            # Fallback: try without whitelist
            for thresh_val in thresholds:
                _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                text = pytesseract.image_to_string(thresh, config="--psm 6").strip()
                dt = parse_dt(text)
                if dt:
                    logger.debug(f"Successfully OCR'd timestamp (no whitelist) from {name} using thresh={thresh_val}: '{text}' → {dt.isoformat()}")
                    return dt
    except Exception as e:
        logger.warning(f"OCR timestamp error: {e}")
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def infer_camera_start_time(
    camera_file_map: Dict[str, str],
) -> Dict[str, datetime]:
    """
    Try to OCR start timestamps from first frame of each camera.
    For cameras where OCR fails, infer from others.
    Assumes all cameras are synchronised.
    """
    from collections import Counter
    start_times = {}
    fallback_dt = datetime.now(timezone.utc)

    max_valid_date = None
    for camera_id, path in camera_file_map.items():
        if not path or not os.path.exists(path):
            continue

        mtime = os.path.getmtime(path)
        mtime_dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        if max_valid_date is None or mtime_dt > max_valid_date:
            max_valid_date = mtime_dt

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue
        dt = ocr_timestamp_from_frame(frame)
        if dt:
            start_times[camera_id] = dt
            logger.info(f"Raw OCR start time for {camera_id}: {dt.isoformat()}")

    if max_valid_date:
        logger.info(f"Max valid date based on file mtime: {max_valid_date.date().isoformat()}")
    else:
        max_valid_date = fallback_dt

    valid_dates = []
    for camera_id, dt in start_times.items():
        if dt.date() <= max_valid_date.date():
            valid_dates.append(dt.date())
        else:
            logger.warning(f"Discarding impossible future date for {camera_id}: {dt.date().isoformat()}")

    if valid_dates:
        from collections import Counter
        consensus_date = Counter(valid_dates).most_common(1)[0][0]
    else:
        consensus_date = fallback_dt.date()

    logger.info(f"Consensus Date: {consensus_date.isoformat()}")

    # Find reference time from any successful OCR
    reference_dt = None
    for camera_id, dt in start_times.items():
        t = dt.time()
        reference_dt = datetime.combine(consensus_date, t, tzinfo=timezone.utc)
        break
    if not reference_dt:
        reference_dt = fallback_dt

    final_start_times: Dict[str, datetime] = {}
    for camera_id in camera_file_map:
        if camera_id in start_times:
            t = start_times[camera_id].time()
            corrected = datetime.combine(consensus_date, t, tzinfo=timezone.utc)
            final_start_times[camera_id] = corrected
        else:
            final_start_times[camera_id] = reference_dt

    return final_start_times



def run_pipeline(
    store_id:        str,
    output_path:     str,
    camera_file_map: Dict[str, str],
    camera_role_map: Optional[Dict[str, str]] = None,
    adjacency_map:   Optional[Dict[str, List[str]]] = None,
    gui_port:        int   = 8080,
    speed:           float = 1.0,
    clips_dir:       str   = "",   # retained for legacy CLI compat only
):
    """
    Start the detection pipeline.

    camera_file_map : {camera_id: /path/to/video.mp4}  — required
    camera_role_map : {camera_id: role}  e.g. {"CAM_ENTRY_01": "entry"}
    adjacency_map   : {camera_id: [neighbour_camera_id, ...]}  (wizard step 3.5)
    """
    if not camera_file_map:
        logger.error("run_pipeline: camera_file_map is empty — nothing to process")
        return
    logger.info(f"Starting detection pipeline for {store_id}")
    logger.info(f"Cameras: {list(camera_file_map.keys())}  Output: {output_path}  Speed: {speed}x")

    # --- Wire dynamic topology into tracker + store_graph ---
    role_map = camera_role_map or {}
    adj: Dict[str, set] = {cam: set(nbs) for cam, nbs in (adjacency_map or {}).items()}
    try:
        from tracker import set_adjacency_map
        set_adjacency_map(adj)
    except Exception as e:
        logger.warning(f"Could not set adjacency map: {e}")
    try:
        from store_graph import set_camera_topology
        set_camera_topology(adj, role_map=role_map)
    except Exception as e:
        logger.warning(f"Could not set camera topology: {e}")

    # --- Start GUI server ---
    run_server(port=gui_port, shared=SHARED)
    logger.info(f"GUI available at http://localhost:{gui_port}")

    # --- Shared detection modules ---
    # (Must be instantiated BEFORE the emitter closure that captures them)
    detector      = YOLODetector(device="auto")
    identity_mgr  = VisitorIdentityManager()
    staff_tracker = StaffBehaviourTracker()
    behavior_sm   = BehaviorStateMachine()   # shared across all cameras
    memory_mgr    = VisitorMemoryManager()
    camera_rep    = CameraReputation()
    # Seed camera reputation from wizard roles
    if role_map:
        camera_rep.seed_from_role_map(role_map)
    auditor       = SystemAuditor(str(Path(output_path).parent))
    auditor.set_broadcast(lambda w: SHARED.push_warning(w) if hasattr(SHARED, 'push_warning') else None)
    SHARED.identity_mgr = identity_mgr

    # --- Event emitter ---
    emitter = EventEmitter(output_path, store_id)
    emitter.open()
    emitter.set_broadcast_queue(None)   # GUI broadcast handled via SHARED.push_event

    # Monkey-patch emitter to also push to SHARED and wire new modules
    _orig_emit = emitter.emit
    def _emit_and_share(event):
        # 1. Update memory graph
        if hasattr(identity_mgr, "_memory_graph"):
            try:
                identity_mgr._memory_graph.add_event(
                    event.visitor_id, event.camera_id,
                    getattr(event, "zone_id", None), event.event_type, time.time()
                )
            except Exception:
                pass

        # 2. Append confidence lineage
        try:
            passport = identity_mgr.get_passport(event.visitor_id)
            if passport and passport.last_reid_explanation:
                event.metadata["confidence_lineage"] = passport.last_reid_explanation
        except Exception:
            pass

        result = _orig_emit(event)
        if result:
            SHARED.push_event(event.to_dict())
        return result
    emitter.emit = _emit_and_share

    # --- Infer per-camera start timestamps ---
    start_times = infer_camera_start_time(
        camera_file_map=camera_file_map
    )

    # --- Build per-camera processors ---
    processors: Dict[str, CameraProcessor] = {}

    for camera_id, path in camera_file_map.items():
        if not path or not os.path.exists(path):
            logger.warning(f"Video file not found for {camera_id}: {path}")
            continue
        start_dt = start_times.get(camera_id, datetime.now(timezone.utc))
        role = role_map.get(camera_id, "floor")
        proc = CameraProcessor(
            camera_id=camera_id,
            video_path=path,
            detector=detector,
            identity_mgr=identity_mgr,
            staff_tracker=staff_tracker,
            emitter=emitter,
            start_time_utc=start_dt,
            behavior_sm=behavior_sm,
            memory_mgr=memory_mgr,
            camera_rep=camera_rep,
            camera_role=role,
        )
        if proc.open():
            processors[camera_id] = proc

    if not processors:
        logger.error("No cameras opened. Exiting.")
        emitter.close()
        return

    active_cams = list(processors.keys())
    logger.info(f"Running with cameras: {active_cams}")

    # --- Main loop ---
    # Use the actual mean FPS of all cameras for the sleep throttle so
    # speed=1.0 really means "process at footage real-time rate".
    avg_fps = (
        sum(p.fps for p in processors.values()) / len(processors)
        if processors else 15.0
    )
    frame_delay = 0.0 if speed <= 0 else ((1.0 / avg_fps) / speed)
    start_wall  = time.time()
    global_frame = 0

    # Track which cameras are still producing frames
    exhausted_cameras: set = set()

    try:
        while True:
            t_frame_start = time.time()
            any_frames    = False

            for camera_id, proc in list(processors.items()):
                ret, frame = proc.cap.read()
                if ret:
                    # CPU Bottleneck Optimization: Downscale 1080p frame immediately to 960x540.
                    # This reduces CPU pixel load by 4x for all subsequent operations (tracking, cropping, color histogram, overlays).
                    frame = cv2.resize(frame, (960, 540))
                if not ret:
                    if camera_id not in exhausted_cameras:
                        # --- Bug 3 fix: flush dangling zone dwells on clip end ---
                        exhausted_cameras.add(camera_id)
                        wall_time = start_wall + proc.frame_index / proc.fps
                        ts = make_timestamp(proc.start_time, proc.frame_index, proc.fps)
                        logger.info(
                            f"{camera_id}: video ended at frame {proc.frame_index} "
                            f"— flushing {len(proc._visitor_zones)} active visitor zones"
                        )
                        for visitor_id, zone_id in list(proc._visitor_zones.items()):
                            if zone_id is None:
                                continue
                            dwell_ms = proc.dwell_tracker.exit(visitor_id, zone_id, wall_time)
                            if dwell_ms is None:
                                dwell_ms = 0
                            zone_obj = next(
                                (z for z in proc.zones if z.zone_id == zone_id), None
                            )
                            sku = zone_obj.sku_zone if zone_obj else zone_id
                            is_staff, conf = staff_tracker.is_staff(visitor_id)
                            seq = identity_mgr.get_session_seq(visitor_id)
                            proc.emitter.emit_zone_exit(
                                visitor_id, camera_id, ts, zone_id, sku,
                                dwell_ms, is_staff, conf, seq
                            )
                        proc._visitor_zones.clear()
                    continue
                any_frames = True

                # Wall clock time for this frame
                wall_time = start_wall + proc.frame_index / proc.fps

                annotated, ts = proc.process_frame(frame, wall_time)

                # Push to GUI
                if annotated is not None:
                    # Encode as JPEG at moderate quality for streaming
                    gui_frame = cv2.resize(annotated, (640, 360))
                    ok, buf = cv2.imencode(".jpg", gui_frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                    if ok:
                        SHARED.push_frame(camera_id, bytes(buf), ts)

            if not any_frames:
                logger.info("All cameras exhausted. Pipeline complete.")
                break

            # Update GUI metrics
            staff_ids = {
                vid.visitor_id for vid in identity_mgr._active.values()
                if staff_tracker.is_staff(vid.visitor_id)[0]
            }
            queue_d = max(
                (p.queue_tracker.current_queue_depth
                 for p in processors.values() if p.queue_tracker),
                default=0
            )
            SHARED.update_metrics(
                active_visitors=identity_mgr.active_count - len(staff_ids),
                staff_count=len(staff_ids),
                queue_depth=queue_d,
                cameras_active=active_cams,
            )

            # --- Audit ---
            if global_frame % cfg.AUDIT_INTERVAL_FRAMES == 0:
                active_cams_map = {p.visitor_id: p.camera_id for p in identity_mgr._active.values()}
                auditor.check(
                    active_passports=identity_mgr._active,
                    active_cameras=active_cams_map,
                    staff_ids=staff_ids,
                    queue_depth=queue_d,
                    health_scores=identity_mgr.all_health_scores(),
                    new_ids_this_cycle=0,  # Could be tracked if needed, keeping simple
                    wall_time=time.time(),
                )

            global_frame += 1
            if global_frame % 150 == 0:
                elapsed = time.time() - start_wall
                logger.info(f"Frame {global_frame} | elapsed {elapsed:.0f}s | "
                             f"events {emitter.count} | "
                             f"active {identity_mgr.active_count}")

            # Pace output to simulate real-time
            elapsed_frame = time.time() - t_frame_start
            sleep_for = frame_delay - elapsed_frame
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
    finally:
        for proc in processors.values():
            proc.release()
        emitter.close()
        logger.info(f"Pipeline finished. Total events: {emitter.count}")
        logger.info(f"Output written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import json as _json
    parser = argparse.ArgumentParser(
        description="Store Intelligence Detection Pipeline\n\n"
                    "Requires --camera_map (JSON dict mapping camera_id to video path).\n"
                    "Example:\n"
                    "  python detect.py \\\n"
                    "    --store_id MY_STORE \\\n"
                    "    --camera_map '{\"CAM_ENTRY_01\":\"/data/entry.mp4\",\"CAM_FLOOR_02\":\"/data/floor.mp4\"}' \\\n"
                    "    --output /data/events.jsonl"
    )
    parser.add_argument("--store_id",
                        default=os.environ.get("STORE_ID", ""),
                        help="Store ID. Falls back to STORE_ID env var.")
    parser.add_argument("--camera_map",
                        default=os.environ.get("CAMERA_MAP", "{}"),
                        help="JSON dict: {camera_id: video_path}. Also settable via CAMERA_MAP env var.")
    parser.add_argument("--camera_roles",
                        default=os.environ.get("CAMERA_ROLES", "{}"),
                        help="JSON dict: {camera_id: role}. Also settable via CAMERA_ROLES env var.")
    parser.add_argument("--adjacency",
                        default=os.environ.get("CAMERA_ADJACENCY", "{}"),
                        help="JSON dict: {camera_id: [neighbour_ids]}. Also settable via CAMERA_ADJACENCY env var.")
    parser.add_argument("--output",
                        default=os.environ.get("OUTPUT", "/data/events.jsonl"),
                        help="Path for output events JSONL file.")
    parser.add_argument("--gui_port",   type=int, default=int(os.environ.get("GUI_PORT", "8080")),
                        help="Port for web GUI dashboard.")
    parser.add_argument("--speed",      type=float, default=float(os.environ.get("SPEED", "1.0")),
                        help="Playback speed multiplier (1.0 = real-time, 0 = max speed).")
    parser.add_argument("--log_level",  default=os.environ.get("LOG_LEVEL", "INFO"),
                        help="Logging level.")
    parser.add_argument("--clips_dir",  default=os.environ.get("CLIPS_DIR", "/data/clips"),
                        help="Directory containing video clips.")
    parser.add_argument("--cam3_start_iso", default=os.environ.get("CAM3_START_ISO", ""),
                        help="ISO-8601 UTC override for entry cam start time.")

    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        camera_file_map = _json.loads(args.camera_map) if args.camera_map else {}
    except Exception as e:
        parser.error(f"--camera_map is not valid JSON: {e}")

    try:
        camera_role_map = _json.loads(args.camera_roles) if args.camera_roles and args.camera_roles != "{}" else None
    except Exception:
        camera_role_map = None

    try:
        adjacency_map = _json.loads(args.adjacency) if args.adjacency and args.adjacency != "{}" else None
    except Exception:
        adjacency_map = None

    if not camera_file_map:
        logger.info("No camera map provided. Starting GUI server in Wizard mode on port %d...", args.gui_port)
        from gui_server import run_server, SHARED
        run_server(port=args.gui_port, shared=SHARED)
        # Keep main thread alive while wizard runs
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping Wizard server...")
        return

    if not args.store_id:
        parser.error("--store_id is required (or set STORE_ID env var)")

    run_pipeline(
        store_id=args.store_id,
        output_path=args.output,
        camera_file_map=camera_file_map,
        camera_role_map=camera_role_map,
        adjacency_map=adjacency_map,
        gui_port=args.gui_port,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
