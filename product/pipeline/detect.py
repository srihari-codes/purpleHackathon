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
from events     import EventEmitter, make_timestamp
from tracker    import VisitorIdentityManager
from staff      import StaffBehaviourTracker
from zones      import get_zones_for_camera, zone_for_point, ENTRY_LINE_NORM
from entry_exit import EntryExitDetector
from queue      import QueueTracker
from gui_server import run_server, SHARED

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
)
logger = logging.getLogger("detect")

# ---------------------------------------------------------------------------
# Camera ID mapping
# ---------------------------------------------------------------------------
CAMERA_MAP = {
    "1": "CAM_FLOOR_01",
    "2": "CAM_FLOOR_02",
    "3": "CAM_ENTRY_03",
    "4": "CAM_GODOWN_04",
    "5": "CAM_BILLING_05",
}

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
            from ultralytics import YOLO
            model_path = os.environ.get("YOLO_MODEL", "yolov8n.pt")
            self._model = YOLO(model_path)
            # Pick device
            if device == "auto":
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                dev = device
            self._model.to(dev)
            self._use_yolo = True
            logger.info(f"YOLODetector: YOLOv8 loaded on {dev}")
        except Exception as e:
            logger.warning(f"YOLODetector: YOLOv8 unavailable ({e}). "
                           f"Falling back to MOG2 background subtraction.")
            self._use_yolo = False
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

    # Draw entry line
    if entry_detector and camera_id == "CAM_ENTRY_03":
        x1, y, x2, _ = entry_detector.get_line_pixels(frame_w, frame_h)
        cv2.line(annotated, (x1, y), (x2, y), (0, 255, 255), 2)
        cv2.putText(annotated, "ENTRY LINE", (x1+4, y-6),
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
        label = f"{'STAFF' if is_staff else vid[-6:]} {conf:.2f}"
        if state.zone_id:
            label += f" [{state.zone_id.replace('ZONE_','')}]"
        cv2.putText(annotated, label, (x1, max(y1-4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # HUD
    hud_text = [
        f"CAM: {camera_id}",
        f"ACTIVE: {identity_mgr.active_count}",
    ]
    if camera_id == "CAM_BILLING_05":
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
    ):
        self.camera_id     = camera_id
        self.video_path    = video_path
        self.detector      = detector
        self.identity_mgr  = identity_mgr
        self.staff_tracker = staff_tracker
        self.emitter       = emitter
        self.start_time    = start_time_utc
        self.fps           = fps

        self.zones         = get_zones_for_camera(camera_id)
        self.dwell_tracker = ZoneDwellTracker()

        # Camera-specific modules
        self.entry_detector: Optional[EntryExitDetector] = None
        self.queue_tracker:  Optional[QueueTracker]      = None
        if camera_id == "CAM_ENTRY_03":
            cfg = ENTRY_LINE_NORM.get("CAM_ENTRY_03", {})
            self.entry_detector = EntryExitDetector(
                camera_id=camera_id,
                line_y_norm=cfg.get("line_y", 0.50),
                line_x_start=cfg.get("line_x_start", 0.10),
                line_x_end=cfg.get("line_x_end", 0.90),
                inside_is=cfg.get("inside_is", "below"),
            )
        if camera_id == "CAM_BILLING_05":
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

    def process_frame(self, frame_bgr: np.ndarray, wall_time: float) -> Optional[np.ndarray]:
        """
        Process one frame. Returns annotated frame (for GUI) or None on error.
        """
        h, w = frame_bgr.shape[:2]
        ts   = make_timestamp(self.start_time, self.frame_index, self.fps)

        detections = self.detector.detect_and_track(frame_bgr, self.camera_id)

        # Track IDs seen this frame
        seen_track_ids = set()

        billing_present = []   # for queue tracker

        for track_id, bbox_xyxy, conf in detections:
            seen_track_ids.add(track_id)

            # Extract appearance embedding
            embedding = self.identity_mgr.extract_embedding(frame_bgr, bbox_xyxy)

            # Resolve visitor identity
            visitor_id, is_new = self.identity_mgr.resolve(
                track_id, self.camera_id, bbox_xyxy, embedding, wall_time
            )

            # Get state
            state = self.identity_mgr.get_active_state_by_track(track_id, self.camera_id)
            if state is None:
                continue

            # Update staff tracker
            cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
            cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
            current_zone_obj = zone_for_point(cx, cy, w, h, self.camera_id)
            zone_id = current_zone_obj.zone_id if current_zone_obj else None
            self.staff_tracker.update(visitor_id, frame_bgr, bbox_xyxy,
                                      self.camera_id, zone_id)
            is_staff, staff_conf = self.staff_tracker.is_staff(visitor_id)
            state.is_staff  = is_staff
            state.staff_conf = staff_conf

            seq = self.identity_mgr.get_session_seq(visitor_id)

            # -------------------------------------------------------
            # ENTRY / EXIT logic (Camera 3 only)
            # -------------------------------------------------------
            if self.entry_detector is not None:
                crossing = self.entry_detector.update(
                    track_id, bbox_xyxy, w, h
                )
                if crossing == "ENTRY":
                    already_exited = self.identity_mgr.exit_count(visitor_id) > 0
                    seq = self.identity_mgr.increment_session_seq(visitor_id)
                    self.emitter.emit_entry(
                        visitor_id=visitor_id,
                        camera_id=self.camera_id,
                        timestamp=ts,
                        is_staff=is_staff,
                        confidence=conf,
                        session_seq=seq,
                        is_reentry=already_exited,
                    )
                    state.has_entered = True
                elif crossing == "EXIT":
                    seq = self.identity_mgr.increment_session_seq(visitor_id)
                    # Flush zone dwell on exit
                    zone_exits = self.dwell_tracker.clear_visitor(visitor_id, wall_time)
                    for zid, dwell_ms in zone_exits:
                        zone_obj = next((z for z in self.zones if z.zone_id == zid), None)
                        sku = zone_obj.sku_zone if zone_obj else zid
                        self.emitter.emit_zone_exit(
                            visitor_id, self.camera_id, ts, zid, sku,
                            dwell_ms, is_staff, conf, seq
                        )
                    self.emitter.emit_exit(
                        visitor_id=visitor_id,
                        camera_id=self.camera_id,
                        timestamp=ts,
                        is_staff=is_staff,
                        confidence=conf,
                        session_seq=seq,
                    )
                    self.identity_mgr.mark_exited(visitor_id)
                    state.has_exited = True

            # -------------------------------------------------------
            # ZONE logic (Cameras 1, 2, 4, 5)
            # -------------------------------------------------------
            prev_zone = self._visitor_zones.get(visitor_id)

            if zone_id != prev_zone:
                # Zone exit
                if prev_zone is not None:
                    dwell_ms = self.dwell_tracker.exit(visitor_id, prev_zone, wall_time)
                    if dwell_ms is None:
                        dwell_ms = 0
                    zone_obj = next((z for z in self.zones if z.zone_id == prev_zone), None)
                    sku = zone_obj.sku_zone if zone_obj else prev_zone
                    seq = self.identity_mgr.increment_session_seq(visitor_id)
                    self.emitter.emit_zone_exit(
                        visitor_id, self.camera_id, ts, prev_zone, sku,
                        dwell_ms, is_staff, conf, seq
                    )
                # Zone enter
                if zone_id is not None:
                    self.dwell_tracker.enter(visitor_id, zone_id, wall_time)
                    zone_obj = current_zone_obj
                    sku = zone_obj.sku_zone if zone_obj else zone_id
                    seq = self.identity_mgr.increment_session_seq(visitor_id)
                    self.emitter.emit_zone_enter(
                        visitor_id, self.camera_id, ts, zone_id, sku,
                        is_staff, conf, seq
                    )
                self._visitor_zones[visitor_id] = zone_id
                state.zone_id = zone_id

            else:
                # Still in same zone — check for dwell tick
                if zone_id is not None:
                    dwell_ms = self.dwell_tracker.tick(visitor_id, zone_id, wall_time)
                    if dwell_ms is not None:
                        zone_obj = current_zone_obj
                        sku = zone_obj.sku_zone if zone_obj else zone_id
                        seq = self.identity_mgr.get_session_seq(visitor_id)
                        self.emitter.emit_zone_dwell(
                            visitor_id, self.camera_id, ts, zone_id, sku,
                            dwell_ms, is_staff, conf, seq
                        )

            # -------------------------------------------------------
            # BILLING QUEUE (Camera 5 only)
            # -------------------------------------------------------
            if (self.queue_tracker is not None and
                    zone_id in ("ZONE_BILLING_QUEUE", "ZONE_CASH_COUNTER")):
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
                if qev["type"] == "BILLING_QUEUE_JOIN":
                    self.emitter.emit_billing_queue_join(
                        qev["visitor_id"], self.camera_id, ts,
                        qev["queue_depth"], qev["is_staff"],
                        qev["confidence"], seq
                    )
                elif qev["type"] == "BILLING_QUEUE_ABANDON":
                    self.emitter.emit_billing_queue_abandon(
                        qev["visitor_id"], self.camera_id, ts,
                        qev["dwell_ms"], qev["is_staff"],
                        qev["confidence"], seq
                    )

        # Purge stale lost tracks periodically
        if self.frame_index % 150 == 0:
            self.identity_mgr.purge_stale_lost(wall_time)

        self.frame_index += 1

        # Annotate frame
        queue_depth = self.queue_tracker.current_queue_depth if self.queue_tracker else 0
        annotated = annotate_frame(
            frame_bgr, self.camera_id, detections, self.identity_mgr,
            self.staff_tracker, self.zones, w, h,
            self.entry_detector, queue_depth,
        )
        return annotated


# ---------------------------------------------------------------------------
# Timestamp extractor (OCR for visible watermarks)
# ---------------------------------------------------------------------------

def ocr_timestamp_from_frame(frame_bgr: np.ndarray) -> Optional[datetime]:
    """
    Attempt to OCR the timestamp watermark from the upper-right corner.
    Returns a UTC-aware datetime or None.

    Requires pytesseract and tesseract-ocr installed.
    Cam3 watermark is blurred — will return None (handled by caller).
    """
    try:
        import pytesseract
        h, w = frame_bgr.shape[:2]
        # Upper-right 300×60 px region
        roi = frame_bgr[0:60, w-320:w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh,
                                           config="--psm 7 -c tessedit_char_whitelist=0123456789:-T ")
        text = text.strip()
        # Try to parse ISO-like format
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%d-%m-%Y %H:%M:%S",
                    "%Y/%m/%d %H:%M:%S", "%H:%M:%S"):
            try:
                dt = datetime.strptime(text[:19], fmt)
                if dt.year < 2000:
                    dt = dt.replace(year=2026)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def find_video_file(clips_dir: str, cam_num: str) -> Optional[str]:
    """Find video file for camera number in clips directory."""
    clips = Path(clips_dir)
    patterns = [
        f"CAM {cam_num}.*",
        f"cam{cam_num}.*",
        f"CAM{cam_num}.*",
        f"camera{cam_num}.*",
        f"camera_{cam_num}.*",
    ]
    for pat in patterns:
        matches = list(clips.glob(pat))
        if matches:
            return str(matches[0])
    return None


def infer_camera_start_time(
    clips_dir: str, cam_nums: list
) -> Dict[str, datetime]:
    """
    Try to OCR start timestamps from first frame of each camera.
    For cameras where OCR fails (e.g. Cam3 blur), infer from others.
    Assumes all cameras are synchronised.
    """
    start_times = {}
    fallback_dt = datetime(2026, 3, 3, 9, 0, 0, tzinfo=timezone.utc)

    for cam_num in cam_nums:
        path = find_video_file(clips_dir, cam_num)
        if not path:
            continue
        camera_id = CAMERA_MAP.get(cam_num)
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
            logger.info(f"OCR start time {camera_id}: {dt.isoformat()}")

    # Fill missing cameras with the first successfully OCR'd time
    reference_dt = next(iter(start_times.values()), fallback_dt)
    for cam_num in cam_nums:
        camera_id = CAMERA_MAP.get(cam_num)
        if camera_id and camera_id not in start_times:
            start_times[camera_id] = reference_dt
            logger.info(f"Inferred start time {camera_id}: {reference_dt.isoformat()} (synced)")

    return start_times


def run_pipeline(
    store_id:    str,
    clips_dir:   str,
    output_path: str,
    gui_port:    int  = 8080,
    speed:       float = 1.0,
    cam3_start:  Optional[str] = None,
):
    logger.info(f"Starting detection pipeline for {store_id}")
    logger.info(f"Clips dir: {clips_dir}  Output: {output_path}  Speed: {speed}x")

    # --- Start GUI server ---
    run_server(port=gui_port, shared=SHARED)
    logger.info(f"GUI available at http://localhost:{gui_port}")

    # --- Event emitter ---
    emitter = EventEmitter(output_path, store_id)
    emitter.open()
    emitter.set_broadcast_queue(None)   # GUI broadcast handled via SHARED.push_event

    # Monkey-patch emitter to also push to SHARED
    _orig_emit = emitter.emit
    def _emit_and_share(event):
        result = _orig_emit(event)
        if result:
            SHARED.push_event(event.to_dict())
        return result
    emitter.emit = _emit_and_share

    # --- Shared detection modules ---
    detector      = YOLODetector(device="auto")
    identity_mgr  = VisitorIdentityManager()
    staff_tracker = StaffBehaviourTracker()

    # --- Discover camera files ---
    cam_nums  = list(CAMERA_MAP.keys())
    start_times = infer_camera_start_time(clips_dir, cam_nums)

    # Override Cam3 start time if provided
    if cam3_start:
        try:
            dt = datetime.fromisoformat(cam3_start.replace("Z", "+00:00"))
            start_times["CAM_ENTRY_03"] = dt
            logger.info(f"Cam3 start time overridden: {dt.isoformat()}")
        except Exception as e:
            logger.warning(f"Could not parse cam3_start_iso: {e}")

    # --- Build per-camera processors ---
    processors: Dict[str, CameraProcessor] = {}
    caps: Dict[str, cv2.VideoCapture] = {}

    for cam_num, camera_id in CAMERA_MAP.items():
        path = find_video_file(clips_dir, cam_num)
        if not path:
            logger.warning(f"No video file found for {camera_id} (cam {cam_num})")
            continue
        start_dt = start_times.get(camera_id,
                                   datetime(2026, 3, 3, 9, 0, 0, tzinfo=timezone.utc))
        proc = CameraProcessor(
            camera_id=camera_id,
            video_path=path,
            detector=detector,
            identity_mgr=identity_mgr,
            staff_tracker=staff_tracker,
            emitter=emitter,
            start_time_utc=start_dt,
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
    frame_delay = (1.0 / 15.0) / max(speed, 0.01)   # sleep between frames
    start_wall  = time.time()
    global_frame = 0

    try:
        while True:
            t_frame_start = time.time()
            any_frames    = False

            for camera_id, proc in list(processors.items()):
                ret, frame = proc.cap.read()
                if not ret:
                    logger.info(f"{camera_id}: video ended at frame {proc.frame_index}")
                    continue
                any_frames = True

                # Wall clock time for this frame
                wall_time = start_wall + proc.frame_index / proc.fps

                annotated = proc.process_frame(frame, wall_time)

                # Push to GUI
                if annotated is not None:
                    # Encode as JPEG at moderate quality for streaming
                    gui_frame = cv2.resize(annotated, (640, 360))
                    ok, buf = cv2.imencode(".jpg", gui_frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                    if ok:
                        SHARED.push_frame(camera_id, bytes(buf))

            if not any_frames:
                logger.info("All cameras exhausted. Pipeline complete.")
                break

            # Update GUI metrics
            staff_ids = {
                vid for vid in identity_mgr._active.values()
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
    parser = argparse.ArgumentParser(
        description="Store Intelligence Detection Pipeline"
    )
    parser.add_argument("--store_id",   default="STORE_BLR_002",
                        help="Store ID (e.g. STORE_BLR_002)")
    parser.add_argument("--clips_dir",  default="/data/clips",
                        help="Directory containing CAM 1.mp4 … CAM 5.mp4")
    parser.add_argument("--output",     default="/data/events.jsonl",
                        help="Path for output events JSONL file")
    parser.add_argument("--gui_port",   type=int, default=8080,
                        help="Port for web GUI dashboard")
    parser.add_argument("--speed",      type=float, default=1.0,
                        help="Playback speed multiplier (1.0 = real-time, 0 = max speed)")
    parser.add_argument("--cam3_start_iso", default=None,
                        help="Override Cam3 start timestamp (ISO-8601 UTC)")
    parser.add_argument("--log_level",  default="INFO",
                        help="Logging level")

    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    run_pipeline(
        store_id=args.store_id,
        clips_dir=args.clips_dir,
        output_path=args.output,
        gui_port=args.gui_port,
        speed=args.speed,
        cam3_start=args.cam3_start_iso,
    )


if __name__ == "__main__":
    main()
