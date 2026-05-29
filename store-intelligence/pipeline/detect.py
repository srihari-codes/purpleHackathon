"""
detect.py — Main detection + tracking pipeline.

Architecture:
  1. Load YOLOv8n model (ultralytics) — detects persons
  2. ByteTrack (built-in ultralytics tracker) assigns track IDs per frame
  3. OSNet Re-ID (torchreid) extracts appearance embeddings per crop
  4. VisitorTracker maps track IDs → stable visitor_ids across cameras
  5. Zone classification: point-in-polygon for floor camera, direction for entry
  6. EventEmitter sends structured events to the API

Processing is done at 5fps (every 3rd frame from 15fps source) for speed.
At 5fps, a 20-minute clip = 6000 frames — manageable on CPU.
GPU is used automatically if CUDA is available.

Usage:
  python -m pipeline.detect --clip path/to/clip.mp4 \
                             --store-id STORE_BLR_002 \
                             --camera-id CAM_ENTRY_01 \
                             --layout data/store_layout.json \
                             --clip-start "2026-03-03T14:00:00Z"
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─────────────────────────────────────────────
# Lazy imports (heavy models)
# ─────────────────────────────────────────────

def _load_yolo():
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")   # downloads on first run
    logger.info("YOLOv8n loaded")
    return model


def _load_reid_model():
    """Load OSNet for appearance embeddings. Falls back gracefully if torchreid unavailable."""
    try:
        import torchreid
        model = torchreid.models.build_model(
            name="osnet_x0_25",
            num_classes=1000,
            pretrained=True,
        )
        model.eval()
        logger.info("OSNet Re-ID model loaded")
        return model
    except Exception as exc:
        logger.warning("OSNet not available (%s) — using bbox-trajectory Re-ID fallback", exc)
        return None


def _extract_embedding(reid_model, crop: np.ndarray) -> Optional[np.ndarray]:
    """Extract OSNet embedding from a person crop."""
    if reid_model is None:
        return None
    try:
        import torch
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(crop).unsqueeze(0)
        with torch.no_grad():
            embedding = reid_model(tensor)
        return embedding.squeeze().numpy()
    except Exception as exc:
        logger.debug("Embedding extraction failed: %s", exc)
        return None


# ─────────────────────────────────────────────
# Zone classification
# ─────────────────────────────────────────────

def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def classify_zone(
    bbox_center: Tuple[float, float],
    camera_id: str,
    layout: dict,
    frame_height: int,
) -> Optional[str]:
    """
    Returns zone_id for the given bbox center and camera.
    For entry cameras: determine ENTRY_THRESHOLD.
    For floor cameras: polygon-based zone lookup.
    Falls back to None if no zone matches.
    """
    store = layout
    for zone in store.get("zones", []):
        if camera_id not in zone.get("camera_ids", []):
            continue
        polygon = zone.get("polygon")
        if polygon and point_in_polygon(bbox_center, polygon):
            return zone["zone_id"]
    return None


def direction_is_entry(
    track_history: List[Tuple[float, float]],
    frame_height: int,
    camera_type: str,
) -> Optional[bool]:
    """
    For entry/exit cameras: return True if moving inward (increasing y = deeper in store).
    Requires at least 5 frames of history.
    """
    if camera_type != "entry_exit" or len(track_history) < 5:
        return None
    ys = [p[1] for p in track_history[-10:]]
    delta = ys[-1] - ys[0]
    return delta > 0   # positive y movement = entering


# ─────────────────────────────────────────────
# Main processing function
# ─────────────────────────────────────────────

def process_clip(
    clip_path: Path,
    store_id: str,
    camera_id: str,
    camera_type: str,
    layout: dict,
    clip_start: datetime,
    output_jsonl: Optional[Path] = None,
    process_every_n: int = 3,   # 15fps → 5fps effective
    api_url: str = "http://localhost:8000",
) -> int:
    """
    Process a single CCTV clip and emit structured events.
    Returns total number of events emitted.
    """
    import cv2
    from pipeline.tracker import VisitorTracker, is_staff_by_colour
    from pipeline.emit import EventEmitter

    import os
    os.environ["API_BASE_URL"] = api_url

    yolo_model = _load_yolo()
    reid_model = _load_reid_model()

    tracker = VisitorTracker(store_id=store_id)
    emitter = EventEmitter(
        store_id=store_id,
        clip_start_time=clip_start,
        output_jsonl=output_jsonl,
    )

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        logger.error("Cannot open clip: %s", clip_path)
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frame_idx = 0
    track_histories: Dict[int, List[Tuple[float, float]]] = {}
    visitor_in_zone: Dict[str, str] = {}   # visitor_id → current zone_id

    logger.info("Processing %s — camera %s — %.0f fps", clip_path.name, camera_id, fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % process_every_n != 0:
            continue

        timestamp = emitter.frame_to_ts(frame_idx, fps=fps)
        h, w = frame.shape[:2]

        # ── YOLOv8 + ByteTrack ──────────────────────────────────
        results = yolo_model.track(
            frame,
            persist=True,
            classes=[0],   # person class only
            conf=0.35,
            iou=0.45,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        if not results or results[0].boxes is None:
            continue

        boxes = results[0].boxes
        active_tracks = set()

        for box in boxes:
            if box.id is None:
                continue
            track_id = int(box.id.item())
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            active_tracks.add(track_id)

            # Track history
            if track_id not in track_histories:
                track_histories[track_id] = []
            track_histories[track_id].append((cx, cy))

            # ── Re-ID / visitor assignment ───────────────────────
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            embedding = _extract_embedding(reid_model, crop) if crop.size > 0 else None
            visitor_id, is_reentry = tracker.get_or_create_visitor(
                track_id, clip_path.stem, timestamp, embedding=embedding
            )

            # ── Staff detection ──────────────────────────────────
            is_staff = False
            if crop.size > 0:
                is_staff = is_staff_by_colour(crop)
            if not is_staff:
                tracker.record_zone(visitor_id, "check")  # update zone counter
                is_staff = tracker.classify_staff(visitor_id)
            tracker.set_staff(visitor_id, is_staff)

            # ── Entry / Exit (entry camera only) ─────────────────
            if camera_type == "entry_exit":
                direction = direction_is_entry(track_histories[track_id], h, camera_type)
                if direction is True and len(track_histories[track_id]) == 5:
                    emitter.emit_entry(camera_id, visitor_id, timestamp,
                                       is_staff=is_staff, confidence=conf,
                                       is_reentry=is_reentry)
                elif direction is False and len(track_histories[track_id]) == 5:
                    emitter.emit_exit(camera_id, visitor_id, timestamp,
                                      is_staff=is_staff, confidence=conf)
                    tracker.mark_exit(visitor_id, timestamp)

            # ── Zone classification (floor / billing cameras) ─────
            else:
                zone_id = classify_zone((cx, cy), camera_id, layout, h)
                prev_zone = visitor_in_zone.get(visitor_id)

                if zone_id != prev_zone:
                    # Zone change
                    if prev_zone is not None:
                        dwell_ms = 0  # approximate; precise dwell tracked by emitter
                        emitter.emit_zone_exit(camera_id, visitor_id, prev_zone,
                                               timestamp, dwell_ms=dwell_ms,
                                               is_staff=is_staff, confidence=conf)

                    if zone_id is not None:
                        visitor_in_zone[visitor_id] = zone_id
                        tracker.record_zone(visitor_id, zone_id)
                        # Billing queue special handling
                        if zone_id == "BILLING_QUEUE":
                            # Count current people in BILLING_QUEUE as rough queue depth
                            queue_depth = sum(
                                1 for vid, z in visitor_in_zone.items()
                                if z == "BILLING_QUEUE" and not tracker.is_staff(vid)
                            )
                            emitter.emit_billing_queue_join(
                                camera_id, visitor_id, timestamp,
                                queue_depth=queue_depth,
                                is_staff=is_staff, confidence=conf,
                            )
                        else:
                            emitter.emit_zone_enter(camera_id, visitor_id, zone_id,
                                                    timestamp, is_staff=is_staff,
                                                    confidence=conf)
                    else:
                        visitor_in_zone.pop(visitor_id, None)

                # Dwell ticker — every 30s of continuous presence
                emitter.tick_dwell(camera_id, visitor_id, timestamp,
                                   is_staff=is_staff, confidence=conf)

        # Handle track losses (exit without explicit exit event)
        lost_tracks = set(track_histories.keys()) - active_tracks
        # (Handled implicitly by ByteTrack when tracks drop)

    cap.release()

    # Flush remaining events
    emitter.close()
    logger.info("Clip processing complete. Frame count: %d", frame_idx)
    return len(emitter._buffer)


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Store Intelligence Detection Pipeline")
    parser.add_argument("--clip", required=True, help="Path to CCTV clip (.mp4)")
    parser.add_argument("--store-id", required=True, help="Store ID (e.g. STORE_BLR_002)")
    parser.add_argument("--camera-id", required=True, help="Camera ID (e.g. CAM_ENTRY_01)")
    parser.add_argument("--camera-type", default="main_floor",
                        choices=["entry_exit", "main_floor", "billing"],
                        help="Camera type")
    parser.add_argument("--layout", default="data/store_layout.json",
                        help="Path to store_layout.json")
    parser.add_argument("--clip-start", required=True,
                        help="Clip start time in ISO-8601 UTC (e.g. 2026-03-03T14:00:00Z)")
    parser.add_argument("--output-jsonl", default=None,
                        help="Path to write output events JSONL")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="Store Intelligence API URL")
    parser.add_argument("--every-n-frames", type=int, default=3,
                        help="Process every Nth frame (default: 3 = 5fps from 15fps)")

    args = parser.parse_args()

    clip_path = Path(args.clip)
    if not clip_path.exists():
        logger.error("Clip not found: %s", clip_path)
        sys.exit(1)

    layout_path = Path(args.layout)
    if not layout_path.exists():
        logger.error("Layout file not found: %s", layout_path)
        sys.exit(1)

    with open(layout_path) as f:
        full_layout = json.load(f)

    # Find the store layout
    store_layout = None
    stores = full_layout if isinstance(full_layout, list) else full_layout.get("stores", [full_layout])
    for s in stores:
        if s.get("store_id") == args.store_id:
            store_layout = s
            break
    if store_layout is None:
        logger.error("Store %s not found in layout file", args.store_id)
        sys.exit(1)

    clip_start = datetime.fromisoformat(args.clip_start.replace("Z", "+00:00"))

    process_clip(
        clip_path=clip_path,
        store_id=args.store_id,
        camera_id=args.camera_id,
        camera_type=args.camera_type,
        layout=store_layout,
        clip_start=clip_start,
        output_jsonl=Path(args.output_jsonl) if args.output_jsonl else None,
        process_every_n=args.every_n_frames,
        api_url=args.api_url,
    )


if __name__ == "__main__":
    main()
