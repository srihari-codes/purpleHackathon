"""
wizard_backend.py — Session management + API endpoints for the onboarding wizard.
Imported by gui_server.py create_app() to register all wizard routes.

Camera role → internal camera_id mapping:
  entry   → CAM_ENTRY_03
  billing → CAM_BILLING_05
  floor   → CAM_FLOOR_01, CAM_FLOOR_02, ...
  godown  → CAM_GODOWN_04, CAM_GODOWN_05, ...
"""

import base64
import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SESSION_DIR = Path("/tmp/si_session")

ROLE_CAM_PREFIXES = {
    "entry":   "CAM_ENTRY",
    "billing": "CAM_BILLING",
    "floor":   "CAM_FLOOR",
    "godown":  "CAM_GODOWN",
}

ROLE_LABELS = {
    "entry":   "Entrance / Exit",
    "billing": "Billing Counter",
    "floor":   "Sales Floor",
    "godown":  "Godown / Staff",
}


@dataclass
class CameraSlot:
    slot_id: str          # UUID
    role: str             # entry | billing | floor | godown
    camera_id: str        # e.g. CAM_FLOOR_01
    label: str            # human label
    file_path: Optional[str] = None
    filename: Optional[str] = None
    uploaded: bool = False


@dataclass
class WizardSession:
    store_id:   str = ""
    store_name: str = ""
    store_code: str = ""
    analysis_date: str = ""
    cameras: List[CameraSlot] = field(default_factory=list)
    pipeline_running: bool = False
    pipeline_done: bool = False
    pipeline_error: str = ""
    events_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def camera_file_map(self) -> Dict[str, str]:
        return {c.camera_id: c.file_path for c in self.cameras if c.uploaded and c.file_path}

    def next_camera_id(self, role: str) -> str:
        prefix = ROLE_CAM_PREFIXES.get(role, "CAM_UNKNOWN")
        existing = [c for c in self.cameras if c.role == role]
        n = len(existing) + 1
        return f"{prefix}_{n:02d}"

    def to_dict(self):
        return {
            "store_id": self.store_id,
            "store_name": self.store_name,
            "store_code": self.store_code,
            "analysis_date": self.analysis_date,
            "cameras": [
                {
                    "slot_id": c.slot_id,
                    "role": c.role,
                    "camera_id": c.camera_id,
                    "label": c.label,
                    "filename": c.filename,
                    "uploaded": c.uploaded,
                }
                for c in self.cameras
            ],
            "pipeline_running": self.pipeline_running,
            "pipeline_done": self.pipeline_done,
            "pipeline_error": self.pipeline_error,
            "events_count": self.events_count,
        }


SESSION = WizardSession()


def register_wizard_routes(app, shared):
    """Register all wizard API endpoints on the FastAPI app."""
    from fastapi import HTTPException, UploadFile, File
    from fastapi.responses import JSONResponse, Response

    SESSION_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Store Setup
    # ------------------------------------------------------------------

    @app.post("/api/setup")
    async def api_setup(body: dict):
        SESSION.store_name  = body.get("store_name", "")
        SESSION.store_code  = body.get("store_code", "")
        SESSION.store_id    = body.get("store_id") or f"ST_{SESSION.store_code.upper()}"
        SESSION.analysis_date = body.get("analysis_date", "")
        return {"ok": True, "store_id": SESSION.store_id}

    # ------------------------------------------------------------------
    # Step 2 — Camera Management
    # ------------------------------------------------------------------

    @app.post("/api/camera/add")
    def api_camera_add(body: dict):
        role = body.get("role", "floor")
        if role not in ROLE_CAM_PREFIXES:
            raise HTTPException(400, "Invalid role")
        camera_id = SESSION.next_camera_id(role)
        slot = CameraSlot(
            slot_id=str(uuid.uuid4()),
            role=role,
            camera_id=camera_id,
            label=f"{ROLE_LABELS[role]} ({camera_id})",
        )
        SESSION.cameras.append(slot)
        (SESSION_DIR / slot.slot_id).mkdir(parents=True, exist_ok=True)
        return {"ok": True, "slot": slot.__dict__}

    @app.delete("/api/camera/{slot_id}")
    def api_camera_remove(slot_id: str):
        SESSION.cameras = [c for c in SESSION.cameras if c.slot_id != slot_id]
        slot_dir = SESSION_DIR / slot_id
        if slot_dir.exists():
            shutil.rmtree(slot_dir, ignore_errors=True)
        return {"ok": True}

    @app.post("/api/camera/{slot_id}/upload")
    def api_camera_upload(slot_id: str, file: UploadFile = File(...)):
        slot = next((c for c in SESSION.cameras if c.slot_id == slot_id), None)
        if not slot:
            raise HTTPException(404, "Camera slot not found")
        dest_dir = SESSION_DIR / slot_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        slot.file_path = str(dest)
        slot.filename  = file.filename
        slot.uploaded  = True
        return {"ok": True, "path": str(dest), "camera_id": slot.camera_id}

    @app.get("/api/camera/{slot_id}/frame")
    def api_camera_frame(slot_id: str):
        slot = next((c for c in SESSION.cameras if c.slot_id == slot_id), None)
        if not slot or not slot.uploaded:
            raise HTTPException(404, "Camera not uploaded")
        try:
            import cv2
            cap = cv2.VideoCapture(slot.file_path)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise HTTPException(500, "Cannot read frame")
            frame = cv2.resize(frame, (640, 360))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buf.tobytes()).decode()
            return {"ok": True, "frame": b64, "camera_id": slot.camera_id}
        except Exception as e:
            raise HTTPException(500, str(e))

    # ------------------------------------------------------------------
    # Step 3 — Zone Config (handled by external tool on port 8081)
    # ------------------------------------------------------------------

    @app.get("/api/calibration/url/{slot_id}")
    async def api_calib_url(slot_id: str):
        slot = next((c for c in SESSION.cameras if c.slot_id == slot_id), None)
        if not slot:
            raise HTTPException(404, "Camera slot not found")
        return {"url": f"http://localhost:8081/?camera_id={slot.camera_id}",
                "camera_id": slot.camera_id}

    # ------------------------------------------------------------------
    # Step 4 — Start Analysis
    # ------------------------------------------------------------------

    @app.get("/api/session")
    async def api_session():
        return SESSION.to_dict()

    @app.post("/api/analysis/start")
    async def api_analysis_start(body: dict):
        if SESSION.pipeline_running:
            return {"ok": False, "error": "Already running"}
        if not SESSION.cameras or not any(c.uploaded for c in SESSION.cameras):
            raise HTTPException(400, "No cameras uploaded")

        speed = float(body.get("speed", 0))   # default: max speed

        import sys, os
        pipeline_dir = os.path.dirname(os.path.abspath(__file__))
        if pipeline_dir not in sys.path:
            sys.path.insert(0, pipeline_dir)

        def _run():
            try:
                SESSION.pipeline_running = True
                SESSION.pipeline_error   = ""
                from detect import run_pipeline
                output_path = str(SESSION_DIR / "events.jsonl")
                run_pipeline(
                    store_id        = SESSION.store_id or "STORE_DEMO",
                    clips_dir       = str(SESSION_DIR),
                    output_path     = output_path,
                    gui_port        = 8080,
                    speed           = speed,
                    camera_file_map = SESSION.camera_file_map(),
                )
                SESSION.pipeline_done = True
            except Exception as e:
                SESSION.pipeline_error = str(e)
                logger.exception("Pipeline error")
            finally:
                SESSION.pipeline_running = False

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return {"ok": True, "store_id": SESSION.store_id}

    @app.get("/api/analysis/status")
    async def api_analysis_status():
        return {
            "running": SESSION.pipeline_running,
            "done":    SESSION.pipeline_done,
            "error":   SESSION.pipeline_error,
        }

