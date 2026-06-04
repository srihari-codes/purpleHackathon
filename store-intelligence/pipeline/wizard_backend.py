"""
wizard_backend.py — Session management + API endpoints for the onboarding wizard.
Imported by gui_server.py create_app() to register all wizard routes.

Camera roles: entry | billing | floor | godown
Camera IDs are assigned dynamically by the wizard (e.g. CAM_ENTRY_01, CAM_FLOOR_02).
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

SESSION_DIR = Path("/data/si_session")

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
    # adjacency_map: {camera_id: [neighbour_camera_id, ...]}
    adjacency_map: Dict[str, List[str]] = field(default_factory=dict)
    pipeline_running: bool = False
    pipeline_done: bool = False
    pipeline_error: str = ""
    events_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def camera_file_map(self) -> Dict[str, str]:
        return {c.camera_id: c.file_path for c in self.cameras if c.uploaded and c.file_path}

    def camera_role_map(self) -> Dict[str, str]:
        """Return {camera_id: role} for all slots."""
        return {c.camera_id: c.role for c in self.cameras}

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
                    "file_path": c.file_path,
                    "uploaded": c.uploaded,
                }
                for c in self.cameras
            ],
            "adjacency_map": self.adjacency_map,
            "pipeline_running": self.pipeline_running,
            "pipeline_done": self.pipeline_done,
            "pipeline_error": self.pipeline_error,
            "events_count": self.events_count,
        }


SESSION = WizardSession()


def save_session():
    try:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        with open(SESSION_DIR / "session.json", "w") as f:
            json.dump(SESSION.to_dict(), f, indent=2)
    except Exception as e:
        logger.error("Failed to save session: %s", e)


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
        save_session()
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
        save_session()
        return {"ok": True, "slot": slot.__dict__}

    @app.delete("/api/camera/{slot_id}")
    def api_camera_remove(slot_id: str):
        SESSION.cameras = [c for c in SESSION.cameras if c.slot_id != slot_id]
        slot_dir = SESSION_DIR / slot_id
        if slot_dir.exists():
            shutil.rmtree(slot_dir, ignore_errors=True)
        save_session()
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
        save_session()
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
    # Step 3.5 — Camera Adjacency
    # ------------------------------------------------------------------

    @app.post("/api/adjacency")
    async def api_adjacency(body: dict):
        """
        Save camera adjacency map provided by the user.
        Body: {adjacency: {camera_id: [neighbour_ids]}}
        """
        adj = body.get("adjacency", {})
        # Validate that all referenced camera IDs exist in the session
        valid_cams = {c.camera_id for c in SESSION.cameras}
        for cam, nbs in adj.items():
            unknown = [c for c in ([cam] + list(nbs)) if c not in valid_cams]
            if unknown:
                raise HTTPException(400, f"Unknown camera IDs: {unknown}")
        SESSION.adjacency_map = {k: list(v) for k, v in adj.items()}
        save_session()
        return {"ok": True, "adjacency": SESSION.adjacency_map}

    @app.get("/api/adjacency")
    async def api_adjacency_get():
        return {
            "adjacency": SESSION.adjacency_map,
            "cameras": [
                {"camera_id": c.camera_id, "role": c.role, "label": c.label}
                for c in SESSION.cameras
            ],
        }

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

        # Set STORE_ID env var so zone_mapper picks it up
        os.environ["STORE_ID"] = SESSION.store_id or "STORE_DEMO"

        def _run():
            try:
                SESSION.pipeline_running = True
                SESSION.pipeline_error   = ""
                from detect import run_pipeline
                output_path = str(SESSION_DIR / "events.jsonl")
                run_pipeline(
                    store_id        = SESSION.store_id or "STORE_DEMO",
                    output_path     = output_path,
                    camera_file_map = SESSION.camera_file_map(),
                    camera_role_map = SESSION.camera_role_map(),
                    adjacency_map   = SESSION.adjacency_map or None,
                    gui_port        = 8080,
                    speed           = speed,
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

