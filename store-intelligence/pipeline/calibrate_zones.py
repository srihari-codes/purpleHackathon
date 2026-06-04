"""
calibrate_zones.py — Professional zone calibration studio.
Usage:  python calibrate_zones.py --store_id MY_STORE --clips_dir /my/clips --port 8081
"""
import argparse, base64, json, logging, os
from pathlib import Path
from datetime import datetime, timezone
import cv2, numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)
HERE  = Path(__file__).parent
ROOT  = HERE.parent
HTML  = HERE / "_calib_ui.html"


def _resolve_calib_path(store_id: str) -> Path:
    """
    Resolve calibration save path for a given store_id.
    Search order: CALIB_DIR env var → /data/calibration/ → config/calibration/
    """
    env = os.environ.get("CALIB_DIR", "").strip()
    if env:
        p = Path(env); p.mkdir(parents=True, exist_ok=True)
        return p / f"{store_id}.json"
    docker = Path("/data/calibration")
    if docker.exists():
        return docker / f"{store_id}.json"
    local = ROOT / "config" / "calibration"
    local.mkdir(parents=True, exist_ok=True)
    return local / f"{store_id}.json"


def _load_cameras_from_calib(calib_path: Path) -> list:
    """
    Return the list of camera IDs from the existing calibration JSON.
    Returns an empty list if file does not exist.
    """
    if not calib_path.exists():
        return []
    try:
        with open(calib_path) as f:
            data = json.load(f)
        return list(data.get("cameras", {}).keys())
    except Exception as e:
        logger.warning("Could not read cameras from %s: %s", calib_path, e)
        return []


def first_frame_b64(clips_dir: str, camera_id: str, width: int = 960) -> str:
    """Extract and encode the first frame from any video matching the camera_id."""
    # First try reading from the shared session.json file if it exists
    session_file = Path("/data/si_session/session.json")
    if session_file.exists():
        try:
            with open(session_file) as f:
                sess_data = json.load(f)
            for cam in sess_data.get("cameras", []):
                if cam.get("camera_id") == camera_id and cam.get("uploaded"):
                    fp = cam.get("file_path")
                    if fp and os.path.exists(fp):
                        cap = cv2.VideoCapture(fp)
                        ok, fr = cap.read(); cap.release()
                        if ok and fr is not None:
                            h, w = fr.shape[:2]
                            fr = cv2.resize(fr, (width, int(h * width / w)))
                            _, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 82])
                            return base64.b64encode(bytes(buf)).decode()
        except Exception as e:
            # Fall back to standard folder scanning
            pass

    clips = Path(clips_dir)
    # Try patterns: exact camera_id, camera_id with spaces, lowercase, generic mp4
    for pat in [f"{camera_id}.*", f"{camera_id.replace('_', ' ')}.*", f"*{camera_id}*"]:
        hits = list(clips.glob(pat))
        if hits:
            cap = cv2.VideoCapture(str(hits[0]))
            ok, fr = cap.read(); cap.release()
            if ok and fr is not None:
                h, w = fr.shape[:2]
                fr = cv2.resize(fr, (width, int(h * width / w)))
                _, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 82])
                return base64.b64encode(bytes(buf)).decode()
    blank = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(blank, f"No video for {camera_id}", (60, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 2)
    _, buf = cv2.imencode(".jpg", blank)
    return base64.b64encode(bytes(buf)).decode()


def make_app(store_id: str, clips_dir: str):
    app = FastAPI(title="Zone Calibration Studio")
    _frames = {}

    def get_active_store_info():
        # Default fallback
        active_store_id = store_id
        session_file = Path("/data/si_session/session.json")
        session_cameras = []
        if session_file.exists():
            try:
                with open(session_file) as f:
                    sess_data = json.load(f)
                s_id = sess_data.get("store_id")
                if s_id:
                    active_store_id = s_id
                session_cameras = [cam.get("camera_id") for cam in sess_data.get("cameras", []) if cam.get("camera_id")]
            except Exception:
                pass
        
        path = _resolve_calib_path(active_store_id)
        return active_store_id, path, session_cameras

    def load_calib():
        active_store_id, path, session_cameras = get_active_store_info()
        fallback_cameras = _load_cameras_from_calib(path)

        if path.exists():
            with open(path) as f:
                data = json.load(f)
            # Make sure all cameras from the session are in the calibration data dict
            if "cameras" not in data:
                data["cameras"] = {}
            # If session cameras are specified, strictly filter to show only those
            if session_cameras:
                data["cameras"] = {c: data["cameras"][c] for c in session_cameras if c in data["cameras"]}
            for c in session_cameras:
                if c not in data["cameras"]:
                    data["cameras"][c] = {"shapes": []}
            data["store_id"] = active_store_id
            return data

        # Empty template with whatever cameras we know about
        cams_to_use = session_cameras if session_cameras else fallback_cameras
        return {
            "store_id": active_store_id,
            "version": "v1",
            "cameras": {c: {"shapes": []} for c in cams_to_use},
        }

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html = HTML.read_text()
        calib = load_calib()
        # Use cameras from the loaded calibration
        cam_list = list(calib.get("cameras", {}).keys())
        html = html.replace("__CAMERAS__", json.dumps(cam_list))
        html = html.replace("__CALIB__", json.dumps(calib))
        return html

    @app.get("/frame/{camera_id}")
    async def frame(camera_id: str):
        if camera_id not in _frames:
            _frames[camera_id] = first_frame_b64(clips_dir, camera_id)
        return JSONResponse({"frame": _frames[camera_id]})

    @app.post("/save")
    async def save(req: Request):
        data = await req.json()
        active_store_id, path, session_cameras = get_active_store_info()
        data["store_id"] = active_store_id
        data["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        # reload zone_mapper
        try:
            import sys; sys.path.insert(0, str(HERE))
            from zone_mapper import get_mapper
            get_mapper(active_store_id).reload()
        except Exception:
            pass
        return JSONResponse({"ok": True, "path": str(path)})

    @app.post("/validate")
    async def validate(req: Request):
        data = await req.json()
        try:
            from verifier import CalibrationVerifier
            issues = CalibrationVerifier().validate_dict(data)
        except Exception as e:
            issues = [{"severity": "INFO", "code": "VALIDATOR_UNAVAILABLE",
                       "message": str(e), "camera_id": None, "shape_id": None}]
        return JSONResponse({"issues": issues})

    return app


def main():
    p = argparse.ArgumentParser(description="Zone Calibration Studio")
    p.add_argument("--store_id",
                   default=os.environ.get("STORE_ID", ""),
                   help="Store ID (required). Falls back to STORE_ID env var.")
    p.add_argument("--clips_dir",
                   default=os.environ.get("CLIPS_DIR", "/data/clips"),
                   help="Directory containing CCTV video clips.")
    p.add_argument("--port", type=int, default=8081)
    args = p.parse_args()

    if not args.store_id:
        p.error("--store_id is required (or set STORE_ID env var)")

    import uvicorn
    app = make_app(store_id=args.store_id, clips_dir=args.clips_dir)
    print(f"\n  Zone Calibration Studio [{args.store_id}] → http://localhost:{args.port}\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
