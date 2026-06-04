"""
calibrate_zones.py — Professional zone calibration studio.
Usage:  python calibrate_zones.py --clips_dir ../data/clips --port 8081
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

# Resolve calibration save path (same logic as zone_mapper.py):
# 1. CALIB_DIR env var
# 2. /data/calibration/ — Docker shared volume
# 3. config/calibration/ — local dev
def _resolve_calib_path() -> Path:
    env = os.environ.get("CALIB_DIR", "").strip()
    if env:
        p = Path(env); p.mkdir(parents=True, exist_ok=True); return p / "STORE_BLR_002.json"
    docker = Path("/data/calibration")
    if docker.exists():
        return docker / "STORE_BLR_002.json"
    local = ROOT / "config" / "calibration"
    local.mkdir(parents=True, exist_ok=True)
    return local / "STORE_BLR_002.json"

CALIB = _resolve_calib_path()


CAMERA_MAP = {
    "1":"CAM_FLOOR_01","2":"CAM_FLOOR_02","3":"CAM_ENTRY_03",
    "4":"CAM_GODOWN_04","5":"CAM_BILLING_05"
}
CAMERAS = list(CAMERA_MAP.values())
CLIPS_DIR = "/data/clips"

def first_frame_b64(clips_dir, camera_id, width=960):
    num = {v:k for k,v in CAMERA_MAP.items()}.get(camera_id,"")
    for pat in [f"CAM {num}.*", f"CAM{num}.*", f"cam{num}.*"]:
        hits = list(Path(clips_dir).glob(pat))
        if hits:
            cap = cv2.VideoCapture(str(hits[0]))
            ok, fr = cap.read(); cap.release()
            if ok and fr is not None:
                h,w = fr.shape[:2]; fr = cv2.resize(fr,(width,int(h*width/w)))
                _,buf = cv2.imencode(".jpg",fr,[cv2.IMWRITE_JPEG_QUALITY,82])
                return base64.b64encode(bytes(buf)).decode()
    blank = np.zeros((540,960,3),dtype=np.uint8)
    cv2.putText(blank,f"No video for {camera_id}",(60,270),cv2.FONT_HERSHEY_SIMPLEX,1.2,(80,80,80),2)
    _,buf = cv2.imencode(".jpg",blank); return base64.b64encode(bytes(buf)).decode()

def load_calib():
    if CALIB.exists():
        with open(CALIB) as f: return json.load(f)
    return {"store_id":"STORE_BLR_002","version":"v1","cameras":{c:{"shapes":[]} for c in CAMERAS}}

def make_app(clips_dir):
    app = FastAPI(title="Zone Calibration Studio")
    _frames = {}
    @app.get("/", response_class=HTMLResponse)
    async def index():
        html = HTML.read_text()
        calib = load_calib()
        html = html.replace("__CAMERAS__", json.dumps(CAMERAS))
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
        data["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
        CALIB.parent.mkdir(parents=True, exist_ok=True)
        with open(CALIB,"w") as f: json.dump(data,f,indent=2)
        # reload zone_mapper
        try:
            import sys; sys.path.insert(0,str(HERE))
            from zone_mapper import get_mapper
            get_mapper().reload()
        except Exception: pass
        return JSONResponse({"ok":True,"path":str(CALIB)})
    @app.post("/validate")
    async def validate(req: Request):
        data = await req.json()
        try:
            from verifier import CalibrationVerifier
            issues = CalibrationVerifier().validate_dict(data)
        except Exception as e:
            issues = [{"severity":"INFO","code":"VALIDATOR_UNAVAILABLE","message":str(e),"camera_id":None,"shape_id":None}]
        return JSONResponse({"issues": issues})
    return app

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips_dir", default="/data/clips")
    p.add_argument("--port", type=int, default=8081)
    args = p.parse_args()
    global CLIPS_DIR; CLIPS_DIR = args.clips_dir
    import uvicorn
    app = make_app(args.clips_dir)
    print(f"\n  Zone Calibration Studio → http://localhost:{args.port}\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
