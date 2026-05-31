"""
calibrate_zones.py — Interactive zone polygon editor.

Loads the first frame from each camera clip and serves a browser-based
polygon editor. You can drag-create zone polygons per camera and export
a zones_override.json that the main pipeline will use.

Usage:
    python calibrate_zones.py --clips_dir /data/clips --port 8081

Then open http://localhost:8081 in your browser.
"""

import argparse
import base64
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CAMERA_MAP = {
    "1": "CAM_FLOOR_01",
    "2": "CAM_FLOOR_02",
    "3": "CAM_ENTRY_03",
    "4": "CAM_GODOWN_04",
    "5": "CAM_BILLING_05",
}

ZONE_COLORS = [
    "#00C878", "#0055FF", "#FF6600", "#CC00FF", "#00CCFF",
    "#FF0055", "#FFCC00", "#00FF88", "#FF4488", "#88FF00",
]


def grab_first_frame(video_path: str, width: int = 960) -> str:
    """Return base64-encoded JPEG of first frame, scaled to `width`."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        # Return blank frame
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.putText(frame, "Could not read video", (60, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
    h, w = frame.shape[:2]
    scale = width / w
    new_h = int(h * scale)
    frame = cv2.resize(frame, (width, new_h))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(bytes(buf)).decode()


CALIB_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Zone Calibration Tool</title>
<style>
* { box-sizing: border-box; }
body { background:#0d1117; color:#c9d1d9; font-family:monospace; margin:0; }
h1 { padding:10px 16px; background:#161b22; border-bottom:1px solid #30363d;
     font-size:.95rem; color:#58a6ff; }
#layout { display:flex; height:calc(100vh - 44px); }
#cambar { width:160px; background:#161b22; border-right:1px solid #30363d;
          overflow-y:auto; padding:8px 0; }
.cam-btn { display:block; width:100%; padding:8px 12px; background:none;
           border:none; color:#8b949e; text-align:left; cursor:pointer;
           font-family:monospace; font-size:.75rem; }
.cam-btn.active { background:#21262d; color:#58a6ff; }
#canvas-wrap { position:relative; flex:1; overflow:hidden; background:#000; }
canvas { cursor:crosshair; display:block; }
#sidebar { width:280px; background:#161b22; border-left:1px solid #30363d;
           padding:12px; display:flex; flex-direction:column; gap:10px;
           overflow-y:auto; }
#sidebar h2 { font-size:.8rem; color:#8b949e; }
.zone-item { background:#0d1117; border:1px solid #30363d; border-radius:4px;
             padding:8px; font-size:.72rem; }
.zone-item input { background:#21262d; border:1px solid #30363d; color:#c9d1d9;
                   font-family:monospace; padding:2px 6px; width:100%; margin-top:4px; }
.del-btn { float:right; background:none; border:none; color:#f85149;
           cursor:pointer; font-size:.8rem; }
button.action { background:#21262d; border:1px solid #30363d; color:#58a6ff;
                padding:6px 12px; cursor:pointer; font-family:monospace;
                font-size:.75rem; border-radius:4px; width:100%; }
button.action:hover { background:#30363d; }
#export-out { background:#0d1117; border:1px solid #30363d; padding:8px;
              font-size:.68rem; white-space:pre-wrap; max-height:180px;
              overflow-y:auto; border-radius:4px; }
#hint { font-size:.68rem; color:#8b949e; line-height:1.5; }
</style>
</head>
<body>
<h1>🗺 Zone Calibration — draw zone polygons on real camera frames</h1>
<div id="layout">
  <div id="cambar" id="cambar"></div>
  <div id="canvas-wrap">
    <canvas id="c"></canvas>
  </div>
  <div id="sidebar">
    <div>
      <h2>INSTRUCTIONS</h2>
      <p id="hint">
        1. Select a camera on the left.<br>
        2. Click to add polygon vertices.<br>
        3. Right-click or press Enter to close the polygon.<br>
        4. Name the zone in the list below.<br>
        5. Click Export to download zones_override.json.
      </p>
    </div>
    <div>
      <h2>ZONES — <span id="cur-cam">none</span></h2>
      <div id="zone-list"></div>
      <button class="action" onclick="clearLast()">Undo Last Point</button>
      <button class="action" onclick="clearCurrent()">Cancel Current</button>
    </div>
    <div>
      <h2>EXPORT</h2>
      <button class="action" onclick="exportZones()">Export zones_override.json</button>
      <div id="export-out"></div>
    </div>
  </div>
</div>
<script>
const CAMERAS = __CAMERAS__;
const COLORS  = __COLORS__;
let frames    = {};  // camera_id → base64 jpeg
let allZones  = {};  // camera_id → [{zone_id, sku_zone, polygon_norm, color}]
CAMERAS.forEach(c => { allZones[c] = []; });

let currentCam     = null;
let currentPolygon = [];   // [{x,y} normalised]
let imgEl          = null;
let imgW = 0, imgH = 0;

const canvas = document.getElementById("c");
const ctx    = canvas.getContext("2d");

// --- Camera selection ---
const cambar = document.getElementById("cambar");
CAMERAS.forEach((cid, i) => {
  const btn = document.createElement("button");
  btn.className = "cam-btn";
  btn.textContent = cid;
  btn.onclick = () => selectCam(cid);
  btn.id = "btn-" + cid;
  cambar.appendChild(btn);
});

function selectCam(cid) {
  currentCam = cid;
  currentPolygon = [];
  document.querySelectorAll(".cam-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("btn-"+cid).classList.add("active");
  document.getElementById("cur-cam").textContent = cid;

  if (frames[cid]) {
    drawScene();
    renderZoneList();
    return;
  }
  fetch("/frame/" + encodeURIComponent(cid))
    .then(r => r.json())
    .then(d => {
      frames[cid] = d.frame;
      drawScene();
      renderZoneList();
    });
}

// --- Drawing ---
function drawScene() {
  if (!currentCam || !frames[currentCam]) return;
  const img = new Image();
  img.onload = () => {
    canvas.width  = img.width;
    canvas.height = img.height;
    imgW = img.width; imgH = img.height;
    ctx.drawImage(img, 0, 0);
    drawZones();
    drawCurrentPolygon();
  };
  img.src = "data:image/jpeg;base64," + frames[currentCam];
}

function drawZones() {
  if (!currentCam) return;
  (allZones[currentCam] || []).forEach((z, i) => {
    if (z.polygon_norm.length < 2) return;
    ctx.beginPath();
    z.polygon_norm.forEach(([nx,ny], j) => {
      const px = nx * imgW, py = ny * imgH;
      j === 0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
    });
    ctx.closePath();
    ctx.strokeStyle = z.color;
    ctx.lineWidth   = 2;
    ctx.stroke();
    ctx.fillStyle   = z.color + "33";
    ctx.fill();
    // Label
    const cx = z.polygon_norm.reduce((s,p)=>s+p[0],0)/z.polygon_norm.length * imgW;
    const cy = z.polygon_norm.reduce((s,p)=>s+p[1],0)/z.polygon_norm.length * imgH;
    ctx.fillStyle = z.color;
    ctx.font = "12px monospace";
    ctx.fillText(z.sku_zone || z.zone_id, cx - 20, cy);
  });
}

function drawCurrentPolygon() {
  if (currentPolygon.length === 0) return;
  ctx.beginPath();
  currentPolygon.forEach(({x,y}, i) => {
    const px = x*imgW, py = y*imgH;
    i === 0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
    ctx.arc(px, py, 4, 0, Math.PI*2);
    ctx.moveTo(px, py);
    i < currentPolygon.length - 1 && ctx.lineTo(
      currentPolygon[i+1].x*imgW, currentPolygon[i+1].y*imgH
    );
  });
  ctx.strokeStyle = "#fff";
  ctx.lineWidth   = 1.5;
  ctx.stroke();
}

canvas.addEventListener("click", e => {
  if (!currentCam || !frames[currentCam]) return;
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width  / rect.width;
  const sy = canvas.height / rect.height;
  const nx = (e.clientX - rect.left)  * sx / imgW;
  const ny = (e.clientY - rect.top)   * sy / imgH;
  currentPolygon.push({x: nx, y: ny});
  drawScene();
});

canvas.addEventListener("contextmenu", e => {
  e.preventDefault();
  closePoly();
});

document.addEventListener("keydown", e => {
  if (e.key === "Enter") closePoly();
});

function closePoly() {
  if (currentPolygon.length < 3) { alert("Need at least 3 points"); return; }
  const color = COLORS[allZones[currentCam].length % COLORS.length];
  const idx   = allZones[currentCam].length + 1;
  const zone  = {
    zone_id:      "ZONE_" + currentCam.split("_").pop() + "_" + idx,
    sku_zone:     "ZONE_" + idx,
    polygon_norm: currentPolygon.map(p => [+p.x.toFixed(4), +p.y.toFixed(4)]),
    color,
  };
  allZones[currentCam].push(zone);
  currentPolygon = [];
  drawScene();
  renderZoneList();
}

function clearLast()    { currentPolygon.pop(); drawScene(); }
function clearCurrent() { currentPolygon = []; drawScene(); }

function renderZoneList() {
  const list = document.getElementById("zone-list");
  list.innerHTML = "";
  (allZones[currentCam] || []).forEach((z, i) => {
    const div = document.createElement("div");
    div.className = "zone-item";
    div.style.borderLeft = "3px solid " + z.color;
    div.innerHTML = `
      <b style="color:${z.color}">${z.zone_id}</b>
      <button class="del-btn" onclick="deleteZone(${i})">✕</button>
      <input type="text" value="${z.sku_zone}" placeholder="SKU label"
             onchange="allZones['${currentCam}'][${i}].sku_zone=this.value;
                       allZones['${currentCam}'][${i}].zone_id='ZONE_'+this.value.toUpperCase().replace(/ /g,'_');">
    `;
    list.appendChild(div);
  });
}

function deleteZone(i) {
  allZones[currentCam].splice(i, 1);
  drawScene();
  renderZoneList();
}

function exportZones() {
  const out = {};
  CAMERAS.forEach(cid => {
    if (allZones[cid] && allZones[cid].length > 0) {
      out[cid] = allZones[cid].map(z => ({
        zone_id:      z.zone_id,
        sku_zone:     z.sku_zone,
        polygon_norm: z.polygon_norm,
        color_bgr:    hexToBGR(z.color),
      }));
    }
  });
  const txt = JSON.stringify(out, null, 2);
  document.getElementById("export-out").textContent = txt;
  // Download
  const blob = new Blob([txt], {type:"application/json"});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url; a.download = "zones_override.json"; a.click();
}

function hexToBGR(hex) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return [b, g, r];
}

// Auto-select first camera
if (CAMERAS.length) selectCam(CAMERAS[0]);
</script>
</body>
</html>
"""


def make_calib_app(clips_dir: str):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="Zone Calibration")
    cam_frames: Dict[str, str] = {}

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html = (CALIB_HTML
                .replace("__CAMERAS__", json.dumps(list(CAMERA_MAP.values())))
                .replace("__COLORS__",  json.dumps(ZONE_COLORS)))
        return html

    @app.get("/frame/{camera_id}")
    async def get_frame(camera_id: str):
        if camera_id in cam_frames:
            return JSONResponse({"frame": cam_frames[camera_id]})
        # Find video
        for num, cid in CAMERA_MAP.items():
            if cid == camera_id:
                for pat in [f"CAM {num}.*", f"cam{num}.*", f"CAM{num}.*"]:
                    matches = list(Path(clips_dir).glob(pat))
                    if matches:
                        b64 = grab_first_frame(str(matches[0]))
                        cam_frames[camera_id] = b64
                        return JSONResponse({"frame": b64})
        # Return blank
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(blank, f"No video for {camera_id}", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        _, buf = cv2.imencode(".jpg", blank)
        b64 = base64.b64encode(bytes(buf)).decode()
        return JSONResponse({"frame": b64})

    return app


def main():
    parser = argparse.ArgumentParser(description="Zone Calibration Tool")
    parser.add_argument("--clips_dir", default="/data/clips")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    import uvicorn
    app = make_calib_app(args.clips_dir)
    print(f"\n  Zone calibration tool: http://localhost:{args.port}")
    print("  Draw polygons in your browser, then Export → zones_override.json")
    print("  Place zones_override.json in the pipeline/ directory\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
