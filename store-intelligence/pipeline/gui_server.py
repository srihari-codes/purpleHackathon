"""
gui_server.py — Web GUI for real-time detection debugging.

Serves:
  GET  /          → dashboard HTML page
  WS   /ws/frames → MJPEG-over-WebSocket frame stream (base64 JPEG)
  WS   /ws/events → live event JSON stream
  GET  /api/state → current snapshot (visitor count, staff, queue, etc.)

The pipeline calls push_frame(camera_id, annotated_bgr) and the server
broadcasts it to connected browser clients.
"""

import asyncio
import base64
import json
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Dict, Optional, Set
import queue as stdlib_queue

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process state shared between pipeline thread and async server
# ---------------------------------------------------------------------------

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        # camera_id → latest annotated JPEG bytes
        self.latest_frames: Dict[str, bytes] = {}
        # ring buffer of last 200 events
        self.event_log: deque = deque(maxlen=200)
        # ring buffer of last 50 warnings
        self.warnings_log: deque = deque(maxlen=50)
        # live metrics
        self.metrics: dict = {
            "active_visitors":  0,
            "staff_count":      0,
            "queue_depth":      0,
            "total_entries":    0,
            "total_exits":      0,
            "total_events":     0,
            "cameras_active":   [],
        }
        # asyncio queues for WebSocket broadcast (set by server startup)
        self._frame_queues: Dict[str, asyncio.Queue] = {}
        self._event_queues: Set[asyncio.Queue] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    # Called from pipeline (sync) thread --------------------------------
    def push_frame(self, camera_id: str, jpeg_bytes: bytes, ts: str = ""):
        with self.lock:
            self.latest_frames[camera_id] = jpeg_bytes
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._broadcast_frame(camera_id, jpeg_bytes, ts), self._loop
            )

    def push_event(self, event_dict: dict):
        with self.lock:
            self.event_log.appendleft(event_dict)
            self.metrics["total_events"] += 1
            etype = event_dict.get("event_type", "")
            if etype == "ENTRY":
                self.metrics["total_entries"] += 1
            elif etype == "EXIT":
                self.metrics["total_exits"] += 1
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._broadcast_event(event_dict), self._loop
            )

    def push_warning(self, warning_dict: dict):
        with self.lock:
            self.warnings_log.appendleft(warning_dict)
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._broadcast_event(warning_dict), self._loop
            )

    def update_metrics(self, active_visitors: int, staff_count: int,
                       queue_depth: int, cameras_active: list):
        with self.lock:
            self.metrics["active_visitors"] = active_visitors
            self.metrics["staff_count"]     = staff_count
            self.metrics["queue_depth"]     = queue_depth
            self.metrics["cameras_active"]  = cameras_active

    # Async broadcast helpers -------------------------------------------
    async def _broadcast_frame(self, camera_id: str, jpeg_bytes: bytes, ts: str = ""):
        b64 = base64.b64encode(jpeg_bytes).decode()
        msg = json.dumps({"camera_id": camera_id, "frame": b64, "timestamp": ts})
        dead = set()
        for q in list(self._frame_queues.values()):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass
            except Exception:
                dead.add(q)

    async def _broadcast_event(self, event_dict: dict):
        msg = json.dumps(event_dict)
        dead = set()
        for q in list(self._event_queues):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass
            except Exception:
                dead.add(q)
        self._event_queues -= dead


SHARED = SharedState()


# ---------------------------------------------------------------------------
# FastAPI server
# ---------------------------------------------------------------------------
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Store Intelligence — Detection Dashboard</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d1117; color: #c9d1d9; font-family: monospace; }
h1  { padding: 12px 16px; background: #161b22; border-bottom: 1px solid #30363d;
      font-size: 1rem; letter-spacing: .05em; color: #58a6ff; }
#main { display: flex; flex-wrap: wrap; gap: 12px; padding: 12px; }
.cam-panel { flex: 1 1 380px; background: #161b22; border: 1px solid #30363d;
             border-radius: 6px; overflow: hidden; }
.cam-label { padding: 4px 8px; font-size: .75rem; color: #8b949e;
             border-bottom: 1px solid #21262d; }
.cam-panel img { width: 100%; display: block; background: #000; min-height: 220px; }
#sidebar { flex: 0 0 340px; display: flex; flex-direction: column; gap: 12px; }
#metrics { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
           padding: 12px; }
#metrics h2 { font-size: .8rem; color: #8b949e; margin-bottom: 8px; }
.metric { display: flex; justify-content: space-between; padding: 3px 0;
          border-bottom: 1px solid #21262d; font-size: .85rem; }
.metric span:last-child { color: #58a6ff; font-weight: bold; }
/* Queue depth visual bar */
#queue-bar-wrap { margin-top: 6px; }
#queue-bar { height: 6px; background: #e3b341; border-radius: 3px;
             transition: width 0.4s ease; width: 0%; max-width: 100%; }
#panels-wrap { flex: 1; overflow: hidden; display: flex; flex-direction: column; gap: 12px; }
#events { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
          flex: 1; display: flex; flex-direction: column; min-height: 200px; }
#events h2, #warnings h2 { font-size: .8rem; color: #8b949e; padding: 8px 12px;
             border-bottom: 1px solid #21262d; }
#event-list, #warning-list { overflow-y: auto; flex: 1; padding: 4px 0; }
#warnings { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
            flex: 0 0 150px; display: flex; flex-direction: column; }
.ev, .warn { padding: 4px 12px; font-size: .70rem; border-bottom: 1px solid #0d1117;
       cursor: default; }
.ev:hover, .ghost:hover, .court:hover { background: #21262d; }
.ev .etype { font-weight: bold; min-width: 120px; display: inline-block; }
.ev .bstate { font-size:.62rem; background:#21262d; padding:1px 5px;
              border-radius:3px; color:#8b949e; margin-left:4px; }
.ev .confbadge { font-size:.60rem; color:#3fb950; margin-left:4px; }
.ENTRY    { color: #56d364; }
.EXIT     { color: #f85149; }
.REENTRY  { color: #d2a8ff; }
.ZONE_ENTER { color: #79c0ff; }
.ZONE_EXIT  { color: #ffa657; }
.ZONE_DWELL { color: #8b949e; }
.BILLING_QUEUE_JOIN    { color: #e3b341; }
.BILLING_QUEUE_ABANDON { color: #f85149; }
.warn .etype { color: #ffa657; font-weight: bold; }
.warn.CRITICAL .etype { color: #f85149; }
#status { font-size: .7rem; color: #8b949e; padding: 4px 12px; }
</style>
</head>
<body>
<h1>🏪 Store Intelligence — Detection Layer Dashboard</h1>
<div id="main">
  <div style="flex:1 1 100%; display:flex; flex-wrap:wrap; gap:12px;" id="cams">
    <!-- camera panels injected by JS -->
  </div>
  <div id="sidebar">
    <div id="metrics">
      <h2>LIVE METRICS</h2>
      <div class="metric"><span>Active Visitors</span><span id="m-visitors">—</span></div>
      <div class="metric"><span>Staff Detected</span><span id="m-staff">—</span></div>
      <div class="metric"><span>Queue Depth</span><span id="m-queue">—</span></div>
      <div id="queue-bar-wrap"><div id="queue-bar"></div></div>
      <div class="metric"><span>Total Entries</span><span id="m-entries">—</span></div>
      <div class="metric"><span>Total Exits</span><span id="m-exits">—</span></div>
      <div class="metric"><span>Events Emitted</span><span id="m-events">—</span></div>
      <div class="metric"><span>Cameras Active</span><span id="m-cams">—</span></div>
    </div>
    <div id="panels-wrap">
      <div id="warnings">
        <h2>SYSTEM AUDITOR WARNINGS</h2>
        <div id="warning-list"></div>
      </div>
      <div id="events">
        <h2>LIVE EVENT LOG</h2>
        <div id="event-list"></div>
      </div>
      <div id="ghosts" style="background:#161b22; border:1px solid #30363d; border-radius:6px; flex:1; display:flex; flex-direction:column; min-height:150px;">
        <h2 style="font-size:.8rem; color:#8b949e; padding:8px 12px; border-bottom:1px solid #21262d;">👻 GHOST IDENTITIES</h2>
        <div id="ghost-list" style="overflow-y:auto; flex:1; padding:4px 0;"></div>
      </div>
      <div id="courtroom" style="background:#161b22; border:1px solid #30363d; border-radius:6px; flex:1; display:flex; flex-direction:column; min-height:150px;">
        <h2 style="font-size:.8rem; color:#8b949e; padding:8px 12px; border-bottom:1px solid #21262d;">⚖️ IDENTITY COURTROOM</h2>
        <div id="court-list" style="overflow-y:auto; flex:1; padding:4px 0;"></div>
      </div>
    </div>
    <div id="status">connecting...</div>
  </div>
</div>
<script>
const CAMERAS = ["CAM_ENTRY_03","CAM_FLOOR_01","CAM_FLOOR_02","CAM_BILLING_05","CAM_GODOWN_04"];
const camImgs = {};

// Build camera panels
const camsDiv = document.getElementById("cams");
CAMERAS.forEach(cid => {
  const panel = document.createElement("div");
  panel.className = "cam-panel";
  panel.innerHTML = `<div class="cam-label">${cid} <span id="ts-${cid}" style="float:right; color:#58a6ff; font-weight:bold;"></span></div><img id="img-${cid}" src="" alt="${cid}">`;
  camsDiv.appendChild(panel);
  camImgs[cid] = document.getElementById("img-"+cid);
});

// Frame WebSocket
const wsf = new WebSocket(`ws://${location.host}/ws/frames`);
wsf.onmessage = e => {
  const d = JSON.parse(e.data);
  const img = camImgs[d.camera_id];
  if (img) {
      img.src = "data:image/jpeg;base64," + d.frame;
      if (d.timestamp) {
          const tsSpan = document.getElementById("ts-" + d.camera_id);
          if (tsSpan) tsSpan.textContent = d.timestamp;
      }
  }
};
wsf.onopen = () => { document.getElementById("status").textContent = "frames: connected"; };
wsf.onclose= () => { document.getElementById("status").textContent = "frames: disconnected"; };

// Event WebSocket
const wse = new WebSocket(`ws://${location.host}/ws/events`);
const evList = document.getElementById("event-list");
wse.onmessage = e => {
  const ev = JSON.parse(e.data);
  if (ev._type === "metrics") {
    document.getElementById("m-visitors").textContent = ev.active_visitors;
    document.getElementById("m-staff").textContent    = ev.staff_count;
    document.getElementById("m-queue").textContent    = ev.queue_depth;
    document.getElementById("m-entries").textContent  = ev.total_entries;
    document.getElementById("m-exits").textContent    = ev.total_exits;
    document.getElementById("m-events").textContent   = ev.total_events;
    document.getElementById("m-cams").textContent     = (ev.cameras_active||[]).join(", ");
    // Queue bar: max display at 10 people
    const qPct = Math.min((ev.queue_depth || 0) / 10 * 100, 100);
    document.getElementById("queue-bar").style.width = qPct + "%";
    return;
  }
  if (ev._type === "warning") {
    const div = document.createElement("div");
    div.className = "warn " + ev.severity;
    const ts = (ev.timestamp||"").substring(11,19);
    div.innerHTML = `<span style="color:#8b949e">${ts}</span> ` +
                    `<span class="etype">${ev.anomaly_type}</span><br>` +
                    `<span>${ev.description}</span>`;
    const wl = document.getElementById("warning-list");
    wl.insertBefore(div, wl.firstChild);
    while (wl.children.length > 50) wl.removeChild(wl.lastChild);
    return;
  }
  const div = document.createElement("div");
  div.className = "ev";
  const ts = (ev.timestamp||"").substring(11,19);
  const meta   = ev.metadata || {};
  const bstate = meta.behavior_state || "";
  const conf   = ev.confidence != null ? ev.confidence.toFixed(3) : "";
  div.innerHTML =
    `<span class="etype ${ev.event_type}">${ev.event_type}</span>` +
    `<span style="color:#8b949e"> ${ts}</span> ` +
    `<span style="color:#79c0ff">${ev.visitor_id||""}</span>` +
    (ev.zone_id ? ` <span style="color:#e3b341">${ev.zone_id}</span>` : "") +
    (bstate ? `<span class="bstate">${bstate}</span>` : "") +
    (conf   ? `<span class="confbadge">⬤${conf}</span>` : "") +
    (ev.is_staff ? ` <span style="color:#f85149">[STAFF]</span>` : "");
  evList.insertBefore(div, evList.firstChild);
  while (evList.children.length > 200) evList.removeChild(evList.lastChild);
};

// Polling for Ghosts and Courtroom
setInterval(async () => {
    try {
        const res = await fetch("/api/ghosts");
        const ghosts = await res.json();
        const gl = document.getElementById("ghost-list");
        gl.innerHTML = ghosts.map(g => `<div class="ev ghost" style="padding:4px 12px; font-size:.70rem; border-bottom:1px solid #0d1117;">` +
            `<span style="color:#79c0ff">${g.visitor_id}</span> @ ${g.predicted_camera_id} ` +
            `(TTL: ${Math.max(0, g.expire_at - Date.now()/1000).toFixed(1)}s)</div>`).join("");
            
        const eres = await fetch("/api/evidence");
        const evids = await eres.json();
        const cl = document.getElementById("court-list");
        cl.innerHTML = evids.filter(e=>e.courtroom_verdict).slice(-10).reverse().map(e => {
            const v = e.courtroom_verdict;
            const color = v.should_match ? "#56d364" : "#f85149";
            return `<div class="ev court" style="padding:4px 12px; font-size:.70rem; border-bottom:1px solid #0d1117;">` +
            `<span style="color:${color}; font-weight:bold">${v.should_match ? "MATCH" : "REJECT"}</span> ` +
            `<span style="color:#79c0ff">${e.visitor_id}</span> - ${v.judge_rationale}</div>`;
        }).join("");
    } catch(e){}
}, 2000);
</script>
</body>
</html>
"""


def create_app(shared: SharedState):
    """Create and return the FastAPI app. Import here to avoid top-level dep."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import asyncio

    app = FastAPI(title="Detection Dashboard")

    @app.on_event("startup")
    async def _startup():
        loop = asyncio.get_event_loop()
        shared.set_loop(loop)
        logger.info("GUI server started")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        import os
        path = os.path.join(os.path.dirname(__file__), "wizard.html")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h3>wizard.html not found</h3>", status_code=404)

    @app.get("/api/state")
    async def state():
        with shared.lock:
            return {
                "metrics":    shared.metrics,
                "recent_events": list(shared.event_log)[:20],
                "recent_warnings": list(shared.warnings_log)[:20],
            }

    @app.get("/api/visitor/{visitor_id}/explanation")
    async def explanation(visitor_id: str):
        if hasattr(shared, "identity_mgr") and shared.identity_mgr:
            expl = shared.identity_mgr.last_explanation(visitor_id)
            if expl:
                return expl
        return {"error": "Explanation not found"}

    @app.get("/api/ghosts")
    async def ghosts():
        if hasattr(shared, "identity_mgr") and shared.identity_mgr and hasattr(shared.identity_mgr, "_ghosts"):
            import time
            now = time.time()
            return [
                {
                    "visitor_id": g.visitor_id,
                    "predicted_camera_id": g.last_camera,
                    "expire_at": g.created_at + g.ttl_sec,
                    "conf": g.last_confidence
                }
                for g in shared.identity_mgr._ghosts.get_all()
                if not g.is_expired(now)
            ]
        return []
        
    @app.get("/api/evidence")
    async def evidence():
        if hasattr(shared, "identity_mgr") and shared.identity_mgr and hasattr(shared.identity_mgr, "_evidence"):
            ev = shared.identity_mgr._evidence
            all_snaps = []
            for vid in ev.all_visitor_ids():
                for snap in ev.get_ledger(vid):
                    d = snap.to_dict()
                    d["courtroom_verdict"] = d.pop("courtroom", None)
                    all_snaps.append(d)
            return all_snaps[-200:]  # cap to last 200
        return []

    @app.websocket("/ws/frames")
    async def ws_frames(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        # Register this client's queue
        # We use a dummy camera_id key per client
        client_key = id(ws)
        shared._frame_queues[client_key] = q
        try:
            while True:
                msg = await asyncio.wait_for(q.get(), timeout=30.0)
                await ws.send_text(msg)
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        finally:
            shared._frame_queues.pop(client_key, None)

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        shared._event_queues.add(q)
        try:
            while True:
                # Interleave store events and periodic metrics pushes
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=2.0)
                    await ws.send_text(msg)
                except asyncio.TimeoutError:
                    # Push metrics heartbeat
                    with shared.lock:
                        m = dict(shared.metrics)
                    m["_type"] = "metrics"
                    await ws.send_text(json.dumps(m))
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            shared._event_queues.discard(q)

    # Register wizard routes
    try:
        from wizard_backend import register_wizard_routes
    except ImportError:
        from pipeline.wizard_backend import register_wizard_routes
    
    try:
        register_wizard_routes(app, shared)
    except Exception as e:
        logger.error(f"Failed to register wizard routes: {e}")

    return app


def run_server(host: str = "0.0.0.0", port: int = 8080, shared: SharedState = SHARED):
    """Run the GUI server in a background thread."""
    import uvicorn
    app = create_app(shared)
    config = uvicorn.Config(app, host=host, port=port,
                            log_level="warning", loop="asyncio")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    logger.info(f"GUI dashboard: http://{host}:{port}")
    return t
