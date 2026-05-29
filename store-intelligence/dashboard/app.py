"""
dashboard/app.py — Live terminal dashboard using the `rich` library.

Shows real-time metrics from the Store Intelligence API.
Polls /stores/{store_id}/metrics and /stores/{store_id}/anomalies every 2s.
Also displays funnel and heatmap every 10s.

Usage:
  python dashboard/app.py --store-id STORE_BLR_002 --api-url http://localhost:8000

Part E bonus: proves the pipeline and API are genuinely connected.
"""

import argparse
import time
import sys
import os
from datetime import datetime

import requests
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.align import Align

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
POLL_INTERVAL = 2    # seconds
FULL_REFRESH_INTERVAL = 10  # seconds for funnel/heatmap

console = Console()


def fetch(endpoint: str) -> dict | None:
    try:
        resp = requests.get(f"{API_URL}{endpoint}", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return None


def build_metrics_panel(store_id: str) -> Panel:
    data = fetch(f"/stores/{store_id}/metrics")
    if not data:
        return Panel("[red]⚠ Cannot reach API[/red]", title="📊 Metrics", border_style="red")

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("Metric", style="dim")
    t.add_column("Value", style="bold")

    t.add_row("🧑 Unique Visitors", str(data.get("unique_visitors", 0)))
    conv = data.get("conversion_rate", 0)
    conv_color = "green" if conv >= 0.3 else "yellow" if conv >= 0.15 else "red"
    t.add_row("💰 Conversion Rate", f"[{conv_color}]{conv*100:.1f}%[/{conv_color}]")
    t.add_row("🔢 Queue Depth", str(data.get("queue_depth", 0)))
    abandon = data.get("abandonment_rate", 0)
    t.add_row("🚶 Abandonment Rate", f"{abandon*100:.1f}%")
    t.add_row("🕐 Window", f"{data.get('window_start','')[:16]} → {data.get('window_end','')[:16]}")

    return Panel(t, title=f"📊 Real-Time Metrics — {store_id}", border_style="cyan")


def build_funnel_panel(store_id: str) -> Panel:
    data = fetch(f"/stores/{store_id}/funnel")
    if not data:
        return Panel("[dim]No funnel data[/dim]", title="🔻 Funnel", border_style="dim")

    t = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    t.add_column("Stage", style="bold")
    t.add_column("Count", justify="right")
    t.add_column("Drop-off", justify="right", style="red")

    for stage in data.get("stages", []):
        drop = stage.get("drop_off_pct", 0)
        drop_str = f"{drop:.1f}%" if drop > 0 else "—"
        t.add_row(stage.get("stage", "?"), str(stage.get("count", 0)), drop_str)

    return Panel(t, title="🔻 Conversion Funnel", border_style="magenta")


def build_heatmap_panel(store_id: str) -> Panel:
    data = fetch(f"/stores/{store_id}/heatmap")
    if not data or not data.get("zones"):
        return Panel("[dim]No heatmap data[/dim]", title="🗺 Heatmap", border_style="dim")

    t = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    t.add_column("Zone", style="bold")
    t.add_column("Visits", justify="right")
    t.add_column("Avg Dwell", justify="right")
    t.add_column("Heat", justify="left")

    BARS = "▁▂▃▄▅▆▇█"

    for zone in sorted(data["zones"], key=lambda z: z.get("normalised_score", 0), reverse=True):
        score = zone.get("normalised_score", 0)
        bar_idx = int(score / 100 * (len(BARS) - 1))
        bar = BARS[bar_idx]
        dwell_s = zone.get("avg_dwell_ms", 0) / 1000
        conf = "" if zone.get("data_confidence", True) else " ⚠"
        t.add_row(
            zone.get("zone_id", "?") + conf,
            str(zone.get("visit_frequency", 0)),
            f"{dwell_s:.0f}s",
            f"[{'green' if score > 60 else 'yellow' if score > 30 else 'red'}]{bar * 8}[/]",
        )

    return Panel(t, title="🗺 Zone Heatmap", border_style="green")


def build_anomalies_panel(store_id: str) -> Panel:
    data = fetch(f"/stores/{store_id}/anomalies")
    if not data:
        return Panel("[dim]No anomaly data[/dim]", title="⚠ Anomalies", border_style="dim")

    anomalies = data.get("active_anomalies", [])
    if not anomalies:
        return Panel(
            "[green]✓ No active anomalies[/green]",
            title="⚠ Anomalies",
            border_style="green",
        )

    t = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    t.add_column("Type", style="bold")
    t.add_column("Severity")
    t.add_column("Description")

    SEV_COLOR = {"INFO": "blue", "WARN": "yellow", "CRITICAL": "red"}

    for a in anomalies:
        sev = a.get("severity", "INFO")
        color = SEV_COLOR.get(sev, "white")
        t.add_row(
            a.get("anomaly_type", "?"),
            f"[{color}]{sev}[/{color}]",
            a.get("description", "")[:60],
        )

    return Panel(t, title="⚠ Active Anomalies", border_style="red")


def build_header(store_id: str) -> Panel:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return Panel(
        Align.center(
            Text(f"🏪 Store Intelligence Dashboard  ·  {store_id}  ·  {now}", style="bold white")
        ),
        style="on #1a1a2e",
        border_style="bright_cyan",
    )


def make_layout(store_id: str, full: bool) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(build_header(store_id), name="header", size=3),
        Layout(name="top", size=12),
        Layout(name="bottom"),
    )
    layout["top"].split_row(
        Layout(build_metrics_panel(store_id), name="metrics"),
        Layout(build_anomalies_panel(store_id), name="anomalies"),
    )
    if full:
        layout["bottom"].split_row(
            Layout(build_funnel_panel(store_id), name="funnel"),
            Layout(build_heatmap_panel(store_id), name="heatmap"),
        )
    else:
        layout["bottom"].update(
            Panel("[dim]Funnel & Heatmap refresh in a few seconds...[/dim]",
                  border_style="dim", title="📈 Analytics")
        )
    return layout


def run_dashboard(store_id: str, api_url: str):
    global API_URL
    API_URL = api_url

    console.print(f"\n[bold cyan]Starting live dashboard for {store_id}[/bold cyan]")
    console.print(f"API: {api_url}  ·  Polling every {POLL_INTERVAL}s\n")

    last_full = 0.0

    with Live(make_layout(store_id, True), refresh_per_second=1, screen=True) as live:
        while True:
            now = time.time()
            full = (now - last_full) >= FULL_REFRESH_INTERVAL
            if full:
                last_full = now
            live.update(make_layout(store_id, full))
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Store Intelligence Live Dashboard")
    parser.add_argument("--store-id", default="STORE_BLR_002")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()
    try:
        run_dashboard(args.store_id, args.api_url)
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/dim]")
