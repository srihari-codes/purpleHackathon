#!/usr/bin/env bash
# run.sh — Process all camera clips and emit events.jsonl
# Usage:
#   ./run.sh                         # default paths
#   CLIPS_DIR=/my/clips ./run.sh     # custom clips dir
#   SPEED=0 ./run.sh                 # max speed (no real-time pacing)
#
# Environment variables:
#   CLIPS_DIR       path to directory containing CAM 1.mp4 … CAM 5.mp4
#   OUTPUT          path for output events.jsonl
#   STORE_ID        store ID string
#   GUI_PORT        port for web dashboard (default 8080)
#   SPEED           playback speed multiplier (default 1.0; 0 = max)
#   CAM3_START_ISO  ISO-8601 UTC override for entry cam start time
#   LOG_LEVEL       DEBUG / INFO / WARNING

set -euo pipefail

CLIPS_DIR="${CLIPS_DIR:-/data/clips}"
OUTPUT="${OUTPUT:-/data/events.jsonl}"
STORE_ID="${STORE_ID:-STORE_BLR_002}"
GUI_PORT="${GUI_PORT:-8080}"
SPEED="${SPEED:-1.0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
CAM3_START_ISO="${CAM3_START_ISO:-}"

# Move to pipeline directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo " Store Intelligence — Detection Layer"
echo "========================================"
echo "  Store:     $STORE_ID"
echo "  Clips:     $CLIPS_DIR"
echo "  Output:    $OUTPUT"
echo "  GUI:       http://localhost:$GUI_PORT"
echo "  Speed:     ${SPEED}x"
echo "========================================"

# Build command
CMD=(
    python detect.py
    --store_id     "$STORE_ID"
    --clips_dir    "$CLIPS_DIR"
    --output       "$OUTPUT"
    --gui_port     "$GUI_PORT"
    --speed        "$SPEED"
    --log_level    "$LOG_LEVEL"
)

if [ -n "$CAM3_START_ISO" ]; then
    CMD+=(--cam3_start_iso "$CAM3_START_ISO")
fi

echo ""
echo "Running: ${CMD[*]}"
echo ""
exec "${CMD[@]}"
