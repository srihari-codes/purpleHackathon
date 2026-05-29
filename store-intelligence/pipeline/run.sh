#!/usr/bin/env bash
# run.sh — Process all CCTV clips and feed events into the API.
# Usage: bash pipeline/run.sh [clips_dir] [layout] [api_url]
# Example: bash pipeline/run.sh data/clips data/store_layout.json http://localhost:8000

set -euo pipefail

CLIPS_DIR="${1:-data/clips}"
LAYOUT="${2:-data/store_layout.json}"
API_URL="${3:-http://localhost:8000}"
OUTPUT_DIR="data/events"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo " Store Intelligence Detection Pipeline"
echo " Clips : $CLIPS_DIR"
echo " Layout: $LAYOUT"
echo " API   : $API_URL"
echo "============================================"

# Wait for API to be ready
echo "Waiting for API to be ready..."
for i in $(seq 1 30); do
  if curl -sf "$API_URL/health" > /dev/null 2>&1; then
    echo "API is ready."
    break
  fi
  echo "  Attempt $i/30 — retrying in 2s..."
  sleep 2
done

# Process each clip
# Expected filename convention:
#   {STORE_ID}__{CAMERA_ID}__{CLIP_START_ISO}.mp4
# Example:
#   STORE_BLR_002__CAM_ENTRY_01__2026-03-03T14-00-00Z.mp4
#
# If clips don't follow this convention, edit the variables below manually.

shopt -s nullglob
CLIPS=("$CLIPS_DIR"/*.mp4 "$CLIPS_DIR"/*.MP4 "$CLIPS_DIR"/*.avi)

if [ ${#CLIPS[@]} -eq 0 ]; then
  echo "No video clips found in $CLIPS_DIR"
  echo "Drop your .mp4 files there and re-run."
  exit 0
fi

for CLIP in "${CLIPS[@]}"; do
  BASENAME=$(basename "$CLIP")
  STEM="${BASENAME%.*}"

  # Parse filename: STORE_ID__CAMERA_ID__TIMESTAMP
  IFS='__' read -r STORE_ID CAMERA_ID CLIP_START <<< "$STEM"

  # Fallback defaults if filename doesn't match convention
  STORE_ID="${STORE_ID:-STORE_BLR_002}"
  CAMERA_ID="${CAMERA_ID:-CAM_FLOOR_01}"
  CLIP_START="${CLIP_START:-2026-03-03T14-00-00Z}"

  # Convert dashes in time to colons: 2026-03-03T14-00-00Z → 2026-03-03T14:00:00Z
  CLIP_START=$(echo "$CLIP_START" | sed 's/T\([0-9][0-9]\)-\([0-9][0-9]\)-\([0-9][0-9]\)/T\1:\2:\3/')

  # Determine camera type from camera_id
  CAMERA_TYPE="main_floor"
  if echo "$CAMERA_ID" | grep -qi "ENTRY"; then
    CAMERA_TYPE="entry_exit"
  elif echo "$CAMERA_ID" | grep -qi "BILLING"; then
    CAMERA_TYPE="billing"
  fi

  OUTPUT_JSONL="$OUTPUT_DIR/${STEM}.jsonl"

  echo ""
  echo "Processing: $BASENAME"
  echo "  Store  : $STORE_ID"
  echo "  Camera : $CAMERA_ID ($CAMERA_TYPE)"
  echo "  Start  : $CLIP_START"
  echo "  Output : $OUTPUT_JSONL"

  python -m pipeline.detect \
    --clip "$CLIP" \
    --store-id "$STORE_ID" \
    --camera-id "$CAMERA_ID" \
    --camera-type "$CAMERA_TYPE" \
    --layout "$LAYOUT" \
    --clip-start "$CLIP_START" \
    --output-jsonl "$OUTPUT_JSONL" \
    --api-url "$API_URL" \
    --every-n-frames 3

  echo "  Done → events written to $OUTPUT_JSONL"
done

echo ""
echo "============================================"
echo " All clips processed."
echo " Check API: $API_URL/health"
echo "============================================"
