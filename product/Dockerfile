# Detection Layer Dockerfile
# Supports CUDA (nvidia/cuda base) with automatic CPU fallback.
# Build args:
#   BASE_IMAGE  — override base image (default: nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04)
#
# At runtime the pipeline auto-detects CUDA availability; no manual flag needed.

ARG BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

LABEL description="Purplle Store Intelligence — Detection Layer"

# ── System dependencies ────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    python3.11-venv \
    # OpenCV runtime dependencies
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libopencv-dev \
    # ffmpeg for video decode (handles .mp4 and .mp3 containers)
    ffmpeg \
    # Tesseract OCR for timestamp watermark extraction
    tesseract-ocr \
    # Build tools
    build-essential git wget curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python setup ──────────────────────────────────────────────────────────
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python  && \
    python3 -m pip install --upgrade pip setuptools wheel

# ── Python dependencies ───────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Pre-download YOLOv8 nano weights ─────────────────────────────────────
# This avoids a download at first run inside the container.
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

# ── Copy pipeline source ──────────────────────────────────────────────────
WORKDIR /app/pipeline
COPY pipeline/ /app/pipeline/

# ── Data mount points ─────────────────────────────────────────────────────
# /data/clips   — video files (CAM 1.mp4 … CAM 5.mp4) mounted at runtime
# /data         — output events.jsonl written here
RUN mkdir -p /data/clips

# ── Entry point ───────────────────────────────────────────────────────────
EXPOSE 8080

# Default: run the full pipeline at real-time speed
# Override with docker run -e SPEED=0 to run at max speed
ENTRYPOINT ["/bin/bash", "/app/pipeline/run.sh"]
