# Detection Layer Dockerfile
# ──────────────────────────────────────────────────────────────────────────────
# GPU / CPU auto-selection logic:
#
#   By default the CUDA 12.1 base image is used and PyTorch is installed with
#   the cu121 wheel index so CUDA is actually available at runtime.
#
#   Build args:
#     BASE_IMAGE  — swap base image (default: nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04)
#     USE_CUDA    — "1" force CUDA torch  |  "0" force CPU torch  |  "auto" (default)
#                   When "auto" the script installs CUDA torch if the base image
#                   name contains "cuda", otherwise CPU torch.
#
#   At runtime the pipeline auto-detects CUDA via torch.cuda.is_available();
#   no manual flag needed.
#
# Examples:
#   # GPU build (default):
#   docker build -f pipeline/Dockerfile -t store-intelligence .
#
#   # CPU-only build (CI / machines without NVIDIA):
#   docker build --build-arg BASE_IMAGE=ubuntu:22.04 --build-arg USE_CUDA=0 \
#                -f pipeline/Dockerfile -t store-intelligence .
# ──────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

LABEL description="Purplle Store Intelligence — Detection Layer"

# Re-declare after FROM so the value is available in RUN steps
ARG BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
ARG USE_CUDA=auto

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

# ── PyTorch: CUDA or CPU wheel (decided at build time) ───────────────────
# We handle torch/torchvision here — NOT in requirements.txt — so we can
# choose the right binary index based on the base image.
#
# Logic:
#   USE_CUDA=1   → always install CUDA (cu121) torch
#   USE_CUDA=0   → always install CPU torch
#   USE_CUDA=auto → install CUDA torch when BASE_IMAGE contains "cuda",
#                   CPU torch otherwise
RUN set -eux; \
    CUDA_BASE=$(echo "$BASE_IMAGE" | grep -ic cuda || true); \
    if [ "$USE_CUDA" = "1" ] || { [ "$USE_CUDA" = "auto" ] && [ "$CUDA_BASE" -gt 0 ]; }; then \
        echo ">>> Installing PyTorch with CUDA 12.1 support"; \
        pip install --no-cache-dir \
            "torch>=2.1.0" "torchvision>=0.16.0" \
            --index-url https://download.pytorch.org/whl/cu121; \
    else \
        echo ">>> Installing PyTorch CPU-only"; \
        pip install --no-cache-dir "torch>=2.1.0" "torchvision>=0.16.0"; \
    fi

# ── Remaining Python dependencies (everything except torch) ──────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── YOLO model ────────────────────────────────────────────────────────────
RUN mkdir -p /opt/yolo_models && \
    curl -L -o /opt/yolo_models/yolo11m.pt \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt

# ── Environment variables ─────────────────────────────────────────────────
ENV YOLO_MODEL=/opt/yolo_models/yolo11m.pt
# Put YOLO config in /tmp to avoid "not writable" warning in read-only containers
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# ── Copy pipeline source ──────────────────────────────────────────────────
WORKDIR /app/pipeline
COPY pipeline/ /app/pipeline/

# ── Data mount points ─────────────────────────────────────────────────────
# /data/clips   — video files (CAM 1.mp4 … CAM 5.mp4) mounted at runtime
# /data         — output events.jsonl written here
RUN mkdir -p /data/clips

# ── Entry point ───────────────────────────────────────────────────────────
EXPOSE 8080

# Default: run the full pipeline at max speed (SPEED=0)
# Override with -e SPEED=1.0 for real-time pacing
ENTRYPOINT ["/bin/bash", "/app/pipeline/run.sh"]
