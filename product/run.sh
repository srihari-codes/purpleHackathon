#!/bin/bash

# Ensure we're in the script's directory
cd "$(dirname "$0")" || exit 1

# Check if nvidia-smi is available and NVIDIA Container Runtime is installed
if command -v nvidia-smi &> /dev/null && docker info | grep -i "Runtimes:.*nvidia" &> /dev/null; then
    echo "✅ GPU and NVIDIA Container Runtime detected. Starting with GPU support..."
    docker compose up --build -d
else
    echo "⚠️  No GPU or NVIDIA runtime detected. Falling back to CPU mode..."
    # Strip the GPU deployment reservations dynamically to avoid docker compose validation errors
    sed '/deploy:/,/gpu ]/d' docker-compose.yml > docker-compose-cpu.tmp.yml
    docker compose -f docker-compose-cpu.tmp.yml up --build -d
    rm -f docker-compose-cpu.tmp.yml
fi

echo "🚀 Pipeline is starting! View logs with: docker compose logs -f pipeline"
