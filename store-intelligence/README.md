# Purplle Store Intelligence — Quickstart Guide

Welcome to the Purplle Store Intelligence project! Follow this guide to set up the CCTV video feeds and run the computer vision pipeline from scratch.

---

## Prerequisites

Ensure you have the following installed on your host system:
1. **Docker** and **Docker Compose**
2. **Git**
3. (Optional but recommended) **NVIDIA Container Toolkit** for full GPU/CUDA acceleration. The startup script will automatically fall back to CPU mode if NVIDIA GPU drivers are not present.

---

## Setup & Running Instructions

### Step 1: Clone the Repository
Clone the repository and navigate to the project directory:
```bash
git clone <your-repository-url>
cd purpleHackathon
```

---

### Step 2: Position the CCTV Footage
The detection pipeline requires the 5 CCTV video feeds (`CAM 1.mp4`, `CAM 2.mp4`, `CAM 3.mp4`, `CAM 4.mp4`, and `CAM 5.mp4`).

Choose **one** of the two setup methods below:

#### Option A: Project Local Clips Folder (Recommended & Easiest)
Simply place your 5 `CAM *.mp4` files inside the project's local clips directory:
`store-intelligence/data/clips/`

#### Option B: Use any Custom Folder
You can keep the files anywhere on your machine (e.g., inside your `Downloads/CCTV Footage` folder) and pass the path using the `CLIPS_DIR` environment variable when starting.

---

### Step 3: Start the Pipeline

First, navigate into the project's root folder:
```bash
cd store-intelligence
```

Depending on the setup option you chose in **Step 2**, run the corresponding command:

#### If you set up Option A (Default folder):
Simply execute the run script:
```bash
./run.sh
```

#### If you set up Option B (Custom folder):
Provide the absolute path to your folder via `CLIPS_DIR`:
```bash
CLIPS_DIR="/path/to/your/CCTV Footage" ./run.sh
```

---

## Overall Docker Design

* **Automatic Hardware Detection:** The `./run.sh` script automatically checks for GPU availability. If CUDA is detected, it configures complete hardware-accelerated tracking. If not, it falls back to a clean CPU pipeline automatically.
* **Events File Persistence:** Output events are written to `store-intelligence/data/events.jsonl`. This file is persisted on your host and is ignored in git so that your local testing outputs do not conflict with the code repository.
* **Development Code Mounting:** The local `pipeline/` directory is mounted directly into the container in real-time, allowing you to edit python modules on your host and see changes instantly without rebuilding.

---

## Running the Services Separately

### 1. View the Main GUI Dashboard
Once the container starts, open your browser and navigate to:
```text
http://localhost:8080
```
This launches the real-time event analytics dashboard.

### 2. Run the Interactive Calibration UI
If you need to calibrate entry/exit crossing lines or polygon boundaries, run:
```bash
# If Option A
docker compose up calibrate

# If Option B
CLIPS_DIR="/path/to/your/CCTV Footage" docker compose up calibrate
```
Then navigate to `http://localhost:8081` in your browser. Any custom polygons you draw will be saved directly back to `pipeline/zones_override.json` on your host machine.
