# USV Obstacle Detection and Localization System

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Status](https://img.shields.io/badge/status-research--preview-orange)

## Overview
This repository contains the source code for the Master's Thesis: **"Development of a software module for detection and spatial localization of surface obstacles for an Unmanned Surface Vehicle (USV) based on intelligent multi-sensor data processing."**

The goal is to implement a research-grade pipeline that processes raw data from stereo cameras and sensors (IMU, GPS) to generate a local obstacle map for navigation.

## Repository Structure
The project follows a modified Cookiecutter Data Science structure:

```text
├── configs/        # Configuration files (Hydra/YAML)
├── data/           # Data storage (ignored by Git, keep local)
│   ├── raw/        # Original video/sensor logs
│   └── processed/  # Synchronized and rectified data
├── notebooks/      # Jupyter notebooks for experiments and EDA
├── src/            # Source code
│   ├── data/       # Data loaders and parsers
│   ├── preprocessing/ # Calibration, Synchronization, Rectification
│   ├── perception/    # Neural networks (Segmentation/Detection)
│   └── mapping/       # 3D reconstruction and map generation
├── reports/        # Generated analysis (figures, metrics)
└── requirements.txt # Python dependencies
```

### Prerequisites
*   Python 3.11+
*   [FFmpeg](https://ffmpeg.org/download.html) installed and added to system PATH (required for video processing).
*   [uv](https://github.com/astral-sh/uv) (for dependency management).

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/USV-Obstacle-Detection.git
    cd USV-Obstacle-Detection
    ```

2.  **Install uv (if not installed):**
    *   **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
    *   **macOS/Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`

3.  **Sync environment:**
    This command will install Python 3.11, create a virtual environment, and install all locked dependencies.
    ```bash
    uv sync
    ```

4.  **Activate environment:**
    *   **Windows:** `.venv\Scripts\activate`
    *   **macOS/Linux:** `source .venv/bin/activate`
