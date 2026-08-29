<div align="center">

# 🎾 TennisVision

### End-to-End Tennis Match Analysis with Computer Vision

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?logo=data:image/svg+xml;base64,)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <img src="output_videos/output_video.gif" alt="TennisVision Demo" width="800"/>
</p>

> A production-quality computer vision pipeline that automatically tracks players and the ball, detects court geometry, maps everything onto a mini-court overlay, and computes per-shot biomechanical statistics — all from raw broadcast video.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Project Structure](#-project-structure)
- [Technical Deep Dive](#-technical-deep-dive)
- [Installation](#-installation)
- [Usage](#-usage)
- [Output & Statistics](#-output--statistics)
- [Models](#-models)
- [Configuration](#-configuration)
- [Results](#-results)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

TennisVision is a fully automated sports analytics system that processes broadcast tennis footage and outputs an annotated video enriched with:

- **Player bounding boxes** tracked across all frames
- **Ball trajectory** with gap-filling interpolation
- **Court keypoint detection** using a custom CNN
- **Mini-court overlay** mapping real-world positions from pixel space
- **Live statistics panel** showing shot speed, player speed, and shot counts per player

The system fuses classical computer vision with modern deep learning, integrating YOLOv8 for person detection, a fine-tuned YOLOv5 for ball detection, and a custom ResNet-based keypoint regressor for court geometry.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🧍 Player Tracking | YOLOv8-powered detection + filtering to the two active players |
| 🎾 Ball Tracking | Fine-tuned YOLOv5 + Pandas interpolation to fill missing frames |
| 🏟️ Court Detection | Custom CNN regressor predicting 14 court keypoints |
| 🗺️ Mini-Court Overlay | Homography-based coordinate transformation to top-down view |
| 📊 Live Stats | Per-shot speed (km/h), player movement speed, and shot count |
| ⚡ Stub Caching | Detection results cached as `.pkl` stubs to skip re-inference |
| 🎬 Video Output | Fully annotated `.avi` output ready for review |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT VIDEO                                 │
│                     (broadcast tennis footage)                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                       ▼
  ┌───────────────┐    ┌────────────────┐    ┌──────────────────────┐
  │ PlayerTracker │    │  BallTracker   │    │  CourtLineDetector   │
  │  (YOLOv8m)   │    │  (YOLOv5 fine- │    │  (Custom ResNet CNN) │
  │               │    │   tuned)       │    │  → 28 keypoints      │
  └──────┬────────┘    └───────┬────────┘    └──────────┬───────────┘
         │                     │                         │
         │  filter to 2        │  interpolate            │
         │  players            │  missing frames         │
         ▼                     ▼                         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                         MiniCourt                                 │
  │   Pixel → Real-World → Mini-Court Coordinate Transformation       │
  │   (homography via court keypoints + known ITF court dimensions)   │
  └───────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                     Statistics Engine                             │
  │  • Detect ball shot frames (velocity threshold)                   │
  │  • Assign shot to nearest player                                  │
  │  • Compute: shot speed, opponent speed, rolling averages          │
  └───────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │  OUTPUT VIDEO   │
                        │  (annotated     │
                        │   .avi)         │
                        └─────────────────┘
```

---

## 🔄 Pipeline Walkthrough

### 1. Video Ingestion
Raw broadcast video is loaded frame-by-frame into memory using OpenCV via `utils.read_video()`.

### 2. Player Detection & Filtering
`PlayerTracker` runs YOLOv8m on every frame to detect all persons. A court-geometry filter (`choose_and_filter_players`) then isolates exactly the two players on court, discarding ball kids, spectators, and officials by computing proximity to detected court keypoints.

### 3. Ball Detection & Interpolation
`BallTracker` applies a fine-tuned YOLOv5 model specialized for the small, fast-moving tennis ball. Because the ball is frequently occluded or motion-blurred, `interpolate_ball_positions()` uses Pandas forward/backward-fill to bridge missing detections, producing a smooth continuous trajectory.

### 4. Court Keypoint Regression
`CourtLineDetector` feeds the first frame through a custom CNN that regresses the pixel coordinates of **14 standard ITF court keypoints** (service lines, baselines, center marks, etc.). These landmarks anchor the entire coordinate system.

### 5. Mini-Court Coordinate Mapping
`MiniCourt` constructs a homography from the detected keypoints to a canonical top-down court diagram. All player and ball positions are projected into this space, enabling distance measurements in meters using real ITF court dimensions (e.g., `DOUBLE_LINE_WIDTH`).

### 6. Shot Detection & Statistics
Ball shot frames are identified by analyzing velocity changes in the ball trajectory. For each rally segment between consecutive shots:
- **Ball speed** is calculated as `Δdistance / Δtime × 3.6` (km/h)
- The **shooting player** is the one closest to the ball at the shot frame
- The **opponent's movement speed** is computed over the same interval
- Statistics are aggregated per frame and forward-filled into a DataFrame for overlay

### 7. Annotated Output
The final pass draws all overlays sequentially: bounding boxes → ball trail → court keypoints → mini-court diagram → player dots → stats panel → frame ID. The result is saved as an `.avi` file.

---

## 📁 Project Structure

```
TennisVision/
├── main.py                    # Orchestration script — full pipeline entry point
├── yolo_inferenc.py           # Standalone YOLO inference utility / debug script
│
├── trackers/
│   ├── player_tracker.py      # YOLOv8 player detection, filtering, bbox drawing
│   └── ball_tracker.py        # YOLOv5 ball detection, interpolation, shot detection
│
├── court_line_detect/
│   ├── src/
│   │   ├── dataset/
│   │   │   ├── __init__.py        # Dataset package init
│   │   │   ├── dataset.py         # Court keypoint dataset class (transforms, augmentation)
│   │   │   └── load_data.py       # Data loading utilities (splits, paths, samplers)
│   │   │
│   │   ├── model/
│   │   │   ├── __init__.py        # Model package init
│   │   │   └── model.py           # CNN architecture — ResNet backbone + keypoint regression head
│   │   │
│   │   └── utils/
│   │       ├── __init__.py        # Utils package init
│   │       ├── config.py          # Training hyperparameters, paths, constants
│   │       └── visualize.py       # Keypoint overlay helpers for debugging
│   │
│   ├── train_model.py             # Training entry point — loss, optimizer, eval loop
│   ├── training.py                # Core training/validation step logic
│   ├── __init__.py                # Package init — exposes CourtLineDetector
│   └── court_line_detect.py       # Inference wrapper — loads model, runs prediction, draws keypoints
│
├── mini_court/
│   └── mini_court.py          # Homography transform, mini-court rendering
│
├── utils/
│   ├── video_utils.py         # read_video / save_video helpers
│   ├── bbox_utils.py          # measure_distance, center_of_bbox
│   ├── conversions.py         # pixel ↔ meters conversion
│   └── player_stats_utils.py  # draw_player_stats overlay
│
├── constants/
│   └── __init__.py            # ITF court dimensions, color palette, FPS
│
├── models/
│   ├── yolov8m.pt             # Pretrained YOLOv8 medium — player detection
│   ├── yolo5_last.pt          # Fine-tuned YOLOv5 — ball detection
│   └── keypoints_model.pth    # Custom CNN — court keypoint regression
│
├── tracker_stubs/
│   ├── players_detections.pkl # Cached player detection results
│   └── ball_detections.pkl    # Cached ball detection results
│
├── notebooks/
│   └── *.ipynb                # Training notebooks, EDA, model experimentation
│
└── output_videos/
    └── output_video.avi       # Pipeline output
```

---

## 🔬 Technical Deep Dive

### Player Tracker
- **Backbone:** YOLOv8m (pretrained on COCO)
- **Filtering Strategy:** Players are selected based on Euclidean distance from court keypoints. YOLO detects all humans; the two closest to the court baseline region are retained. This is geometry-aware filtering — no additional classifier needed.
- **Stub Pattern:** Detections are serialized with `pickle`. On re-runs, `read_from_stub=True` loads cached `.pkl` files, cutting iteration time from minutes to seconds.

### Ball Tracker
- **Backbone:** YOLOv5, fine-tuned on a tennis ball dataset
- **Why YOLOv5?** The ball is tiny (~10–20px diameter at broadcast resolution) and frequently disappears behind the net, rackets, or due to motion blur. A specialized fine-tuned model dramatically outperforms generic detectors here.
- **Interpolation:** `pd.DataFrame.interpolate()` bridges detection gaps, producing a physically plausible trajectory even through partial occlusions.
- **Shot Frame Detection:** Ball shot frames are identified by modeling velocity: large instantaneous displacement relative to recent trajectory signals a racket impact.

### Court Line Detector
- **Architecture:** Custom CNN (ResNet-style backbone) with a regression head outputting `[x₁, y₁, ..., x₁₄, y₁₄]` — 14 court keypoints flattened to 28 values.
- **Training:** Supervised on court keypoint annotations. The model generalizes to different camera angles and lighting conditions common in broadcast feeds.
- **Usage:** Only the first frame is processed (court is static), making this extremely efficient.

### Mini-Court & Homography
- **Coordinate System:** Real-world ITF standard dimensions are used:
  - `DOUBLE_LINE_WIDTH` = 10.97 m
  - Full court length = 23.77 m
- **Transformation:** A perspective transform (homography) is estimated from the 4+ court keypoints mapping pixel space → canonical court space → mini-court pixel space.
- **Distance Computation:** All speed calculations go through `convert_pixel_distance_to_meters()` which applies the scale factor derived from the known `DOUBLE_LINE_WIDTH` and the detected court width in pixels on the mini-court.

### Statistics Engine
```python
# Core speed formula per rally segment
ball_speed_kmh = (distance_meters / time_seconds) * 3.6

# Player attribution: who is closest to the ball at shot moment?
player_shot_ball = min(
    player_positions.keys(),
    key=lambda pid: measure_distance(player_positions[pid], ball_pos)
)
```
Statistics are tracked in a DataFrame with columns:
`frame_num`, `player_N_number_of_shots`, `player_N_last_shot_speed`,
`player_N_average_shot_speed`, `player_N_last_player_speed`, `player_N_average_player_speed`

Forward-fill (`ffill`) ensures every frame has a valid stat row for the overlay renderer.

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended; CPU inference is very slow for video)
- `ffmpeg` (for video reading/writing support)

### 1. Clone the Repository
```bash
git clone https://github.com/ma74a/TennisVision.git
cd TennisVision
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Core dependencies:**
```
ultralytics>=8.0.0      # YOLOv8
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
pandas>=2.0.0
numpy>=1.24.0
```

### 4. Download Pre-trained Models

Place the following model files in the `models/` directory:

| Model | Path | Description |
|---|---|---|
| YOLOv8m | `models/yolov8m.pt` | Download via `ultralytics` or from [Ultralytics releases](https://github.com/ultralytics/assets/releases) |
| YOLOv5 ball | `models/yolo5_last.pt` | Fine-tuned on tennis ball dataset — see [training notebook](notebooks/) |
| Court keypoints | `models/keypoints_model.pth` | Custom CNN — see [training notebook](notebooks/) |

---

## 🚀 Usage

### Basic Run

```bash
python main.py
```

By default, `main.py` reads from `input_videos/input_video.mp4` and writes to `output_videos/output_video.avi`.

### Modify Input Path
Edit `main.py`:
```python
video_frames = read_video(video_path="path/to/your/tennis_video.mp4")
```

### Disable Stub Caching (Force Re-inference)
```python
players_detections = player_tracker.detect_frames(
    frames=video_frames,
    read_from_stub=False   # ← always re-run detection
)
```

### YOLO Inference Standalone
```bash
python yolo_inferenc.py
```
Use this to quickly test your YOLO setup on a single frame or image.

---

## 📊 Output & Statistics

### Annotated Video Overlays

| Overlay | Color | Description |
|---|---|---|
| Player bounding boxes | Green | YOLOv8 detections filtered to 2 players |
| Ball bounding box | Yellow (cyan on mini-court) | Interpolated ball position |
| Court keypoints | Red dots | 14 predicted ITF court landmarks |
| Mini-court diagram | Bottom-right corner | Top-down court with player/ball positions |
| Stats panel | Top-left | Per-player live statistics table |
| Frame ID | Top-left | Frame counter for debugging |

### Statistics Panel
Each frame displays a running statistics table:

```
╔══════════════════════════════════════════╗
║           Player Statistics              ║
╠══════════════╦═══════════════════════════╣
║              ║   Player 1   │  Player 2  ║
╠══════════════╬══════════════╪════════════╣
║  Shots       ║      12      │     11     ║
║  Last Speed  ║   185 km/h   │  172 km/h  ║
║  Avg Speed   ║   178 km/h   │  168 km/h  ║
║  Move Speed  ║    8.2 km/h  │   7.6 km/h ║
╚══════════════╩══════════════╧════════════╝
```

---

## 🤖 Models

### Model Summary

| Model | Architecture | Input | Output | Use Case |
|---|---|---|---|---|
| `yolov8m.pt` | YOLOv8 Medium | Image frame | Bounding boxes + class labels | Person detection |
| `yolo5_last.pt` | YOLOv5 (fine-tuned) | Image frame | Ball bounding box | Tennis ball detection |
| `keypoints_model.pth` | Custom CNN (ResNet-based) | Image frame | 28 floats (14 keypoints × 2) | Court line detection |

### Training
- Player detection uses the COCO-pretrained `yolov8m.pt` directly — no fine-tuning needed (tennis players are well-represented in COCO).
- Ball detection model was fine-tuned on a tennis-specific dataset due to the ball's small size and unique motion characteristics.
- Court keypoint model was trained on annotated court images covering multiple venues, lighting conditions, and camera angles. See `notebooks/` for the full training pipeline.

---

## ⚙️ Configuration

Key constants are defined in `constants/__init__.py`:

```python
# ITF standard court dimensions
DOUBLE_LINE_WIDTH = 10.97       # meters — used for pixel-to-meter scaling
SINGLE_LINE_WIDTH = 8.23        # meters
COURT_LENGTH = 23.77            # meters

# Video
FPS = 24                        # expected input video frame rate

# Detection
PLAYER_1_ID = 1
PLAYER_2_ID = 2
```

Adjust `FPS` if your input video runs at a different frame rate — ball speed calculations depend on it.

---

## 📈 Results

The system produces accurate real-time analytics on standard broadcast tennis footage:

- **Player tracking:** robust across baseline and mid-court positions; filtering successfully isolates the two players in typical broadcast angles
- **Ball tracking:** interpolation maintains trajectory continuity through typical occlusion events (net crossings, racket contact)
- **Shot speed estimation:** results are in the physically plausible range for professional tennis (160–220 km/h for serves, 80–150 km/h for groundstrokes)
- **Court detection:** keypoint prediction is stable for static camera feeds; performance degrades with rapid pan/zoom

> 📌 Sample output video is available at `output_videos/output_video.avi`.

---

## 🗺️ Roadmap

- [ ] `requirements.txt` with pinned versions
- [ ] CLI argument parsing (input path, output path, model paths)
- [ ] Support for dynamic cameras (moving broadcast feeds)
- [ ] Player identity re-ID across camera cuts
- [ ] Serve/return/rally shot classification
- [ ] Web dashboard for statistics visualization
- [ ] Dockerized inference environment
- [ ] Export statistics to CSV/JSON

---

## 🤝 Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests where applicable
4. Commit: `git commit -m "feat: describe your change"`
5. Push and open a Pull Request

Please open an issue first for major changes to discuss the approach.

---

## 👤 Author

**Mahmoud Etman**
Machine Learning Engineer · Computer Vision

[![GitHub](https://img.shields.io/badge/GitHub-ma74a-181717?logo=github)](https://github.com/ma74a)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-mahm0ud--etman-0A66C2?logo=linkedin)](https://www.linkedin.com/in/mahm0ud-etman/)
[![LeetCode](https://img.shields.io/badge/LeetCode-mahmoud__a21-FFA116?logo=leetcode)](https://leetcode.com/u/mahmoud_a21/)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**If this project was useful, consider giving it a ⭐ on GitHub!**

</div>
