# Aerial Guardian: Drone-Based Person Detection & Tracking

## Overview

This project implements a lightweight and robust pipeline for detecting and tracking persons in aerial drone footage using the VisDrone 2019 MOT dataset.

The system is designed to handle:
- Small object detection (high altitude)
- Camera motion (ego-motion from drone)
- Occlusion and re-identification
- ID consistency across frames

---

## Features

- YOLO-based person detection
- SAHI (Sliced Inference) for small object detection
- Custom ByteTrack-based tracker
- Homography-based camera motion compensation
- Appearance + Motion + IoU matching
- ID stabilization with memory and recovery
- Lightweight (<300 MB)

---

## Dataset

- VisDrone 2019 MOT Validation Set  
- Target Class: Person

---

## Detection Strategy

- High-resolution inference (1280–1536)
- SAHI slicing for small objects
- Low confidence threshold for recall
- Class filtering (person only)

---

## Tracking Strategy

- Hybrid matching:
  - IoU
  - Appearance similarity (color histogram)
  - Motion consistency

- Enhancements:
  - Homography-based camera motion compensation
  - Track memory for re-identification
  - Velocity prediction
  - Track locking for stable IDs

---

## Summary Report

### Choice of Architecture & Small Object Detection

The detection module is based on the Ultralytics YOLO architecture due to its strong balance between speed and accuracy.

To address small object detection:

- Sliced inference (SAHI) divides frames into overlapping patches
- High input resolution preserves small object details
- Lower confidence threshold improves recall
- Person-only filtering reduces noise

---

### Handling ID Switching

Drone motion and occlusion cause ID instability. The system addresses this using:

- Homography-based motion compensation
- Hybrid matching (IoU + appearance + motion)
- Track memory for re-identification
- Velocity-based prediction
- Track locking for stable identities

These techniques reduce ID switches and improve tracking consistency.

---

### Edge Deployment Strategy

To run on edge devices such as NVIDIA Jetson:

- Use smaller YOLO models (nano/small)
- Convert model to TensorRT
- Enable FP16 inference
- Reduce slice size
- Use lightweight feature extractor
- Limit tracking memory

---

### Engineering Trade-offs

- Higher resolution and slicing improve accuracy but reduce FPS
- Memory tracking improves ID stability but increases computation
- Motion compensation improves robustness but adds overhead

The system balances accuracy and efficiency for real-world drone deployment.

---

### Handling Camera Noise

- Homography separates camera motion from object motion
- Motion smoothing reduces jitter
- Distance gating prevents wrong matches
- Temporal memory maintains identity consistency

---

## Performance

| Metric        | Value |
|--------------|------|
| Device       | CPU |
| FPS          | ~1.7 |
| Model Size   | <300 MB |
| Unique IDs   | ~40–80 |

---

## Run Instructions

python pipeline.py --input <path_to_dataset> --output outputs/result.mp4

---

## Arguments

| Argument | Description |
|--------|------------|
| --input | Input video or image folder |
| --output | Output video path |
| --model | YOLO model path |
| --conf | Confidence threshold |
| --iou | IoU threshold |
| --match-thresh | Tracking match threshold |
| --max-lost | Max frames to keep lost tracks |
| --min-hits | Minimum hits to confirm track |

---

## Output

- Bounding boxes
- Unique IDs
- Trajectory tails

---

## Project Structure

.
├── pipeline.py
├── detector.py
├── bytetrack.py
├── motion_compensation.py
├── visualizer.py
├── outputs/
└── README.md




## Model 

The  model weights are hosted externally due to GitHub file size limitations.

Download link:
https://drive.google.com/file/d/1hy1_pDYB-9779ugDeNAwtHB91_F75kva/view?usp=sharing

After downloading, place the file in the root directory:

Drone-Project/
├── yolo26l.pt

Then run the pipeline normally.

---

## Author

Sarbartha  sankar Mallick
