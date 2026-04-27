import argparse
import time
import os
import glob
import cv2
import numpy as np
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from detector import SlicedDetector
from bytetrack import ByteTrackPlus
from motion_compensation import MotionCompensator
from visualizer import Visualizer


def get_frame_source(input_path: str):
    if input_path.isdigit():
        return cv2.VideoCapture(int(input_path)), None, None

    p = Path(input_path)
    if p.is_dir():
        exts = ["*.jpg", "*.jpeg", "*.png"]
        files = []
        for ext in exts:
            files.extend(sorted(glob.glob(str(p / ext))))
        return None, files, None

    if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        return cap, None, fps

    raise ValueError(f"Unsupported input: {input_path}")


def simple_feature_extractor(frame, detections):
    feats = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            feats.append(None)
            continue

        crop = cv2.resize(crop, (32, 64))
        hist = cv2.calcHist([crop], [0,1,2], None, [8,8,8], [0,256]*3)
        hist = cv2.normalize(hist, hist).flatten()
        feats.append(hist)

    return feats


def run_pipeline(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    detector = SlicedDetector(
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        device=device,
        use_slicing=not args.no_slicing,
    )

    tracker = ByteTrackPlus(
        high_thresh=args.conf,
        match_thresh=args.match_thresh,
        max_lost=args.max_lost,
        min_hits=args.min_hits
    )

    compensator = MotionCompensator(use_affine=True)

    visualizer = Visualizer(
        tail_length=args.tail_length,
        show_conf=True,
        show_motion_vec=True,
    )

    cap, image_files, src_fps = get_frame_source(args.input)

    if cap is not None:
        ret, test_frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read video")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        H, W = test_frame.shape[:2]
        src_fps = src_fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        test_frame = cv2.imread(image_files[0])
        H, W = test_frame.shape[:2]
        src_fps = 30.0

    os.makedirs(Path(args.output).parent, exist_ok=True)
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        min(src_fps, 30.0),
        (W, H)
    )

    print(f"\n[Pipeline] Device: {device}")
    print(f"[Pipeline] Starting: {args.input}\n")

    frame_idx = 0
    fps_hist = []
    total_det = 0
    unique_ids = set()

    def generator():
        if cap is not None:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        else:
            for f in image_files:
                frame = cv2.imread(f)
                if frame is not None:
                    yield frame

    for frame in generator():

        t0 = time.perf_counter()

        detections = detector.detect(frame)
        total_det += len(detections)

        compensator.update(frame)

        features = simple_feature_extractor(frame, detections)

        tracks = tracker.update(detections, features, frame)

        for t in tracks:
            unique_ids.add(t["id"])

        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        fps_hist.append(fps)
        avg_fps = np.mean(fps_hist[-30:])

        vis = visualizer.draw_tracks(frame, tracks, avg_fps, compensator.get_translation())

        writer.write(vis)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx} | FPS {avg_fps:.1f} | Dets {len(detections)} | Tracks {len(tracks)}")

    if cap is not None:
        cap.release()
    writer.release()

    print("\n========== RESULT ==========")
    print(f"Frames: {frame_idx}")
    print(f"Avg FPS: {np.mean(fps_hist):.2f}")
    print(f"Total detections: {total_det}")
    print(f"Unique IDs: {len(unique_ids)}")
    print(f"Saved: {args.output}")
    print("===========================\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/result.mp4")

    parser.add_argument("--model", default="yolo26l.pt")

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)

    parser.add_argument("--match-thresh", type=float, default=0.4)
    parser.add_argument("--max-lost", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=2)

    parser.add_argument("--tail-length", type=int, default=30)
    parser.add_argument("--no-slicing", action="store_true")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()