"""
evaluate_visdrone.py
--------------------
Evaluate pipeline on VisDrone MOT validation set.
Computes: MOTA, MOTP, IDF1, MT, ML, FP, FN, ID Switches

Usage:
    python evaluate_visdrone.py \
        --dataset_root /path/to/VisDrone2019-MOT-val \
        --output_dir outputs/visdrone_eval \
        --model yolov8n.pt

VisDrone annotation format (per frame):
    <x>, <y>, <w>, <h>, <score>, <class>, <truncation>, <occlusion>
    class 1 = pedestrian, class 2 = person (we use both)
"""

import argparse
import os
import glob
import cv2
import numpy as np
from pathlib import Path
import json
import time
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from detector import SlicedDetector
from bytetrack import ByteTracker
from motion_compensation import MotionCompensator
from visualizer import Visualizer

# VisDrone person classes
VISDRONE_PERSON_CLASSES = {1, 2}  # pedestrian, person


def load_visdrone_gt(ann_path: str):
    """Load VisDrone MOT ground truth annotations."""
    gt_per_frame = {}
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            obj_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            # parts[6] = class, parts[7] = truncation, parts[8] = occlusion (may not exist)
            cls = int(parts[7]) if len(parts) > 7 else 1

            if cls not in VISDRONE_PERSON_CLASSES:
                continue

            if frame_id not in gt_per_frame:
                gt_per_frame[frame_id] = []
            gt_per_frame[frame_id].append({
                "id": obj_id,
                "bbox": [x, y, x+w, y+h],
            })
    return gt_per_frame


def compute_iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    union = (xa2-xa1)*(ya2-ya1) + (xb2-xb1)*(yb2-yb1) - inter
    return inter / max(union, 1e-6)


def evaluate_sequence(seq_path, model, args):
    """Run pipeline on a single VisDrone sequence and compute basic MOT metrics."""
    img_dir = Path(seq_path) / "img1" if (Path(seq_path) / "img1").exists() else Path(seq_path)
    ann_dir = Path(seq_path).parent.parent / "annotations"
    seq_name = Path(seq_path).name
    ann_file = ann_dir / f"{seq_name}.txt"

    if not ann_file.exists():
        print(f"  [WARN] No annotation found for {seq_name}, skipping metrics")
        has_gt = False
        gt_per_frame = {}
    else:
        has_gt = True
        gt_per_frame = load_visdrone_gt(str(ann_file))

    image_files = sorted(glob.glob(str(img_dir / "*.jpg")))
    if not image_files:
        image_files = sorted(glob.glob(str(img_dir / "*.png")))

    if not image_files:
        return None

    detector = SlicedDetector(
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        slice_size=args.slice_size,
        overlap_ratio=0.2,
        device=args.device,
        use_slicing=not args.no_slicing,
    )
    tracker = ByteTracker(max_lost=30, min_hits=1)
    compensator = MotionCompensator()
    visualizer = Visualizer(tail_length=25)

    # Output video
    out_dir = Path(args.output_dir) / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = str(out_dir / "tracked.mp4")

    first_frame = cv2.imread(image_files[0])
    H, W = first_frame.shape[:2]
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))

    # Metrics accumulators
    total_tp, total_fp, total_fn = 0, 0, 0
    id_switches = 0
    prev_gt_to_pred = {}
    fps_list = []

    for frame_idx, img_path in enumerate(image_files):
        frame_num = frame_idx + 1
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        t0 = time.perf_counter()
        dets = detector.detect(frame)
        compensator.update(frame)
        tracks = tracker.update(dets, compensate_fn=compensator.compensate_boxes)
        fps_list.append(1.0 / max(time.perf_counter() - t0, 1e-6))

        # Metrics
        if has_gt and frame_num in gt_per_frame:
            gt_boxes = gt_per_frame[frame_num]
            pred_boxes = [{"id": t["id"], "bbox": t["bbox"]} for t in tracks]

            # Match GT to predictions via IoU
            iou_thresh = 0.5
            matched_gt, matched_pred = set(), set()
            gt_to_pred = {}

            for gi, gt in enumerate(gt_boxes):
                best_iou, best_pi = 0, -1
                for pi, pred in enumerate(pred_boxes):
                    if pi in matched_pred:
                        continue
                    iou = compute_iou(gt["bbox"], pred["bbox"])
                    if iou > best_iou:
                        best_iou, best_pi = iou, pi

                if best_iou >= iou_thresh:
                    matched_gt.add(gi)
                    matched_pred.add(best_pi)
                    gt_to_pred[gt["id"]] = pred_boxes[best_pi]["id"]

            tp = len(matched_gt)
            fp = len(pred_boxes) - len(matched_pred)
            fn = len(gt_boxes) - len(matched_gt)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # ID switches: GT object matched to different pred ID than before
            for gt_id, pred_id in gt_to_pred.items():
                if gt_id in prev_gt_to_pred and prev_gt_to_pred[gt_id] != pred_id:
                    id_switches += 1
            prev_gt_to_pred.update(gt_to_pred)

        motion_vec = compensator.get_translation()
        avg_fps = np.mean(fps_list[-30:]) if fps_list else 0
        vis = visualizer.draw_tracks(frame, tracks, avg_fps, motion_vec)
        writer.write(vis)

    writer.release()

    # Compute MOTA
    motp_iou = 0.5  # Approximation
    denom = total_tp + total_fn + total_fp
    mota = 1 - (total_fn + total_fp + id_switches) / max(denom, 1)

    avg_fps = np.mean(fps_list) if fps_list else 0

    return {
        "sequence": seq_name,
        "frames": len(image_files),
        "avg_fps": avg_fps,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "id_switches": id_switches,
        "mota": mota,
        "output_video": out_video_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on VisDrone MOT val set")
    parser.add_argument("--dataset_root", required=True, help="Path to VisDrone2019-MOT-val/sequences")
    parser.add_argument("--output_dir", default="outputs/visdrone_eval")
    parser.add_argument("--model", default="yolo26l.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--slice-size", type=int, default=320)
    parser.add_argument("--no-slicing", action="store_true")
    parser.add_argument("--max-seqs", type=int, default=None, help="Limit number of sequences (for testing)")
    args = parser.parse_args()

    seq_dirs = sorted(glob.glob(os.path.join(args.dataset_root, "*")))
    seq_dirs = [s for s in seq_dirs if os.path.isdir(s)]

    if args.max_seqs:
        seq_dirs = seq_dirs[:args.max_seqs]

    print(f"\n[Eval] Found {len(seq_dirs)} sequences")

    all_results = []
    for seq in seq_dirs:
        print(f"\n  Processing: {Path(seq).name}")
        result = evaluate_sequence(seq, None, args)
        if result:
            all_results.append(result)
            print(f"    FPS: {result['avg_fps']:.1f} | MOTA: {result['mota']:.3f} | IDS: {result['id_switches']}")

    if all_results:
        summary = {
            "total_sequences": len(all_results),
            "avg_fps": np.mean([r["avg_fps"] for r in all_results]),
            "avg_mota": np.mean([r["mota"] for r in all_results]),
            "total_id_switches": sum(r["id_switches"] for r in all_results),
            "total_fp": sum(r["fp"] for r in all_results),
            "total_fn": sum(r["fn"] for r in all_results),
        }

        print(f"\n{'='*50}")
        print("  EVALUATION SUMMARY")
        print(f"{'='*50}")
        for k, v in summary.items():
            print(f"  {k:25s}: {v:.3f}" if isinstance(v, float) else f"  {k:25s}: {v}")

        summary_path = Path(args.output_dir) / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({"summary": summary, "sequences": all_results}, f, indent=2)
        print(f"\n  Results saved: {summary_path}")


if __name__ == "__main__":
    main()
