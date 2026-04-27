import cv2
import numpy as np
from ultralytics import YOLO
import torch

TARGET_CLASSES = [0]

SAHI_AVAILABLE = False

try:
    import sahi
    from sahi.models.yolov8 import Yolov8DetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except Exception:
    SAHI_AVAILABLE = False


class FastDetector:
    def __init__(
        self,
        model_path="yolo26l.pt",
        conf_thresh=0.15,
        iou_thresh=0.5,
        device="auto",
        imgsz=1536,
        slice_size=384,
        overlap_ratio=0.25,
        **kwargs
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz

        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio

        self.use_sahi = SAHI_AVAILABLE

        if self.use_sahi:
            self.sahi_model = Yolov8DetectionModel(
                model_path=model_path,
                confidence_threshold=self.conf_thresh,
                device=self.device,
                model_type="ultralytics"
            )
            print(f"[Detector] SAHI ON | Device: {self.device}")
        else:
            print(f"[Detector] SAHI OFF → fallback YOLO | Device: {self.device}")

    def detect(self, frame):
        if self.use_sahi:
            result = get_sliced_prediction(
                frame,
                self.sahi_model,
                slice_height=self.slice_size,
                slice_width=self.slice_size,
                overlap_height_ratio=self.overlap_ratio,
                overlap_width_ratio=self.overlap_ratio,
                postprocess_type="NMS",
                postprocess_match_metric="IOU",
                postprocess_match_threshold=0.5
            )

            if len(result.object_prediction_list) == 0:
                return np.empty((0, 6), dtype=np.float32)

            dets = []
            for obj in result.object_prediction_list:
                cls_id = obj.category.id
                if cls_id not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = obj.bbox.to_xyxy()
                score = obj.score.value

                dets.append([x1, y1, x2, y2, score, cls_id])

            if len(dets) == 0:
                return np.empty((0, 6), dtype=np.float32)

            return np.array(dets, dtype=np.float32)

        else:
            results = self.model(
                frame,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                classes=TARGET_CLASSES,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )[0]

            if results.boxes is None or len(results.boxes) == 0:
                return np.empty((0, 6), dtype=np.float32)

            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()

            return np.concatenate([boxes, confs[:, None], clss[:, None]], axis=1).astype(np.float32)


SlicedDetector = FastDetector