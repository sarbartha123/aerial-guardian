import cv2
import numpy as np
from typing import Optional, Tuple


class MotionCompensator:
    
    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: int = 10,
        use_affine: bool = True,  # False = full homography (more DoF, less stable)
    ):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.use_affine = use_affine

        # LK optical flow params
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self.prev_gray: Optional[np.ndarray] = None
        self.transform: Optional[np.ndarray] = None  # Last estimated transform

    def update(self, frame: np.ndarray) -> Optional[np.ndarray]:
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        # Detect trackable feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=7,
        )

        if prev_pts is None or len(prev_pts) < 4:
            self.prev_gray = gray
            return None

        # Track those points into the current frame
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )

        # Keep only successfully tracked points
        status = status.flatten()
        prev_good = prev_pts[status == 1]
        curr_good = curr_pts[status == 1]

        if len(prev_good) < 4:
            self.prev_gray = gray
            return None

        # Estimate transform: how did the camera move?
        if self.use_affine:
            # Partial affine (translation + rotation + scale): more stable
            M, inliers = cv2.estimateAffinePartial2D(
                prev_good, curr_good,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )
        else:
            M, inliers = cv2.findHomography(
                prev_good, curr_good,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )
            if M is not None:
                M = M[:2, :]  # Use only affine part for box compensation

        self.prev_gray = gray
        self.transform = M
        return M

    def compensate_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
       
        if self.transform is None or len(boxes_xyxy) == 0:
            return boxes_xyxy

        M = self.transform
        if M is None:
            return boxes_xyxy

        # Invert the transform (we want to remove camera motion)
        try:
            M_inv = cv2.invertAffineTransform(M)
        except cv2.error:
            return boxes_xyxy

        compensated = boxes_xyxy.copy().astype(np.float32)
        for i, box in enumerate(compensated):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            # Transform center point
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            new_pt = cv2.transform(pt, M_inv)[0][0]
            ncx, ncy = new_pt

            compensated[i] = [ncx - w/2, ncy - h/2, ncx + w/2, ncy + h/2]

        return compensated

    def get_translation(self) -> Tuple[float, float]:
        
        if self.transform is None:
            return 0.0, 0.0
        return float(self.transform[0, 2]), float(self.transform[1, 2])
