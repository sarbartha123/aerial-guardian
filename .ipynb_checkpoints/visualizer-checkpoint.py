import cv2
import numpy as np
from typing import List, Dict, Tuple
import colorsys


def id_to_color(track_id: int) -> Tuple[int, int, int]:
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


class Visualizer:
    def __init__(
        self,
        tail_length: int = 30,
        tail_fade: bool = True,
        box_thickness: int = 2,
        font_scale: float = 0.5,
        show_conf: bool = True,
        show_motion_vec: bool = True,
    ):
        self.tail_length = tail_length
        self.tail_fade = tail_fade
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.show_conf = show_conf
        self.show_motion_vec = show_motion_vec

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict],
        fps: float = 0.0,
        motion_vec: Tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:

        out = frame.copy()
        H, W = out.shape[:2]

        for track in tracks:
            tid = track.get("id", -1)
            bbox = track.get("bbox", [0, 0, 0, 0])
            conf = track.get("conf", 0.0)
            history = track.get("history", [])
            color = id_to_color(tid)

            tail_pts = history[-self.tail_length:] if history else []

            if len(tail_pts) > 1:
                for i in range(1, len(tail_pts)):
                    pt1 = (int(tail_pts[i - 1][0]), int(tail_pts[i - 1][1]))
                    pt2 = (int(tail_pts[i][0]), int(tail_pts[i][1]))

                    if self.tail_fade:
                        alpha = i / len(tail_pts)
                        faded_color = tuple(int(c * alpha) for c in color)
                        thickness = max(1, int(2 * alpha))
                    else:
                        faded_color = color
                        thickness = 1

                    cv2.line(out, pt1, pt2, faded_color, thickness, cv2.LINE_AA)

            x1, y1, x2, y2 = (
                int(np.clip(bbox[0], 0, W - 1)),
                int(np.clip(bbox[1], 0, H - 1)),
                int(np.clip(bbox[2], 0, W - 1)),
                int(np.clip(bbox[3], 0, H - 1)),
            )

            cv2.rectangle(out, (x1, y1), (x2, y2), color, self.box_thickness)

            label = f"ID {tid}"
            if self.show_conf:
                label += f" {conf:.2f}"

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )

            label_y = max(y1 - 4, th + 4)

            cv2.rectangle(
                out,
                (x1, label_y - th - 4),
                (x1 + tw + 4, label_y),
                color,
                -1,
            )

            cv2.putText(
                out,
                label,
                (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        self._draw_overlay(out, fps, len(tracks), motion_vec, W, H)

        return out

    def _draw_overlay(self, frame, fps, n_tracks, motion_vec, W, H):
        panel_h = 70
        overlay = frame[:panel_h, :250].copy()

        cv2.rectangle(frame, (0, 0), (250, panel_h), (0, 0, 0), -1)

        cv2.addWeighted(
            overlay, 0.3, frame[:panel_h, :250], 0.7, 0, frame[:panel_h, :250]
        )

        lines = [
            f"FPS: {fps:.1f}",
            f"Tracks: {n_tracks}",
        ]

        if self.show_motion_vec:
            tx, ty = motion_vec
            lines.append(f"Cam dX:{tx:+.1f} dY:{ty:+.1f}")

        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (8, 20 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 200),
                1,
                cv2.LINE_AA,
            )