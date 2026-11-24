"""The ball detector: a picture in, boxes out.

    detector = YoloDetector("models/snooker.pt")
    boxes = detector.detect(rgb_image)

Every detector (the trained YOLO model, or the "perfect" one used in
simulation tests) returns the same thing: a list of PixelDetection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PixelDetection:
    name: str  # label name: "red", "pink", "pocket", "9", "cue", ...
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
    confidence: float  # 0..1
    # Every name the detector considered for this box, with its score, best first
    # (empty = only `name`). Used to pick names for all balls together (identify.py).
    alternatives: tuple[tuple[str, float], ...] = ()

    @property
    def scores(self) -> dict[str, float]:
        return dict(self.alternatives) if self.alternatives else {self.name: self.confidence}

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[PixelDetection]:
        """Find balls and pockets in an RGB image (H x W x 3, uint8)."""
        ...


class YoloDetector:
    """A trained YOLO model (see vision/train.py). Needs the `ultralytics` package."""

    def __init__(self, weights: str | Path, min_confidence: float = 0.25, image_size: int = 1280,
                 device: str | None = None, reader: str | Path | None = None, ball_set: str = "traditional"):
        from ultralytics import YOLO  # only needed when a model is actually used

        from .reader import BallReader

        self.model = YOLO(str(weights))
        # Optional close-up reader: names each ball from a full-size crop (reader.py).
        self.reader = BallReader(reader, device) if reader else None
        self.ball_set = ball_set  # pool: "traditional" or "modern" colours (pool_balls.BALL_SETS)
        self.names = self.model.names
        self.min_confidence = min_confidence
        self.image_size = image_size
        self.device = device

    def detect(self, image: np.ndarray) -> list[PixelDetection]:
        bgr = np.ascontiguousarray(image[..., ::-1])  # ultralytics expects OpenCV colour order
        # Ask for every name the model considers (down to a low score), not just the
        # best one, then gather the boxes that sit on the same ball.
        result = self.model.predict(bgr, conf=min(self.min_confidence, CANDIDATE_SCORE),
                                    imgsz=self.image_size, device=self.device, verbose=False)[0]
        raw = [PixelDetection(self.names[int(cls)], tuple(box), float(conf))
               for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                         result.boxes.conf.tolist())]
        found = on_cloth([d for d in group_boxes(raw) if d.confidence >= self.min_confidence], image)
        if self.reader is not None:
            from .reader import reread

            found = reread(found, image, self.reader, ball_set=self.ball_set)
        return found


CANDIDATE_SCORE = 0.05  # names scored lower than this aren't worth keeping as alternatives


def on_cloth(found: list[PixelDetection], image: np.ndarray, margin: float = 0.3) -> list[PixelDetection]:
    """Drop "balls" that are not on this table: people, chalk on the rails, other tables.

    1. The cloth colour is the most common colour a little way round the balls
       (most of them are on the table).
    2. The table is the biggest patch of that colour in the picture (cloth plus the
       cloth-covered cushions); its outline is that patch's convex hull. This works
       when the near pockets are out of view, and other tables are separate patches.
    3. A ball is kept when the point where it touches the table is inside the outline
       (allowing `margin` ball widths). Pockets are always kept.
    """
    import cv2

    balls = [d for d in found if d.name != "pocket"]
    if len(balls) < 3:
        return found  # too few to tell what the cloth looks like
    h, w = image.shape[:2]
    scale = 640 / w  # the outline doesn't need full resolution
    small = cv2.resize(np.ascontiguousarray(image), (640, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    sh, sw = lab.shape[:2]

    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    samples = []
    for d in balls:
        size = max(d.box[2] - d.box[0], d.box[3] - d.box[1], 2.0) * scale
        for r in (1.5, 2.5):
            xs = (d.centre[0] * scale + np.cos(angles) * size * r).astype(int)
            ys = (d.centre[1] * scale + np.sin(angles) * size * r).astype(int)
            ok = (xs >= 0) & (xs < sw) & (ys >= 0) & (ys < sh)
            samples.append(lab[ys[ok], xs[ok]])
    pts = np.vstack(samples)
    if len(pts) < 20:
        return found
    hist, ea, eb = np.histogram2d(pts[:, 1], pts[:, 2], bins=32, range=[[0, 256], [0, 256]])
    ia, ib = np.unravel_index(np.argmax(hist), hist.shape)
    cloth_a, cloth_b = (ea[ia] + ea[ia + 1]) / 2, (eb[ib] + eb[ib + 1]) / 2

    mask = (np.hypot(lab[..., 1] - cloth_a, lab[..., 2] - cloth_b) < 14).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))  # bridge a cue lying across
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return found
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < 0.03 * sh * sw:
        return found  # no clear table in view: keep everything
    ys, xs = np.nonzero(labels == biggest)
    outline = cv2.convexHull(np.stack([xs, ys], 1).astype(np.float32)) / scale

    def on_table(d: PixelDetection) -> bool:
        size = max(d.box[2] - d.box[0], 1.0)
        foot = (float(d.centre[0]), float(d.box[3]))  # where the ball touches the cloth
        return cv2.pointPolygonTest(outline, foot, True) / size > -margin

    return [d for d in found if d.name == "pocket" or on_table(d)]


def iou(a, b) -> float:
    """Overlap of two boxes: shared area / combined area (1 = identical)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def group_boxes(found: list[PixelDetection], same_ball_iou: float = 0.6) -> list[PixelDetection]:
    """Boxes with different names on the same ball -> one box with alternatives.

    The model may say "8 (0.6)" and "4 (0.3)" for the same ball. We keep the most
    confident box and remember the other names as alternatives. Pockets are never
    grouped with balls.
    """
    groups: list[tuple[PixelDetection, dict[str, float]]] = []
    for d in sorted(found, key=lambda d: -d.confidence):
        for lead, scores in groups:
            if (lead.name == "pocket") == (d.name == "pocket") and iou(lead.box, d.box) >= same_ball_iou:
                scores.setdefault(d.name, d.confidence)
                break
        else:
            groups.append((d, {d.name: d.confidence}))
    return [PixelDetection(lead.name, lead.box, lead.confidence,
                           tuple(sorted(scores.items(), key=lambda kv: -kv[1])))
            for lead, scores in groups]
