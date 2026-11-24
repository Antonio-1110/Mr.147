"""A camera for simulation: renders the simulated table from the robot's head.

Used to test the whole vision chain without hardware:

    physics table  ->  SimCamera renders a picture  ->  detector  ->  CameraVision
                   ->  table positions  ->  table memory

SimCamera reports a slightly wrong camera pose on purpose (like a real robot
body would), so the pocket-based correction gets exercised too.

PerfectDetector reads the boxes straight from the renderer instead of running
YOLO. It lets us test the geometry before a model is trained; swap in
YoloDetector to test the real thing.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from mr147.table_memory.table_state import TableState

from .augment import camera_effects
from .detector import PixelDetection
from .generate import label_view
from .scene import CameraPose, Look, Renderer, random_look
from .table import SNOOKER, Ball, TableSpec


def balls_from_table(table: TableState, game: str = "snooker") -> list[Ball]:
    """Our ball ids ("red_03", "black", "white"; pool "9", "white") -> balls the renderer can draw."""
    out = []
    for ball_id, (x, y) in table.positions.items():
        if game == "pool":
            out.append(Ball("cue" if ball_id == "white" else ball_id, x, y))
        elif ball_id.startswith("red"):
            out.append(Ball("red", x, y, int(ball_id.split("_")[1]) - 1))
        else:
            out.append(Ball(ball_id, x, y))
    return out


class SimCamera:
    def __init__(self, renderer: Renderer, true_table: Callable[[], TableState], start: CameraPose,
                 spec: TableSpec = SNOOKER, look: Look | None = None, pose_error_m: float = 0.02,
                 roll_error_deg: float = 0.5, camera_effects_on: bool = True, seed: int = 0):
        self.renderer = renderer
        self.true_table = true_table
        self.spec = spec
        self.pose = start
        self.rng = np.random.default_rng(seed)
        self.look = look or random_look(self.rng, spec)
        self.pose_error_m = pose_error_m
        self.roll_error_deg = roll_error_deg
        self.camera_effects_on = camera_effects_on
        self.last_labels: tuple[list, list] = ([], [])  # (balls, pockets) from the renderer

    @property
    def size(self) -> tuple[int, int]:
        return self.renderer.w, self.renderer.h

    def move_to(self, pose: CameraPose) -> None:
        self.pose = pose

    def capture(self) -> tuple[np.ndarray, CameraPose]:
        balls = balls_from_table(self.true_table(), self.spec.game)
        self.renderer.build(balls, self.look, self.spec)
        self.renderer.set_camera(self.pose)
        rgb, ids = self.renderer.render_frame()
        _, info, pockets = label_view(self.renderer, balls, ids, self.spec)
        self.last_labels = (info, pockets)
        if self.camera_effects_on:
            rgb = camera_effects(rgb, self.rng)
        e = self.pose_error_m
        rough = CameraPose(
            tuple(np.array(self.pose.pos) + self.rng.normal(0, e, 3)),
            tuple(np.array(self.pose.look_at) + self.rng.normal(0, e, 3)),
            self.pose.fov_h,
            self.pose.roll + self.rng.normal(0, self.roll_error_deg),
        )
        return rgb, rough


class PerfectDetector:
    """Boxes straight from the simulator, with a little pixel noise like a good detector."""

    def __init__(self, camera: SimCamera, box_noise_px: float = 0.5, seed: int = 0):
        self.camera = camera
        self.box_noise_px = box_noise_px
        self.rng = np.random.default_rng(seed)

    def detect(self, image: np.ndarray) -> list[PixelDetection]:
        info, pockets = self.camera.last_labels
        h, w = image.shape[:2]
        out = []
        for e in info + pockets:
            if e["box"] is None:
                continue
            box = np.array(e["box"]) + self.rng.normal(0, self.box_noise_px, 4)
            box = np.clip(box, 0, [w, h, w, h])
            out.append(PixelDetection(e["class"], tuple(float(v) for v in box), 0.95))
        return out
