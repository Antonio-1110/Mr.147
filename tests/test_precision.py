"""Precision helpers: tightening ball boxes to the edge, and combining several looks."""

import math

import cv2
import numpy as np

from mr147.vision import table as T
from mr147.vision.locate import fuse_looks, sight_weight
from mr147.vision.refine import refine_ball_box


def _ball_picture(cx, cy, r, w=200, h=150, scale=8):
    """A smooth ball on cloth, drawn large and shrunk so its edge is anti-aliased like a photo."""
    big = np.zeros((h * scale, w * scale, 3), np.uint8)
    big[:] = (40, 110, 200)  # blue cloth
    centre = (int(round(cx * scale)), int(round(cy * scale)))
    cv2.circle(big, centre, int(round(r * scale)), (230, 190, 30), -1, cv2.LINE_AA)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)


def test_refine_finds_the_real_edge_from_a_sloppy_box():
    cx, cy, r = 101.3, 74.6, 14.0
    img = _ball_picture(cx, cy, r)
    true = np.array([cx - r, cy - r, cx + r, cy + r])
    sloppy = true + [2.0, -1.5, 3.0, 1.0]
    got, ok = refine_ball_box(img, sloppy)
    assert ok
    # Left/right/top edges are measured; the bottom (where the ball's shadow is) comes from the fit.
    assert np.abs(got - true).max() < 0.35, got - true


def test_refine_keeps_the_box_when_there_is_no_ball():
    img = np.full((150, 200, 3), (40, 110, 200), np.uint8)
    img = (img + np.random.default_rng(0).normal(0, 3, img.shape)).clip(0, 255).astype(np.uint8)
    box = np.array([80.0, 60.0, 108.0, 88.0])
    got, ok = refine_ball_box(img, box)
    assert not ok and np.array_equal(got, box)


def test_sight_weight_trusts_across_the_line_of_sight_more():
    W = sight_weight((1.0, 0.0), (-1.0, 0.0, 0.8), T.POOL.ball_r)  # looking along +x
    assert W[1, 1] > 5 * W[0, 0]


def test_two_looks_from_different_sides_beat_their_average():
    true = np.array([0.3, 0.2])
    # Each look is mostly wrong along its own line of sight (the usual case from head height).
    cam_a, cam_b = np.array([-1.6, 0.2, 0.8]), np.array([0.3, -1.3, 0.8])
    a = true + 0.012 * (true - cam_a[:2]) / np.linalg.norm(true - cam_a[:2]) + [0, 0.001]
    b = true - 0.010 * (true - cam_b[:2]) / np.linalg.norm(true - cam_b[:2]) + [0.001, 0]
    fused = fuse_looks([(tuple(a), cam_a), (tuple(b), cam_b)], T.POOL.ball_r)
    average = (a + b) / 2
    assert math.dist(fused, true) < 0.5 * math.dist(average, true)


def test_refine_helps_rendered_balls_with_sloppy_boxes(renderer):
    from mr147.vision.generate import label_view
    from mr147.vision.layouts import random_layout
    from mr147.vision.scene import random_camera, random_look

    rng = np.random.default_rng(5)
    spec = T.POOL
    before, after = [], []
    for _ in range(3):
        balls = random_layout(rng, spec)
        renderer.build(balls, random_look(rng, spec), spec)
        renderer.set_camera(random_camera(rng, spec))
        rgb, ids = renderer.render_frame()
        _, info, _ = label_view(renderer, balls, ids, spec)
        for e in info:
            if e["box"] is None or e["visible_frac"] < 0.95 or e["box"][2] - e["box"][0] < 12:
                continue
            true = np.array(e["box"])
            sloppy = true + rng.normal(0, 1.5, 4)
            got, _ = refine_ball_box(rgb, sloppy)
            before.append(np.abs(sloppy - true)[:3].mean())  # left, top, right edges
            after.append(np.abs(got - true)[:3].mean())
    assert len(before) >= 5
    assert np.median(after) < np.median(before)
