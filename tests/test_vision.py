"""Checks for the synthetic data generator."""

import math

import numpy as np
import pytest

from mr147.vision import table as T
from mr147.vision.layouts import opening_rack, pool_rack_8ball, pool_rack_9ball, random_layout


def _no_overlap(balls, spec=T.SNOOKER):
    for i, a in enumerate(balls):
        assert a.name in spec.class_id
        assert abs(a.x) <= spec.length / 2 and abs(a.y) <= spec.width / 2
        for b in balls[i + 1:]:
            assert math.hypot(a.x - b.x, a.y - b.y) >= spec.ball_d - 1e-6


def test_opening_rack():
    balls = opening_rack(np.random.default_rng(1))
    names = [b.name for b in balls]
    assert names.count("red") == 15 and names.count("white") == 1 and len(balls) == 22
    _no_overlap(balls)


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("spec", [T.SNOOKER, T.POOL], ids=["snooker", "pool"])
def test_random_layouts_are_legal(seed, spec):
    balls = random_layout(np.random.default_rng(seed), spec)
    _no_overlap(balls, spec)
    if spec.game == "pool":  # every pool ball appears at most once
        assert len({b.name for b in balls}) == len(balls)


def test_pool_racks():
    rng = np.random.default_rng(2)
    eight = pool_rack_8ball(rng)
    assert sorted(b.name for b in eight) == sorted(c for c in T.POOL_CLASSES if c != "pocket")
    _no_overlap(eight, T.POOL)
    nine = pool_rack_9ball(rng)
    assert sorted(b.name for b in nine) == sorted(["cue"] + [str(n) for n in range(1, 10)])
    _no_overlap(nine, T.POOL)
    # The 8 sits in the middle of the third row.
    xs = sorted({round(b.x, 2) for b in eight if b.name != "cue"})
    eight_ball = next(b for b in eight if b.name == "8")
    assert round(eight_ball.x, 2) == xs[2] and abs(eight_ball.y) < 0.002


@pytest.mark.parametrize("spec", [T.SNOOKER, T.POOL], ids=["snooker", "pool"])
def test_labels_match_rendered_balls(renderer, spec):
    """The box worked out from 3D geometry must sit on the ball the renderer drew."""
    from mr147.vision.generate import label_view
    from mr147.vision.scene import CameraPose, random_look

    rng = np.random.default_rng(3)
    balls = random_layout(rng, spec)
    renderer.build(balls, random_look(rng, spec), spec)
    renderer.set_camera(CameraPose((-spec.length / 2 - 0.8, 0.0, 0.8), (0.0, 0.0, 0.0), 65.0))
    _, ids = renderer.render_frame()
    rows, info, pockets = label_view(renderer, balls, ids, spec)
    # From behind the head/baulk cushion, the two far corner pockets and the middles are in view.
    assert sum(p["box"] is not None for p in pockets) >= 3
    assert rows
    checked = 0
    for i, e in enumerate(info):
        if e["visible_frac"] > 0.9 and e["in_frame_frac"] > 0.99:
            ys, xs = np.nonzero(ids == i)
            cx, cy = e["centre_px"]
            r = (e["box"][2] - e["box"][0]) / 2
            assert abs(xs.mean() - cx) < max(1.0, 0.25 * r)
            assert abs(ys.mean() - cy) < max(1.0, 0.25 * r)
            checked += 1
    assert checked >= 2
