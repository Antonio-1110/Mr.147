"""Ball close-up studio: when a stripe ball counts as "unsure", and how the reader treats that."""

import math
from dataclasses import replace

import numpy as np

from mr147.vision.detector import PixelDetection
from mr147.vision.pool_balls import PoolBallStyle
from mr147.vision.reader import reread
from mr147.vision.studio import UNSURE, look_label, stripe_share

CAM = np.array([0.0, 0.0, 3.0])  # camera straight above the ball
BALL = np.zeros(3)
R = 0.028


def test_stripe_seen_side_on_fills_the_middle_and_end_on_hardly_shows():
    side_on = stripe_share(CAM, BALL, R, pole=np.array([1.0, 0, 0]), stripe_half=0.19)
    end_on = stripe_share(CAM, BALL, R, pole=np.array([0, 0, 1.0]), stripe_half=0.19)
    assert side_on > 0.6
    assert end_on < 0.02


def test_a_tilted_cap_shows_some_stripe():
    tilt = math.radians(45)
    share = stripe_share(CAM, BALL, R, pole=np.array([math.sin(tilt), 0, math.cos(tilt)]), stripe_half=0.19)
    assert 0.1 < share < 0.6


def test_only_dark_black_capped_stripes_seen_from_the_cap_are_unsure():
    style = PoolBallStyle.random(np.random.default_rng(3), ball_set="modern")
    black = replace(style, stripe_caps=(0.03, 0.03, 0.03), colours={**style.colours, 2: (0.04, 0.14, 0.62),
                                                                       1: (0.95, 0.72, 0.04)})
    assert look_label(10, black, 0.02, 0.10) == UNSURE  # dark blue ring on a black cap: looks like the 8
    assert look_label(9, black, 0.02, 0.10) == "yellow-stripe"  # a yellow ring still shows
    assert look_label(10, black, 0.5, 0.10) == "blue-stripe"  # stripe in view
    assert look_label(10, replace(black, stripe_caps=None), 0.02, 0.10) == "blue-stripe"  # white cap
    assert look_label(8, black, None, 0.10) == "black"
    assert look_label(0, black, None, 0.10) == "cue"


class FakeReader:
    def __init__(self, answers):
        self.answers = answers

    def read(self, image, boxes):
        return self.answers[: len(boxes)]


def test_unsure_gives_no_number_and_leaves_a_low_score():
    found = [PixelDetection("8", (0, 0, 10, 10), 0.9)]
    reader = FakeReader([{UNSURE: 0.8, "blue-stripe": 0.15, "black": 0.05}])
    (d,) = reread(found, np.zeros((20, 20, 3), np.uint8), reader, ball_set="modern")
    assert d.name == "10"
    assert d.confidence < 0.2  # low: the robot should look again
    assert all(n != UNSURE for n, _ in d.alternatives)


def test_a_ball_cut_off_by_the_picture_edge_gets_a_low_score():
    from mr147.vision.reader import cut_by_edge

    assert cut_by_edge((100, 80, 140, 100), 200, 100)  # bottom edge, 40 wide but 20 tall
    assert not cut_by_edge((100, 50, 140, 90), 200, 100)  # whole ball
    assert not cut_by_edge((0, 50, 40, 90), 200, 100)  # touches the edge but still round
    found = [PixelDetection("8", (100, 80, 140, 100), 0.9)]
    (d,) = reread(found, np.zeros((100, 200, 3), np.uint8), FakeReader([{"black": 0.95, "blue-stripe": 0.05}]),
                  ball_set="modern")
    assert d.name == "8" and d.confidence < 0.25


def test_a_black_ball_mirrors_the_blue_cloth_round_its_edge(renderer):
    """Real shiny balls reflect the cloth: a black 8 on blue cloth has a blue rim (the 8 vs 10 mix-up)."""
    from panda3d.core import LVecBase4f

    from mr147.vision import table as T
    from mr147.vision.scene import CameraPose, random_look
    from mr147.vision.table import Ball

    spec = T.SPECS["pool"]
    look = random_look(np.random.default_rng(3), spec)
    look.cloth_rgb = (0.15, 0.45, 0.85)
    renderer.build([Ball("8", 0.3, 0.1)], look, spec)
    centre = (0.3, 0.1, spec.ball_r)
    renderer.set_camera(CameraPose((0.3, 1.6, 0.75), centre, 3.0, 0.0))
    shiny, ids = renderer.render_frame()
    renderer.ball_np[0].setShaderInput("env_gloss", LVecBase4f(0, 0, 0, 0))
    plain, _ = renderer.render_frame(auto_exposure=False)
    ball = ids == 0
    blue = lambda img: float(img[ball][:, 2].astype(float).mean() - img[ball][:, 0].astype(float).mean())
    assert ball.sum() > 500
    assert blue(shiny) > blue(plain) + 8
