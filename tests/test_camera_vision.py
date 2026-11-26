"""The vision chain end to end, in simulation: render -> boxes -> table positions -> memory."""

import math
import random

import numpy as np
import pytest

from mr147.interfaces.vision import ViewRequest
from mr147.simulated_hardware import open_table
from mr147.snooker.balls import Colour
from mr147.table_memory import TableMemory
from mr147.vision import table as T
from mr147.vision.camera_vision import CameraVision, choose_view, merge_frames
from mr147.vision.locate import PinholeCamera, ball_position, refine_pose
from mr147.vision.scene import CameraPose
from mr147.vision.sim_camera import PerfectDetector, SimCamera


def test_ball_position_inverts_projection():
    cam = PinholeCamera.looking_at((-2.4, -1.2, 0.8), (0, 0, 0), 65, 1280, 720)
    for x, y in [(0.0, 0.0), (-1.0, 0.5), (1.5, -0.7), (-1.6, -0.8)]:
        box = cam.ball_box(x, y, T.BALL_R)
        got = ball_position(cam, box, T.BALL_R)
        assert math.dist(got, (x, y)) < 1e-4


def test_pockets_fix_a_rough_camera_pose():
    true = PinholeCamera.looking_at((-2.4, -1.2, 0.8), (0, 0, 0), 65, 1280, 720)
    rough = PinholeCamera.looking_at((-2.37, -1.23, 0.82), (0.03, -0.02, 0), 65, 1280, 720, roll_deg=0.7)
    boxes = [b for px, py, pr in T.POCKETS if (b := true.pocket_box(px, py, pr, T.SNOOKER)) is not None
             and b[0] > 0 and b[1] > 0 and b[2] < 1280 and b[3] < 720]
    fit = refine_pose(rough, boxes)
    assert fit.pockets_used >= 3
    assert np.linalg.norm(fit.camera.position - true.position) < 0.002


def test_choose_view_avoids_blocked_lines_of_sight():
    target = (0.5, 0.0)
    blockers = [(0.5 + d * 0.06, 0.0) for d in (-1, 1)] + [(0.5, d * 0.06) for d in (-1, 1)]
    pose = choose_view([target], blockers + [target])
    eye = np.array(pose.pos)
    # Whatever it picked, the straight line to the target must miss the blockers.
    from mr147.vision.camera_vision import _line_blocked

    balls = [np.array([x, y, T.BALL_R]) for x, y in blockers]
    assert not _line_blocked(eye, np.array([*target, T.BALL_R]), balls, T.BALL_R)


@pytest.fixture
def table_and_vision(renderer):
    table = open_table(random.Random(4), reds=10)
    start = CameraPose((-2.2, -1.35, 0.75), (0.0, 0.0, 0.0), 65.0)
    cam = SimCamera(renderer, lambda: table, start, seed=1)
    return table, CameraVision(cam, PerfectDetector(cam, seed=1))


def test_rendered_vision_places_balls_on_the_table(table_and_vision):
    table, vision = table_and_vision
    frame = vision.observe()
    assert vision.last_look.pose_fit.pockets_used >= 3
    assert len(frame.detections) >= 8
    errors = []
    for d in frame.detections:
        # Nearest true ball of the same colour.
        candidates = [p for i, p in table.positions.items() if i.split("_")[0] == d.colour.value]
        errors.append(min(math.dist(d.position, p) for p in candidates))
    # Small 640-pixel test pictures with half a pixel of box noise: about 1 cm. Most of
    # it comes from the camera pose (only ~4 pockets to pin it down), not the balls.
    assert np.median(errors) < 0.015
    assert max(errors) < 0.04


def test_better_view_merges_and_memory_accepts(table_and_vision):
    table, vision = table_and_vision
    vision.observe()
    far_corner = (1.5, 0.7)
    frame = vision.request_better_view(ViewRequest(("check the far corner",), (far_corner,)))
    seen = {d.colour for d in frame.detections}
    assert Colour.WHITE in seen and Colour.RED in seen
    memory = TableMemory()
    check = memory.first_look(frame, trust_camera=True)
    assert check.accepted and check.table is not None


def test_merge_keeps_balls_the_new_camera_cannot_see():
    from mr147.interfaces.vision import Detection, DetectionFrame

    old_cam = PinholeCamera.looking_at((-2.4, 0, 0.8), (0, 0, 0), 65, 1280, 720)
    new_cam = PinholeCamera.looking_at((2.4, 0, 0.8), (1.5, 0, 0), 30, 1280, 720)  # narrow view
    old = DetectionFrame((Detection(Colour.BLACK, 1.46, 0.0, 0.6), Detection(Colour.YELLOW, -1.05, -0.29, 0.9)))
    new = DetectionFrame((Detection(Colour.BLACK, 1.462, 0.0, 0.9),))
    merged = merge_frames(old, old_cam, new, new_cam)
    by_colour = {d.colour: d for d in merged.detections}
    assert set(by_colour) == {Colour.BLACK, Colour.YELLOW}  # yellow is out of the new view: kept
    assert 1.46 < by_colour[Colour.BLACK].x < 1.462


def test_two_pockets_fix_the_pose_when_the_body_knows_height_and_roll():
    """Two pockets give 4 clues; the body's height and roll pin down the rest."""
    true = PinholeCamera.looking_at((-1.0, -1.5, 0.8), (0.2, 0.3, 0), 60, 1280, 720)
    # The body knows height and roll well, but is ~3 cm off on the floor and ~1 degree in aim.
    rough = PinholeCamera.looking_at((-0.97, -1.53, 0.803), (0.25, 0.27, 0), 60, 1280, 720, roll_deg=0.2)
    boxes = [b for px, py, pr in T.POCKETS if (b := true.pocket_box(px, py, pr, T.SNOOKER)) is not None
             and b[0] > 0 and b[1] > 0 and b[2] < 1280 and b[3] < 720]
    boxes = boxes[:2]
    assert len(boxes) == 2
    fit = refine_pose(rough, boxes)
    assert fit.pockets_used == 2
    # Balls land much closer to where they really are.
    spots = [(0.0, 0.0), (1.0, 0.5), (-1.0, 0.5), (1.5, -0.7)]

    def error(cam):
        return np.mean([math.dist(ball_position(cam, true.ball_box(x, y, T.BALL_R), T.BALL_R), (x, y))
                        for x, y in spots])

    assert error(fit.camera) < error(rough) / 3
    # The pockets-only fit (the old way) can't use just two.
    assert refine_pose(rough, boxes, body=None).camera is rough
