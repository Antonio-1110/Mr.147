"""Pool through the vision -> memory link: ball numbers, one of each, pool checks."""

import math

import numpy as np
import pytest

from mr147.interfaces.vision import Detection, DetectionFrame
from mr147.table_memory import POOL_BALLS, TableMemory, TableState
from mr147.vision import table as T
from mr147.vision.camera_vision import CameraVision, merge_frames
from mr147.vision.detector import PixelDetection, group_boxes
from mr147.vision.identify import assign_names, name_limits
from mr147.vision.layouts import pool_scatter
from mr147.vision.locate import PinholeCamera
from mr147.vision.scene import CameraPose
from mr147.vision.sim_camera import PerfectDetector, SimCamera

# ---------------------------------------------------------------- naming the balls


def test_three_balls_called_8_get_one_8_each_number_once():
    scores = [
        {"8": 0.9},  # surely the 8
        {"8": 0.6, "4": 0.3},  # dark purple 4 mistaken for the 8
        {"8": 0.5, "15": 0.2, "12": 0.1},  # maroon stripe
    ]
    names = [n for n, _ in assign_names(scores, name_limits("pool"))]
    assert names == ["8", "4", "15"]


def test_a_ball_with_no_good_name_left_is_marked_unsure():
    scores = [{"8": 0.9}, {"8": 0.8}]  # the second has no other suggestion
    (n1, s1), (n2, s2) = assign_names(scores, name_limits("pool"))
    assert n1 == "8" and n2 != "8"
    assert s2 < 0.05  # memory will ask for a closer look at it


def test_snooker_allows_15_reds_but_one_black():
    scores = [{"red": 0.9}] * 15 + [{"black": 0.9, "red": 0.1}, {"black": 0.7, "blue": 0.2}]
    names = [n for n, _ in assign_names(scores, name_limits("snooker"))]
    assert names.count("red") == 15 and names.count("black") == 1 and names[-1] == "blue"


def test_boxes_on_the_same_ball_become_one_box_with_alternatives():
    found = [
        PixelDetection("8", (100, 100, 120, 120), 0.6),
        PixelDetection("4", (101, 100, 121, 120), 0.3),
        PixelDetection("2", (300, 100, 320, 120), 0.9),
        PixelDetection("pocket", (99, 99, 122, 121), 0.8),  # never merged with a ball
    ]
    grouped = group_boxes(found)
    by_name = {g.name: g for g in grouped}
    assert set(by_name) == {"8", "2", "pocket"}
    assert by_name["8"].alternatives == (("8", 0.6), ("4", 0.3))


# ---------------------------------------------------------------- pool table memory


def pool_frame(*balls) -> DetectionFrame:
    return DetectionFrame(tuple(Detection(name, x, y) for name, x, y in balls), game="pool")


START = [("white", -0.6, 0.0), ("1", 0.5, 0.0), ("8", 0.6, 0.2), ("3", 0.8, -0.3), ("11", -0.2, 0.4)]


def test_pool_first_look_counts_missing_balls_as_potted():
    memory = TableMemory(balls=POOL_BALLS)
    check = memory.first_look(pool_frame(*START))
    assert check.accepted
    assert set(check.table.positions) == {"white", "1", "8", "3", "11"}
    assert memory.number_potted() == 11  # 15 object balls, 4 seen


def test_pool_memory_rejects_two_8_balls_and_unknown_names():
    memory = TableMemory(balls=POOL_BALLS)
    check = memory.first_look(pool_frame(*START, ("8", -0.9, -0.4), ("red", 0.0, 0.5)))
    assert not check.accepted
    text = " ".join(check.view_request.reasons)
    assert "2 8 balls" in text and "'red' is not a pool ball" in text


def test_pool_shot_pots_the_planned_ball():
    memory = TableMemory(balls=POOL_BALLS)
    memory.first_look(pool_frame(*START))
    after = [b for b in START if b[0] != "3"]
    check = memory.after_shot(pool_frame(*after), planned_pots=["3"])
    assert check.accepted and check.potted == {"3": 1}
    # A ball can't come back.
    check = memory.after_shot(pool_frame(*START), planned_pots=[])
    assert not check.accepted


def test_merging_two_looks_that_disagree_on_the_number_leaves_it_unsure():
    cam = PinholeCamera.looking_at((-1.8, -1.1, 0.8), (0, 0, 0), 65, 1280, 720)
    old = DetectionFrame((Detection("8", 0.5, 0.1, 0.8, (("8", 0.8), ("4", 0.1))),), game="pool")
    new = DetectionFrame((Detection("4", 0.501, 0.1, 0.7, (("4", 0.7),)),), game="pool")
    merged = merge_frames(old, cam, new, cam, T.POOL)
    (d,) = merged.detections
    assert d.colour in ("8", "4") and d.confidence < 0.5  # unsure: memory will ask for another look


# ---------------------------------------------------------------- the whole chain, rendered


@pytest.fixture
def pool_table_and_vision(renderer):
    rng = np.random.default_rng(7)
    balls = pool_scatter(rng)
    while len(balls) < 8 or not any(b.name == "cue" for b in balls):
        balls = pool_scatter(rng)
    table = TableState({("white" if b.name == "cue" else b.name): (b.x, b.y) for b in balls})
    start = CameraPose((-1.75, -1.15, 0.75), (0.0, 0.0, 0.0), 65.0)
    cam = SimCamera(renderer, lambda: table, start, spec=T.POOL, seed=2)
    return table, CameraVision(cam, PerfectDetector(cam, seed=2), spec=T.POOL)


def test_rendered_pool_table_reaches_memory_with_ball_numbers(pool_table_and_vision):
    table, vision = pool_table_and_vision
    frame = vision.observe()
    assert frame.game == "pool"
    names = [str(d.colour) for d in frame.detections]
    assert len(names) == len(set(names))  # every number at most once
    assert set(names) <= set(table.positions)
    assert len(names) >= len(table.positions) - 1  # at most one ball hidden from both sides
    errors = [math.dist(d.position, table.positions[str(d.colour)]) for d in frame.detections]
    assert np.median(errors) < 0.015 and max(errors) < 0.04

    memory = TableMemory(balls=POOL_BALLS)
    check = memory.first_look(frame, trust_camera=True)
    assert check.accepted and set(check.table.positions) == set(names)


# ---------------------------------------------------------------- who filters what


def test_vision_drops_balls_on_the_next_table():
    from mr147.vision.locate import PoseFit

    cam = PinholeCamera.looking_at((-1.8, -1.2, 0.8), (0.5, 0, 0), 75, 1280, 720)
    vision = CameraVision(camera=None, detector=None, spec=T.POOL)
    ours = cam.ball_box(0.3, 0.1, T.POOL.ball_r)
    next_table = cam.ball_box(1.9, 0.3, T.POOL.ball_r)  # 0.6 m past the far cushion
    found = [PixelDetection("5", tuple(ours), 0.9), PixelDetection("9", tuple(next_table), 0.9)]
    frame = vision.to_frame(found, PoseFit(cam, 4, 0.5, {}))
    assert [str(d.colour) for d in frame.detections] == ["5"]
    assert vision.dropped_off_table == 1


def test_potted_numbers_are_avoided_unless_vision_is_sure():
    limits = name_limits("pool")
    in_play = {"white": 1, "8": 1, "12": 1}  # the 4 was potted
    names = assign_names([{"8": 0.9}, {"4": 0.5, "12": 0.3}], limits, in_play)
    assert [n for n, _ in names] == ["8", "12"]  # unsure "4" -> next-best live number
    (name, score), = assign_names([{"4": 0.98}], limits, in_play)
    assert name == "4" and score < 0.1  # a near-certain 4 stays, flagged for the memory


def test_memory_tells_vision_what_is_still_in_play():
    memory = TableMemory(balls=POOL_BALLS)
    assert memory.in_play() is None  # before the first look anything is possible
    memory.first_look(pool_frame(*START))
    after = [b for b in START if b[0] != "3"]
    memory.after_shot(pool_frame(*after), planned_pots=["3"])
    assert sorted(memory.in_play()) == sorted(["white", "1", "8", "11"])
    # A rejected frame carries the list, so vision's next look avoids potted numbers.
    check = memory.after_shot(pool_frame(*after, ("3", 0.9, 0.1)), planned_pots=[])
    assert not check.accepted and "3" not in check.view_request.in_play


def test_snooker_in_play_keeps_colours_that_get_respotted():
    from mr147.snooker.balls import Colour
    from mr147.snooker.table import SPOTS

    memory = TableMemory()
    start = [(Colour.WHITE, 0.0, 0.0), (Colour.RED, 0.5, 0.3), (Colour.RED, 0.8, -0.4),
             (Colour.BLACK, *SPOTS[Colour.BLACK])]
    memory.first_look(DetectionFrame(tuple(Detection(c, x, y) for c, x, y in start)))
    in_play = memory.in_play()
    assert in_play.count("red") == 2 and "pink" in in_play  # pink could be respotted
