"""The ball position check (vision/locate_check.py) and the pocket shape it relies on."""

import numpy as np

from mr147.vision import table as T
from mr147.vision.locate import PinholeCamera
from mr147.vision.locate_check import Settings, match_to_truth, run, write
from mr147.vision.scene import CameraPose


def test_pocket_box_matches_what_the_renderer_labels(renderer):
    """The pose fix compares seen pocket boxes with predicted ones; both must use the same shape.

    (They didn't: the renderer cuts the rail part of a pocket off at the cushion line,
    the prediction drew a full circle, and the pose fix pulled a perfect camera ~5 mm off.)
    """
    for spec in (T.POOL, T.SNOOKER):
        pose = CameraPose((-spec.length / 2 - 0.6, -spec.width / 2 - 0.5, 0.8), (0.3, 0.1, 0.0), 70.0)
        renderer.build([], _look(spec), spec)
        renderer.set_camera(pose)
        cam = PinholeCamera.from_pose(pose, renderer.w, renderer.h)
        for k, (px, py, pr) in enumerate(spec.pockets):
            outline = renderer.pocket_outline(k)
            want = np.concatenate([outline.min(0), outline.max(0)])
            got = cam.pocket_box(px, py, pr, spec)
            assert np.abs(got - want).max() < 0.5, (spec.game, k, got, want)


def _look(spec):
    from mr147.vision.scene import random_look

    return random_look(np.random.default_rng(0), spec)


def test_perfect_boxes_and_pose_give_no_error(renderer):
    s = Settings(game="pool", tables=1, views=2, pose_error_m=0.0, roll_error_deg=0.0, box_noise_px=0.0, seed=3)
    res = run(renderer, s)
    assert len(res.rows) >= 5
    assert max(r["error_mm"] for r in res.rows) < 1.0
    assert max(v["pose_error_mm"] for v in res.views) < 1.0


def test_small_run_writes_a_report(renderer, tmp_path):
    s = Settings(game="pool", tables=1, views=3, seed=1)
    res = run(renderer, s, keep_examples=2)
    text = write(res, s, tmp_path, renderer.w, renderer.h, 1.0)
    assert "3+ pockets (robot trusts)" in text
    for name in ("summary.txt", "balls.csv", "error-map.jpg", "views.jpg"):
        assert (tmp_path / name).exists()
    fixed = [r for r in res.rows if r["pockets"] >= 3]
    assert fixed, "expected at least one picture with 3+ pockets"
    # Where the pockets fixed the pose, it beats trusting the robot's rough idea of its head.
    assert np.median([r["error_mm"] for r in fixed]) < np.median([r["error_no_fix_mm"] for r in fixed])


def test_match_to_truth_pairs_by_position():
    info = [{"centre_px": [100, 100], "box": [90, 90, 110, 110]},
            {"centre_px": [200, 100], "box": [190, 90, 210, 110]},
            {"centre_px": [300, 100], "box": [290, 90, 310, 110]}]
    found = [(191, 91, 211, 111), (89, 90, 109, 110), (500, 500, 520, 520)]
    pairs, missed, false = match_to_truth(found, info)
    assert sorted(i for i, _ in pairs) == [0, 1]
    assert missed == 1 and false == 1
