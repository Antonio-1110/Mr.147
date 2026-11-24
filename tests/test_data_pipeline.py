"""Guard rails for the picture maker: the bugs we have hit before, so they can't come back.

  * rendering must not get slower with every table (the shadow-buffer leak)
  * the cloth must be well exposed, not burnt out to flat cyan / lime
  * a small generated set must pass the same checks mr147-train runs first
  * the trainer must never mirror or recolour pictures in ways that change a ball's identity
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from mr147.vision import table as T
from mr147.vision.check_data import blown_fraction, check_dataset
from mr147.vision.layouts import random_layout
from mr147.vision.pool_balls import PoolBallStyle
from mr147.vision.scene import (CAPS, CLOTH_ID, POOL_CLOTHS, CameraPose, _blown_cloth, _cloth_level, random_camera,
                                random_look)
from mr147.vision.train import augment_settings


def test_rebuilding_tables_does_not_pile_up_render_buffers(renderer):
    rng = np.random.default_rng(1)
    engine = renderer.base.graphicsEngine

    def build_and_render():
        renderer.build(random_layout(rng, T.POOL), random_look(rng, T.POOL), T.POOL)
        renderer.set_camera(random_camera(rng, T.POOL))
        renderer.render_frame()

    def others():  # buffers that aren't the current lamps' shadow buffers
        return engine.getNumWindows() - len(renderer._lamps)

    build_and_render()
    first = others()
    for _ in range(6):
        build_and_render()
        assert others() == first  # the old lamps' shadow buffers are freed (tables have 2-5 lamps)


@pytest.mark.parametrize("spec", [T.POOL, T.SNOOKER], ids=lambda s: s.game)
def test_cloth_is_well_exposed(renderer, spec):
    rng = np.random.default_rng(5)
    ratios, seen = [], []
    for _ in range(5):
        renderer.build(random_layout(rng, spec), random_look(rng, spec), spec)
        renderer.set_camera(random_camera(rng, spec))
        rgb, _ = renderer.render_frame()
        cloth = renderer.cloth_mask
        assert _blown_cloth(rgb, cloth) < 0.03
        if (level := _cloth_level(rgb, cloth)) is not None:
            ratios.append(level / renderer.cloth_target)
            seen.append(f"cloth {level:.0f} (aim {renderer.cloth_target:.0f}), exposure {renderer.exposure:.2f}, "
                        f"burnt {_blown_cloth(rgb, cloth):.3f}")
    assert ratios and 0.7 <= np.median(ratios) <= 1.15, "; ".join(seen)


@pytest.mark.parametrize("cloth", list(POOL_CLOTHS))
def test_every_cloth_colour_is_exposed_without_burning_out(renderer, cloth):
    rng = np.random.default_rng(8)
    look = random_look(rng, T.POOL)
    _, lo, hi = POOL_CLOTHS[cloth]
    look.cloth_rgb = tuple((np.array(lo) + np.array(hi)) / 2)
    renderer.build(random_layout(rng, T.POOL), look, T.POOL)
    renderer.set_camera(CameraPose((-1.9, -1.2, 0.75), (0.2, 0.0, 0.0), 65.0))
    rgb, ids = renderer.render_frame()
    cloth_px = renderer.cloth_mask
    assert cloth_px.mean() > 0.2  # the renderer knows which pixels are cloth, whatever the colour
    assert not np.isin(ids, [CLOTH_ID]).any()  # callers only ever see balls and pockets
    level = _cloth_level(rgb, cloth_px)
    assert _blown_cloth(rgb, cloth_px) < 0.03
    assert 0.6 * renderer.cloth_target <= level <= 1.2 * renderer.cloth_target, (cloth, level, renderer.cloth_target)


def test_pool_tables_get_many_cloth_colours_and_pocket_looks():
    rng = np.random.default_rng(3)
    looks = [random_look(rng, T.POOL) for _ in range(300)]
    cloths = {lk.cloth_name for lk in looks}
    assert {"electric blue", "green", "grey"} <= cloths and len(cloths) >= 7
    assert {lk.pockets["cap"] for lk in looks} == set(CAPS)
    assert all(random_look(rng, T.SNOOKER).cloth_name == "green" for _ in range(20))


def test_about_half_the_ball_sets_have_black_stripe_edges():
    rng = np.random.default_rng(4)
    black = sum(PoolBallStyle.random(rng).stripe_caps is not None for _ in range(400))
    assert 150 <= black <= 250


def test_ball_numbers_are_a_realistic_size():
    rng = np.random.default_rng(2)
    for _ in range(50):
        style = PoolBallStyle.random(rng)
        # Real number circles are ~0.4 of the ball's width: 0.12-0.15 of pole-to-pole here.
        assert 0.12 <= style.circle_r <= 0.15


@pytest.mark.parametrize("game", ["pool", "snooker"])
def test_trainer_never_mirrors_numbers_or_shifts_colours(game):
    s = augment_settings(game)
    assert s["hsv_h"] == 0.0  # a hue shift turns one ball into another
    assert s["flipud"] == 0.0
    if game == "pool":
        assert s["fliplr"] == 0.0  # a mirrored "6" isn't a 6


def test_a_generated_set_passes_the_training_checks(tmp_path):
    # Panda3D allows one window per process, so the generator runs in its own process.
    out = tmp_path / "synth-pool"
    subprocess.run([sys.executable, "-m", "mr147.vision.generate", "--game", "pool", "--out", str(out),
                    "--layouts", "3", "--views", "1", "--width", "480", "--height", "270",
                    "--val", "0.4", "--seed", "4"], check=True, capture_output=True,
                   env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")})
    problems, stats = check_dataset(out, "pool", require_all_classes=False)
    # A 3-layout set may by chance put every layout in one split; that's the only allowed complaint.
    assert [p for p in problems if not p.startswith("no ")] == []
    meta = json.loads(next((out / "meta").glob("*.json")).read_text())
    assert meta["game"] == "pool" and "pool_balls" in meta["look"] and "cap" in meta["look"]["pockets"]

    # The close-up reader's training crops come from the same set.
    from mr147.vision.reader import make_crops

    counts = make_crops(out, tmp_path / "crops")
    assert counts and all(k.split("/")[1] not in ("pocket",) for k in counts)
    # Crops are sorted by look (the reader learns looks; the ball-set setting gives numbers).
    from mr147.vision.pool_balls import LOOK_SHADES

    looks = set(LOOK_SHADES) | {f"{c}-stripe" for c in LOOK_SHADES} | {"cue"}
    assert {k.split("/")[1] for k in counts} <= looks
    assert meta["look"]["pool_balls"]["ball_set"] in ("traditional", "modern")
    one = next((tmp_path / "crops").glob("*/*/*.jpg"))
    assert cv2.imread(str(one)).shape == (128, 128, 3)


def test_check_dataset_catches_broken_labels(tmp_path):
    (tmp_path / "data.yaml").write_text("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(T.POOL.classes)))
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
    import cv2

    cv2.imwrite(str(tmp_path / "images/train/a.jpg"), np.zeros((20, 20, 3), np.uint8))
    (tmp_path / "labels/train/a.txt").write_text("99 0.5 0.5 0.1 0.1\n3 1.2 0.5 0.1 0.1\n")
    (tmp_path / "labels/train/orphan.txt").write_text("")
    problems, _ = check_dataset(tmp_path, "pool", require_all_classes=False)
    text = " ".join(problems)
    assert "2 bad label lines" in text and "no picture" in text and "no val pictures" in text


def test_pocket_rail_parts_never_hang_over_the_cloth():
    """The rail-level cut-out, liner and cap stop at the cushion line (no floating black lid)."""
    from mr147.vision.scene import _off_cloth
    from mr147.vision.table import POOL

    hl, hw = POOL.length / 2, POOL.width / 2
    corner, middle = _off_cloth((hl, hw), hl, hw), _off_cloth((0.0, hw + 0.025), hl, hw)
    assert corner(-0.03, -0.02) == (0.0, 0.0)  # over the cloth: pulled back to the corner
    assert corner(0.03, -0.02) == (0.03, -0.02)  # over the rail: unchanged
    x, y = middle(0.01, -0.06)
    assert abs(x - 0.01) < 1e-9 and abs(0.025 + y) < 1e-9  # onto the cushion line


def test_stripes_are_on_the_table_about_as_often_as_solids():
    import collections

    from mr147.vision.layouts import random_pool_layout

    rng, seen = np.random.default_rng(0), collections.Counter()
    for _ in range(1500):
        seen.update(b.label for b in random_pool_layout(rng))
    solids = np.mean([seen[str(n)] for n in range(1, 8)])
    stripes = np.mean([seen[str(n)] for n in range(9, 16)])
    assert stripes > 0.8 * solids


def test_grey_cloth_stays_grey():
    from mr147.vision.scene import random_look

    rng = np.random.default_rng(1)
    for _ in range(2000):
        look = random_look(rng, T.POOL)
        if look.cloth_name == "grey":
            c = np.array(look.cloth_rgb)
            assert (c.max() - c.min()) / c.mean() < 0.15, look.cloth_rgb
