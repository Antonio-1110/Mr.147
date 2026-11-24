"""The close-up ball reader: crops cut from the full-size picture, names replaced by the reader."""

import numpy as np

from mr147.vision.detector import PixelDetection
from mr147.vision.identify import name_once_in_picture
from mr147.vision.reader import CROP_SIZE, crop_ball, reader_settings, reread


def test_crop_is_centred_on_the_ball_and_keeps_its_pixels():
    img = np.zeros((2000, 3000, 3), np.uint8)
    img[990:1010, 1490:1510] = 255  # a 20-pixel "ball" in a big picture
    crop = crop_ball(img, (1490, 990, 1510, 1010), pad=0.3)
    assert crop.shape == (CROP_SIZE, CROP_SIZE, 3)
    ys, xs = np.nonzero(crop[..., 0] > 128)
    assert abs(xs.mean() - CROP_SIZE / 2) < 3 and abs(ys.mean() - CROP_SIZE / 2) < 3
    # The ball fills the crop's middle: 1 / (1 + 2 * 0.3) of its width.
    assert abs((xs.max() - xs.min() + 1) / CROP_SIZE - 1 / 1.6) < 0.05


def test_crop_at_the_picture_edge_repeats_the_edge():
    img = np.full((100, 100, 3), 77, np.uint8)
    crop = crop_ball(img, (90, 90, 110, 110))
    assert crop.shape == (CROP_SIZE, CROP_SIZE, 3) and (crop == 77).all()


class FakeReader:
    def __init__(self, answers):
        self.answers = answers

    def read(self, image, boxes):
        return [self.answers[i] for i in range(len(boxes))]


def test_reader_replaces_ball_names_but_leaves_pockets_alone():
    found = [PixelDetection("8", (0, 0, 10, 10), 0.9), PixelDetection("pocket", (50, 50, 80, 80), 0.7),
             PixelDetection("8", (20, 0, 30, 10), 0.8)]
    reader = FakeReader([{"10": 0.7, "8": 0.2, "2": 0.1}, {"8": 0.95, "cue": 0.001}])
    out = reread(found, np.zeros((100, 100, 3), np.uint8), reader)
    by_box = {d.box: d for d in out}
    assert by_box[(0, 0, 10, 10)].name == "10" and by_box[(0, 0, 10, 10)].alternatives[1] == ("8", 0.2)
    assert by_box[(20, 0, 30, 10)].name == "8"
    assert by_box[(50, 50, 80, 80)].name == "pocket" and by_box[(50, 50, 80, 80)].confidence == 0.7
    # The reader's scores feed the one-of-each naming as before.
    assert sorted(d.name for d in name_once_in_picture(out, "pool")) == ["10", "8", "pocket"]


def test_reader_training_never_mirrors_or_recolours():
    s = reader_settings()
    assert s["fliplr"] == 0.0 and s["flipud"] == 0.0 and s["hsv_h"] == 0.0


def test_balls_off_the_cloth_are_dropped_but_a_tight_rack_is_kept():
    from mr147.vision.detector import on_cloth

    img = np.zeros((400, 600, 3), np.uint8)
    img[:] = (30, 30, 30)  # the room
    img[150:400, :] = (20, 110, 220)  # blue cloth on the lower part
    rack = [PixelDetection(str(n), (200 + 21 * k, 250 + 21 * r, 220 + 21 * k, 270 + 21 * r), 0.9)
            for n, (r, k) in enumerate([(r, k) for r in range(3) for k in range(3)], start=1)]
    for d in rack:  # paint the balls so their neighbours' rings see ball, not cloth
        x0, y0, x1, y1 = (int(v) for v in d.box)
        img[y0:y1, x0:x1] = (200, 40, 40)
    lone = PixelDetection("cue", (450, 300, 470, 320), 0.9)
    face = PixelDetection("15", (100, 40, 125, 65), 0.6)  # up in the room
    pocket = PixelDetection("pocket", (0, 10, 40, 40), 0.8)
    kept = on_cloth(rack + [lone, face, pocket], img)
    assert face not in kept and lone in kept and pocket in kept
    assert all(d in kept for d in rack)


def test_ball_sets_turn_looks_into_numbers():
    from mr147.vision.pool_balls import appearance, appearance_to_number

    modern = {1: "yellow", 2: "blue", 3: "red", 4: "pink", 5: "purple", 6: "green", 7: "brown", 8: "black"}
    assert [appearance(n, modern) for n in (0, 4, 5, 8, 12, 13, 15)] == \
        ["cue", "pink", "purple", "black", "pink-stripe", "purple-stripe", "brown-stripe"]
    assert appearance_to_number("purple", "traditional") == "4"
    assert appearance_to_number("purple", "modern") == "5"
    assert appearance_to_number("pink-stripe", "modern") == "12"
    assert appearance_to_number("brown-stripe", "modern") == "15"
    assert appearance_to_number("black", "modern") == "8" and appearance_to_number("cue", "modern") == "cue"
    # Every look the reader can give means some ball in every set.
    from mr147.vision.pool_balls import LOOK_SHADES, LOOK_TO_NUMBER
    for ball_set in LOOK_TO_NUMBER:
        assert set(LOOK_TO_NUMBER[ball_set]) == set(LOOK_SHADES)


def test_reread_adds_up_looks_that_mean_the_same_number():
    found = [PixelDetection("4", (0, 0, 10, 10), 0.9)]
    reader = FakeReader([{"brown": 0.4, "orange": 0.35, "purple": 0.25}])
    out = reread(found, np.zeros((20, 20, 3), np.uint8), reader, ball_set="modern")
    assert out[0].name == "7" and abs(out[0].confidence - 0.75) < 1e-6  # brown + orange both mean 7 here
    out = reread(found, np.zeros((20, 20, 3), np.uint8), reader, ball_set="traditional")
    assert out[0].name == "7" and dict(out[0].alternatives)["5"] == 0.35


def test_random_sets_come_in_both_colour_schemes():
    from mr147.vision.pool_balls import PoolBallStyle

    rng = np.random.default_rng(0)
    styles = [PoolBallStyle.random(rng) for _ in range(200)]
    modern = [s for s in styles if s.ball_set == "modern"]
    assert 60 < len(modern) < 140
    assert all(s.looks[4] == "pink" and s.looks[5] == "purple" for s in modern)
    assert all(s.looks[5] == "orange" for s in styles if s.ball_set == "traditional")


def test_chalk_on_the_rail_and_balls_on_another_table_are_dropped():
    from mr147.vision.detector import on_cloth

    img = np.full((600, 800, 3), 60, np.uint8)  # floor
    img[250:600, 100:700] = (25, 25, 25)  # black rail
    img[280:600, 130:670] = (20, 110, 220)  # this table's cloth
    img[40:140, 550:780] = (20, 110, 220)  # a smaller table further away
    ours = [PixelDetection(str(n), (200 + 60 * n, 400, 220 + 60 * n, 420), 0.9) for n in range(1, 5)]
    chalk = PixelDetection("15", (400, 255, 415, 270), 0.5)  # sitting on the rail
    far = PixelDetection("9", (600, 80, 610, 90), 0.8)  # on the other table
    kept = on_cloth(ours + [chalk, far], img)
    assert kept == ours
