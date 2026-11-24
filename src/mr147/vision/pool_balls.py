"""How pool balls are painted, and how ball sets differ from each other.

Every maker's set follows the same plan (1-7 solids, 8 black, 9-15 stripes in the
same colours as 1-7, numbers in a white circle), but the details differ:

  * colours: traditional sets have a purple 4, orange 5 and maroon 7; modern (TV)
    sets have a pink 4, purple 5 and orange-brown 7 (BALL_SETS)
  * shades: the 2 can be navy or a bright royal blue, the 6 dark green, bright
    green or teal, and so on (LOOK_SHADES)
  * the white: bright white to old ivory
  * the stripes' caps (either side of the stripe): white on most sets, black on some
  * stripe width, number circle size, number font and thickness, a thin ring
    round the circle on some sets
  * the cue ball: plain, a maker's logo, or red/blue dots ("measles" cue balls)
  * shine: glossy new balls to dull, worn ones

The generator picks one random set per table (a real table has one set), so the
detector learns the plan rather than one maker's exact colours.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import cv2
import numpy as np

# How each colour can look (RGB 0..1). The first is the usual one.
LOOK_SHADES = {
    "yellow": [(0.95, 0.72, 0.04), (0.98, 0.80, 0.10), (0.93, 0.62, 0.02)],
    "blue": [(0.04, 0.14, 0.62), (0.05, 0.28, 0.80), (0.02, 0.08, 0.42)],  # navy .. royal
    "red": [(0.78, 0.05, 0.04), (0.90, 0.10, 0.08), (0.65, 0.02, 0.06)],
    "purple": [(0.30, 0.08, 0.46), (0.45, 0.15, 0.62), (0.42, 0.22, 0.72)],
    "pink": [(0.85, 0.30, 0.55), (0.95, 0.42, 0.52), (0.92, 0.35, 0.45)],
    "orange": [(0.96, 0.40, 0.05), (0.95, 0.30, 0.04), (1.00, 0.52, 0.10)],
    "green": [(0.02, 0.42, 0.16), (0.05, 0.55, 0.25), (0.00, 0.45, 0.40)],  # .. teal
    "maroon": [(0.42, 0.06, 0.07), (0.55, 0.05, 0.12)],
    "brown": [(0.45, 0.20, 0.08), (0.78, 0.42, 0.16), (0.70, 0.36, 0.14)],  # brown .. orange-brown
    "black": [(0.02, 0.02, 0.02), (0.03, 0.03, 0.06)],
}

# Ball sets: which look each number 1-8 can have (9-15 are stripes in the 1-7 colours).
# Traditional: 4 purple, 5 orange, 7 maroon. Modern (TV) sets: 4 pink, 5 purple,
# 7 orange-brown, usually with black stripe caps.
BALL_SETS = {
    "traditional": {1: ["yellow"], 2: ["blue"], 3: ["red"], 4: ["purple", "purple", "purple", "pink"],
                    5: ["orange"], 6: ["green"], 7: ["maroon", "brown"], 8: ["black"]},
    "modern": {1: ["yellow"], 2: ["blue"], 3: ["red"], 4: ["pink"], 5: ["purple"], 6: ["green"],
               7: ["brown"], 8: ["black"]},
}

# The reader names a ball by how it looks; a ball-set setting turns that into a number.
# Every look maps to a number in every set (a look the set doesn't use goes to the
# nearest one), so the reader's scores can always be added up per number.
LOOK_TO_NUMBER = {
    "traditional": {"yellow": 1, "blue": 2, "red": 3, "purple": 4, "pink": 4, "orange": 5, "green": 6,
                    "maroon": 7, "brown": 7, "black": 8},
    "modern": {"yellow": 1, "blue": 2, "red": 3, "pink": 4, "purple": 5, "green": 6, "brown": 7,
               "orange": 7, "maroon": 7, "black": 8},
}


def appearance(num: int, looks: dict) -> str:
    """The reader's class for ball `num` in a set with these looks: "cue", "black", "pink", "pink-stripe"."""
    if num == 0:
        return "cue"
    look = looks[(num - 1) % 8 + 1]
    return look if num <= 8 else f"{look}-stripe"


def appearance_to_number(name: str, ball_set: str) -> str:
    """Reader class -> ball label ("cue", "1".."15") for the given ball set."""
    if name == "cue" or name.isdigit():
        return name  # an older reader that already gives numbers
    look, stripe = (name[:-7], True) if name.endswith("-stripe") else (name, False)
    n = LOOK_TO_NUMBER[ball_set][look]
    return str(n + 8 if stripe and n != 8 else n)


# Shade options per number for the traditional set (kept for older callers).
SHADES = {n: LOOK_SHADES[looks[0]] for n, looks in BALL_SETS["traditional"].items()}

FONTS = [cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_TRIPLEX,
         cv2.FONT_HERSHEY_COMPLEX]


@dataclass(frozen=True)
class PoolBallStyle:
    """One ball set's look."""

    colours: dict = field(default_factory=lambda: {n: s[0] for n, s in SHADES.items()})
    ball_set: str = "traditional"
    looks: dict = field(default_factory=lambda: {n: looks[0] for n, looks in BALL_SETS["traditional"].items()})
    white: tuple = (0.95, 0.94, 0.91)
    stripe_caps: tuple | None = None  # colour either side of the stripe; None = the set's white
    stripe_half: float = 0.19  # half the stripe's height, as a fraction of pole-to-pole
    # Number circle radius, as a fraction of pole-to-pole. On a real ball the circle
    # is about 0.4 of the ball's width across, which is ~0.13 here.
    circle_r: float = 0.13
    circle_ring: bool = False  # a thin coloured ring round the number circle
    font: int = cv2.FONT_HERSHEY_DUPLEX
    font_scale: float = 1.0
    thickness: int = 2
    cue: str = "plain"  # "plain", "logo" or "dots"
    cue_mark: tuple = (0.75, 0.05, 0.05)  # colour of the cue ball's logo / dots
    shine: float = 1.0  # multiplies how glossy the balls look
    name: str = "classic"

    @classmethod
    def random(cls, rng: np.random.Generator, ball_set: str | None = None) -> "PoolBallStyle":
        ball_set = ball_set or str(rng.choice(list(BALL_SETS)))
        colours, looks = {}, {}
        for n, options in BALL_SETS[ball_set].items():
            looks[n] = str(options[int(rng.integers(len(options)))])
            shades = LOOK_SHADES[looks[n]]
            rgb = shades[int(rng.integers(len(shades)))]
            h, s, v = colorsys.rgb_to_hsv(*rgb)
            # A little extra drift so no two sets are exactly alike.
            h = (h + rng.normal(0, 0.012)) % 1.0
            s = float(np.clip(s * rng.uniform(0.9, 1.08), 0, 1))
            v = float(np.clip(v * rng.uniform(0.88, 1.1), 0, 1))
            colours[n] = colorsys.hsv_to_rgb(h, s, v)
        ivory = rng.uniform(0, 1) ** 2  # mostly white, sometimes yellowed
        white = (0.96 - 0.03 * ivory, 0.95 - 0.06 * ivory, 0.93 - 0.16 * ivory)
        cue = str(rng.choice(["plain", "logo", "dots"], p=[0.45, 0.3, 0.25]))
        black_caps = 0.85 if ball_set == "modern" else 0.3  # modern sets mostly have black-capped stripes
        caps = tuple(np.full(3, rng.uniform(0.02, 0.07))) if rng.random() < black_caps else None
        return cls(
            colours=colours,
            ball_set=ball_set,
            looks=looks,
            white=white,
            stripe_caps=caps,
            stripe_half=float(rng.uniform(0.15, 0.24)),
            circle_r=float(rng.uniform(0.12, 0.15)),
            circle_ring=bool(rng.random() < 0.25),
            font=int(rng.choice(FONTS)),
            font_scale=float(rng.uniform(0.95, 1.2)),
            thickness=int(rng.choice([2, 2, 3])),
            cue=cue,
            cue_mark=(0.75, 0.05, 0.05) if rng.random() < 0.7 else (0.05, 0.15, 0.6),
            shine=float(rng.uniform(0.5, 1.3)),
            name=f"random-{int(rng.integers(1_000_000))}",
        )


CLASSIC = PoolBallStyle()


def pool_ball_image(num: int, style: PoolBallStyle = CLASSIC, w: int = 512, h: int = 256) -> np.ndarray:
    """Wrap-around picture (RGB uint8) of one ball: u around the equator, v pole to pole."""
    white = (np.array(style.white) * 255).astype(np.uint8)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = white
    if num == 0:
        _paint_cue_ball(img, style)
        return img
    col = (np.array(style.colours[(num - 1) % 8 + 1]) * 255).astype(np.uint8)
    if num <= 8:
        img[:] = col
    else:
        if style.stripe_caps is not None:
            img[:] = (np.array(style.stripe_caps) * 255).astype(np.uint8)
        band = int(h * style.stripe_half)
        img[h // 2 - band:h // 2 + band] = col
    r = int(h * style.circle_r)
    text = str(num)
    base_scale = (0.95 if len(text) == 1 else 0.75) * style.font_scale * style.circle_r / 0.105
    for cu in (w // 4, 3 * w // 4):  # printed on two opposite sides
        if style.circle_ring:
            cv2.circle(img, (cu, h // 2), r + 3, (np.array(col) * 0.6).astype(int).tolist(), -1, cv2.LINE_AA)
        cv2.circle(img, (cu, h // 2), r, white.tolist(), -1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(text, style.font, base_scale, style.thickness)
        cv2.putText(img, text, (cu - tw // 2, h // 2 + th // 2), style.font, base_scale,
                    (20, 20, 20), style.thickness, cv2.LINE_AA)
    return img


def _paint_cue_ball(img: np.ndarray, style: PoolBallStyle) -> None:
    h, w = img.shape[:2]
    mark = (np.array(style.cue_mark) * 255).astype(int).tolist()
    if style.cue == "logo":  # a small maker's mark on two sides
        for cu in (w // 4, 3 * w // 4):
            cv2.circle(img, (cu, h // 2), int(h * 0.06), mark, -1, cv2.LINE_AA)
            cv2.circle(img, (cu, h // 2), int(h * 0.035), (np.array(style.white) * 255).astype(int).tolist(),
                       -1, cv2.LINE_AA)
    elif style.cue == "dots":  # six dots spread over the ball, as on training cue balls
        r = int(h * 0.045)
        for u, v in [(0.0, 0.5), (0.25, 0.5), (0.5, 0.5), (0.75, 0.5), (0.0, 0.12), (0.5, 0.88)]:
            cv2.circle(img, (int(u * w) % w, int(v * h)), r, mark, -1, cv2.LINE_AA)
