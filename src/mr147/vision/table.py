"""Table and ball facts for snooker and pool, in metres.

Coordinates: table centre is (0, 0), x runs along the length (baulk / head end
is -x), y across the width, z up. The cloth surface is z = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TableSpec:
    game: str  # "snooker" or "pool"
    length: float  # playing area between cushion faces
    width: float
    ball_d: float
    cloth_height: float  # cloth surface above the floor
    cushion_h: float  # cushion nose above the cloth
    corner_pocket_r: float
    middle_pocket_r: float
    middle_pocket_offset: float  # how far the middle pocket centre sits behind the cushion line
    classes: tuple[str, ...]  # label names, in label-number order
    cushion_w: float = 0.05
    rail_w: float = 0.13
    rail_h: float = 0.045

    @property
    def ball_r(self) -> float:
        return self.ball_d / 2

    @property
    def class_id(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.classes)}

    @property
    def pockets(self) -> list[tuple[float, float, float]]:
        """Pocket centres (x, y) and radius."""
        hl, hw, c, m = self.length / 2, self.width / 2, self.corner_pocket_r, self.middle_pocket_r
        o = self.middle_pocket_offset
        return [(-hl, -hw, c), (-hl, hw, c), (hl, -hw, c), (hl, hw, c), (0.0, -hw - o, m), (0.0, hw + o, m)]


# --------------------------------------------------------------------------- snooker

# Class names in label order. All 15 reds share one class. Pockets are labelled
# too, because their known positions let us work out where the camera is.
CLASSES = ("red", "yellow", "green", "brown", "blue", "pink", "black", "white", "pocket")

SNOOKER = TableSpec("snooker", 3.569, 1.778, 0.0525, 0.86, 0.037, 0.075, 0.068, 0.02, CLASSES)

# Kept as plain names because the snooker layout code reads them a lot.
LENGTH, WIDTH, BALL_D = SNOOKER.length, SNOOKER.width, SNOOKER.ball_d
BALL_R = BALL_D / 2
CLOTH_HEIGHT = SNOOKER.cloth_height
POCKETS = SNOOKER.pockets

BAULK_X = -LENGTH / 2 + 0.737  # baulk line, measured from the baulk cushion
D_RADIUS = 0.292

# Base colours (0..1). The generator jitters these a little.
BALL_RGB = {
    "red": (0.62, 0.03, 0.03),
    "yellow": (0.95, 0.78, 0.06),
    "green": (0.02, 0.38, 0.14),
    "brown": (0.36, 0.17, 0.06),
    "blue": (0.04, 0.13, 0.58),
    "pink": (0.95, 0.48, 0.58),
    "black": (0.02, 0.02, 0.02),
    "white": (0.93, 0.93, 0.88),
}

# Colour spots (x, y).
SPOTS = {
    "yellow": (BAULK_X, -D_RADIUS),
    "brown": (BAULK_X, 0.0),
    "green": (BAULK_X, D_RADIUS),
    "blue": (0.0, 0.0),
    "pink": (LENGTH / 4, 0.0),
    "black": (LENGTH / 2 - 0.324, 0.0),
}

# --------------------------------------------------------------------------- pool

# Label 0 is the cue ball, then label n is ball number n.
POOL_CLASSES = ("cue",) + tuple(str(n) for n in range(1, 16)) + ("pocket",)

# 9-foot table (2.54 x 1.27 m playing area), 57.15 mm balls.
POOL = TableSpec("pool", 2.54, 1.27, 0.05715, 0.78, 0.040, 0.062, 0.067, 0.025, POOL_CLASSES,
                 cushion_w=0.05, rail_w=0.14, rail_h=0.045)

POOL_HEAD_X = -POOL.length / 4  # head string / head spot
POOL_FOOT_X = POOL.length / 4  # foot spot, where the rack goes

# Ball colours for 1-8; 9-15 are white balls with a stripe of the colour of n - 8.
POOL_RGB = {
    1: (0.95, 0.72, 0.04),  # yellow
    2: (0.04, 0.14, 0.62),  # blue
    3: (0.78, 0.05, 0.04),  # red
    4: (0.30, 0.08, 0.46),  # purple
    5: (0.96, 0.40, 0.05),  # orange
    6: (0.02, 0.42, 0.16),  # green
    7: (0.42, 0.06, 0.07),  # maroon
    8: (0.02, 0.02, 0.02),  # black
}

SPECS = {"snooker": SNOOKER, "pool": POOL}


@dataclass
class Ball:
    name: str  # class name, e.g. "red", "pink", "cue", "9"
    x: float
    y: float
    index: int = 0  # 0..14 for snooker reds, 0 otherwise

    @property
    def label(self) -> str:
        return f"red{self.index + 1}" if self.name == "red" else self.name


def off_cloth(centre, half_l: float, half_w: float):
    """Moves points (relative to `centre`) that lie over the playing area onto its edge.

    The pocket's rail-level parts (the cut-out, liner and cap) must stop at the cushion
    line; drawn as full circles they would hover over the cloth like a black lid.
    Corner pockets sit on the playing area's corner, so their points go to the corner.
    """
    px, py = centre
    corner = abs(abs(px) - half_l) < 1e-6 and abs(abs(py) - half_w) < 1e-6

    def snap(x: float, y: float) -> tuple[float, float]:
        wx, wy = px + x, py + y
        if abs(wx) >= half_l or abs(wy) >= half_w:
            return x, y
        if corner:
            return 0.0, 0.0
        if half_l - abs(wx) < half_w - abs(wy):
            return math.copysign(half_l, wx) - px, y
        return x, math.copysign(half_w, wy) - py

    return snap
