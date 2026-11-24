"""Table size, ball size, pockets and colour spots for a full-size snooker table.

Coordinates (the same ones vision uses):
    metres, (0, 0) is the centre of the table,
    x runs along the length: baulk end is negative x, black end is positive x,
    y runs across the width.

This file is the one place these numbers live. The brain reads them from here,
and vision should too (tests/test_table_numbers.py checks the two agree).
"""

import math

from mr147.game_table import GameTable
from mr147.snooker.balls import Colour

Position = tuple[float, float]

# Playing area, between the cushion faces.
TABLE_LENGTH = 3.569
TABLE_WIDTH = 1.778
BALL_DIAMETER = 0.0525
BALL_RADIUS = BALL_DIAMETER / 2

# Pockets: centre and radius. Middle pocket centres sit a little behind the cushion line.
CORNER_POCKET_RADIUS = 0.075
MIDDLE_POCKET_RADIUS = 0.068
MIDDLE_POCKET_SET_BACK = 0.02
POCKETS: list[Position] = [
    (-TABLE_LENGTH / 2, -TABLE_WIDTH / 2),
    (-TABLE_LENGTH / 2, TABLE_WIDTH / 2),
    (TABLE_LENGTH / 2, -TABLE_WIDTH / 2),
    (TABLE_LENGTH / 2, TABLE_WIDTH / 2),
    (0.0, -TABLE_WIDTH / 2 - MIDDLE_POCKET_SET_BACK),
    (0.0, TABLE_WIDTH / 2 + MIDDLE_POCKET_SET_BACK),
]

# Pocket openings (the gap between the cushion jaws), as the simulator models them.
# These are pooltool's standard snooker values: measure your own table and change them here.
CORNER_POCKET_OPENING = 0.083
MIDDLE_POCKET_OPENING = 0.087

# Baulk line and the D.
BAULK_LINE_X = -TABLE_LENGTH / 2 + 0.737
D_RADIUS = 0.292

# Where each colour is put back when it is respotted.
SPOTS: dict[Colour, Position] = {
    Colour.YELLOW: (BAULK_LINE_X, -D_RADIUS),
    Colour.BROWN: (BAULK_LINE_X, 0.0),
    Colour.GREEN: (BAULK_LINE_X, D_RADIUS),
    Colour.BLUE: (0.0, 0.0),
    Colour.PINK: (TABLE_LENGTH / 4, 0.0),
    Colour.BLACK: (TABLE_LENGTH / 2 - 0.324, 0.0),
}

# Everything the shot chooser and simulator need, in one place.
SNOOKER_TABLE = GameTable(
    name="snooker",
    length=TABLE_LENGTH,
    width=TABLE_WIDTH,
    ball_diameter=BALL_DIAMETER,
    corner_pocket_opening=CORNER_POCKET_OPENING,
    middle_pocket_opening=MIDDLE_POCKET_OPENING,
    simulator_style="snooker",
)


def is_on_table(position: Position, slack: float = 0.0) -> bool:
    """Could a ball centre be at this position? `slack` allows for measurement error."""
    x, y = position
    max_x = TABLE_LENGTH / 2 - BALL_RADIUS + slack
    max_y = TABLE_WIDTH / 2 - BALL_RADIUS + slack
    return abs(x) <= max_x and abs(y) <= max_y


def distance(a: Position, b: Position) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
