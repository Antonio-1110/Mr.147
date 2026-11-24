"""A 9-foot pool table: size, ball size, pockets and spots.

Same coordinates as snooker (mr147.snooker.table): metres, (0, 0) at the centre,
x along the length, y across. The head end, where you break from, is negative x.

This file is the one place these numbers live. Vision should read them from
here (tests/test_table_numbers.py checks the two agree).
"""

import math

from mr147.game_table import GameTable

Position = tuple[float, float]

# Playing area, between the cushion faces.
TABLE_LENGTH = 2.54
TABLE_WIDTH = 1.27
BALL_DIAMETER = 0.05715
BALL_RADIUS = BALL_DIAMETER / 2

CORNER_POCKET_RADIUS = 0.062
MIDDLE_POCKET_RADIUS = 0.067
MIDDLE_POCKET_SET_BACK = 0.025
POCKETS: list[Position] = [
    (-TABLE_LENGTH / 2, -TABLE_WIDTH / 2),
    (-TABLE_LENGTH / 2, TABLE_WIDTH / 2),
    (TABLE_LENGTH / 2, -TABLE_WIDTH / 2),
    (TABLE_LENGTH / 2, TABLE_WIDTH / 2),
    (0.0, -TABLE_WIDTH / 2 - MIDDLE_POCKET_SET_BACK),
    (0.0, TABLE_WIDTH / 2 + MIDDLE_POCKET_SET_BACK),
]

# Pocket openings (the gap between the cushion jaws). Pool rules allow
# 4.5-4.625 inches for corners and 5-5.125 inches for middles: these are the middle of that.
CORNER_POCKET_OPENING = 0.1159
MIDDLE_POCKET_OPENING = 0.1286

# The head string is the line a quarter of the way in from the head end. After a
# scratch on the break, the cue ball goes anywhere behind it (in the "kitchen").
HEAD_STRING_X = -TABLE_LENGTH / 4
HEAD_SPOT: Position = (HEAD_STRING_X, 0.0)
# The rack goes on the foot spot. A respotted ball goes back here.
FOOT_SPOT: Position = (TABLE_LENGTH / 4, 0.0)

# Everything the shot chooser and simulator need, in one place.
POOL_TABLE = GameTable(
    name="pool",
    length=TABLE_LENGTH,
    width=TABLE_WIDTH,
    ball_diameter=BALL_DIAMETER,
    corner_pocket_opening=CORNER_POCKET_OPENING,
    middle_pocket_opening=MIDDLE_POCKET_OPENING,
    simulator_style="pool",
)


def is_on_table(position: Position, slack: float = 0.0) -> bool:
    """Could a ball centre be at this position? `slack` allows for measurement error."""
    x, y = position
    return abs(x) <= TABLE_LENGTH / 2 - BALL_RADIUS + slack and abs(y) <= TABLE_WIDTH / 2 - BALL_RADIUS + slack


def is_behind_head_string(position: Position) -> bool:
    """Is this cue ball position allowed after a scratch on the break?"""
    return position[0] < HEAD_STRING_X and is_on_table(position)


def respot_position(occupied: list[Position]) -> Position:
    """Where a respotted ball goes: the foot spot, or the nearest free place behind it
    on the centre line, towards the foot cushion."""
    x = FOOT_SPOT[0]
    while any(distance((x, 0.0), p) < BALL_DIAMETER for p in occupied):
        x += 0.001
        if x > TABLE_LENGTH / 2 - BALL_RADIUS:  # no room behind: try in front instead
            x = FOOT_SPOT[0]
            while any(distance((x, 0.0), p) < BALL_DIAMETER for p in occupied):
                x -= 0.001
            break
    return (x, 0.0)


def distance(a: Position, b: Position) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
