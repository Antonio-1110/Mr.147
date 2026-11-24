"""Pool starting positions: racks for the break, and random tables for practice.

All positions in table coordinates (see mr147.pool.table). Ball ids are the ones
a TableState uses: "white" for the cue ball, "1".."15" for the rest.

Real racks are never perfect, so every racked ball is nudged by a fraction of a
millimetre. The break planner tests each break on many such racks.
"""

import math
import random

from mr147.pool.balls import ball_id
from mr147.pool.table import BALL_DIAMETER, FOOT_SPOT, HEAD_STRING_X, TABLE_LENGTH, TABLE_WIDTH, Position, distance
from mr147.table_memory.table_state import TableState

ROW_STEP = BALL_DIAMETER * math.sqrt(3) / 2  # rows of a tight triangle


def _rack_spots(rows: list[int], gap: float = 0.0) -> list[Position]:
    """Ball centres for a rack whose rows (from the apex on the foot spot) hold these many balls.

    E.g. [1, 2, 3, 2, 1] is the 9-ball diamond, [1, 2, 3, 4, 5] the 15-ball triangle.
    """
    spots = []
    for r, n in enumerate(rows):
        x = FOOT_SPOT[0] + r * (ROW_STEP + gap)
        for i in range(n):
            spots.append((x, (i - (n - 1) / 2) * (BALL_DIAMETER + gap)))
    return spots


def _jitter(position: Position, rng: random.Random, gap: float) -> Position:
    """Nudge a ball by up to a third of the gap, so neighbours never overlap."""
    return (position[0] + rng.uniform(-gap / 3, gap / 3), position[1] + rng.uniform(-gap / 3, gap / 3))


def nine_ball_rack(rng: random.Random, gap: float = 0.0005, cue_ball: Position | None = None) -> TableState:
    """1 at the apex on the foot spot, 9 in the middle, the rest anywhere."""
    spots = _rack_spots([1, 2, 3, 2, 1], gap)
    others = [2, 3, 4, 5, 6, 7, 8]
    rng.shuffle(others)
    numbers = {spots[0]: 1, spots[4]: 9}  # spot 4 is the centre of the diamond
    rest = iter(others)
    for spot in spots:
        if spot not in numbers:
            numbers[spot] = next(rest)
    positions = {ball_id(n): _jitter(s, rng, gap) for s, n in numbers.items()}
    positions["white"] = cue_ball or (HEAD_STRING_X - 0.05, 0.0)
    return TableState(positions)


def eight_ball_rack(rng: random.Random, gap: float = 0.0005, cue_ball: Position | None = None) -> TableState:
    """15 balls, 8 in the middle of the third row, one solid and one stripe in the back corners."""
    spots = _rack_spots([1, 2, 3, 4, 5], gap)
    centre = spots[4]  # third row, middle
    back_left, back_right = spots[10], spots[14]
    solids, stripes = [1, 2, 3, 4, 5, 6, 7], [9, 10, 11, 12, 13, 14, 15]
    rng.shuffle(solids)
    rng.shuffle(stripes)
    corners = [solids.pop(), stripes.pop()]
    rng.shuffle(corners)
    numbers = {centre: 8, back_left: corners[0], back_right: corners[1]}
    rest = solids + stripes
    rng.shuffle(rest)
    rest_iter = iter(rest)
    for spot in spots:
        if spot not in numbers:
            numbers[spot] = next(rest_iter)
    positions = {ball_id(n): _jitter(s, rng, gap) for s, n in numbers.items()}
    positions["white"] = cue_ball or (HEAD_STRING_X - 0.05, 0.0)
    return TableState(positions)


def random_pool_table(rng: random.Random, numbers: list[int]) -> TableState:
    """These balls spread at random (no clusters), cue ball anywhere: a table to practise running out."""
    positions: dict[str, Position] = {}

    def free_spot() -> Position:
        while True:
            p = (
                rng.uniform(-TABLE_LENGTH / 2 + 0.08, TABLE_LENGTH / 2 - 0.08),
                rng.uniform(-TABLE_WIDTH / 2 + 0.08, TABLE_WIDTH / 2 - 0.08),
            )
            if all(distance(p, q) > 2.2 * BALL_DIAMETER for q in positions.values()):
                return p

    for n in numbers:
        positions[ball_id(n)] = free_spot()
    positions["white"] = free_spot()
    return TableState(positions)
