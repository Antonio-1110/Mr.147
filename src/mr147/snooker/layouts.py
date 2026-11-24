"""Snooker starting positions: open tables, a full rack, and tied-up tables after a break-off."""

import random

from mr147.snooker.balls import CUE_BALL, make_red_id
from mr147.snooker.table import BALL_DIAMETER, SPOTS, TABLE_LENGTH, TABLE_WIDTH, Position, distance
from mr147.table_memory.table_state import TableState


def open_table(rng: random.Random, reds: int = 15) -> TableState:
    """Colours on their spots, reds spread out (no clusters), cue ball near the middle."""
    positions: dict[str, Position] = {colour.value: spot for colour, spot in SPOTS.items()}

    def random_free_position(max_x: float, max_y: float) -> Position:
        while True:
            candidate = (rng.uniform(-max_x, max_x), rng.uniform(-max_y, max_y))
            if all(distance(candidate, p) > 2.2 * BALL_DIAMETER for p in positions.values()):
                return candidate

    for i in range(reds):
        positions[make_red_id(i)] = random_free_position(TABLE_LENGTH / 2 - 0.15, TABLE_WIDTH / 2 - 0.12)
    positions[CUE_BALL.value] = random_free_position(0.6, 0.4)
    return TableState(positions)


# ---- tied-up tables: a real rack, then a simulated break-off ---------------------------


def full_rack(rng: random.Random, cue_ball: Position | None = None) -> TableState:
    """Colours on their spots, 15 reds racked behind the pink, cue ball in the D.

    Real racks are never perfect, so each red is nudged by up to half a millimetre.
    """
    from mr147.snooker.balls import Colour
    from mr147.snooker.table import BAULK_LINE_X, D_RADIUS

    positions: dict[str, Position] = {colour.value: spot for colour, spot in SPOTS.items()}
    apex_x = SPOTS[Colour.PINK][0] + BALL_DIAMETER + 0.001
    row_step = BALL_DIAMETER * 0.866  # rows of a tight triangle are sqrt(3)/2 diameters apart
    k = 0
    for row in range(5):
        for i in range(row + 1):
            x = apex_x + row * (row_step + 0.0003)
            y = (i - row / 2) * (BALL_DIAMETER + 0.0003)
            positions[make_red_id(k)] = (x + rng.uniform(0, 0.0005), y + rng.uniform(-0.00025, 0.00025))
            k += 1
    if cue_ball is None:  # anywhere in the D
        while True:
            x, y = rng.uniform(BAULK_LINE_X - D_RADIUS, BAULK_LINE_X), rng.uniform(-D_RADIUS, D_RADIUS)
            if distance((x, y), (BAULK_LINE_X, 0.0)) < D_RADIUS - BALL_DIAMETER and all(
                distance((x, y), p) > BALL_DIAMETER for p in positions.values()
            ):
                cue_ball = (x, y)
                break
    positions[CUE_BALL.value] = cue_ball
    return TableState(positions)


def after_break_off(rng: random.Random, physics, max_tries: int = 200) -> TableState:
    """A realistic tied-up table: rack the balls, play a break-off into the pack, keep the result.

    The break-off is random: from the D, at 3 to 7 m/s, at a random point on the pack, with
    random side. Kept only if the cue ball stays on the table, the colours are still there
    (a potted colour is put back on its spot), most reds are still clustered, and at least
    one red can be potted, so the robot has somewhere to start.
    """
    from mr147.interfaces.cue_arm import Shot
    from mr147.shot_planner import pot_geometry
    from mr147.shot_planner.table_openness import table_openness
    from mr147.snooker.balls import Colour, colour_of
    from mr147.snooker.table import SNOOKER_TABLE

    for _ in range(max_tries):
        rack = full_rack(rng)
        reds = [rack.positions[r] for r in rack.ball_ids(Colour.RED)]
        target = rng.choice(reds)  # aim at one red's edge, so the hit is thin or full at random
        aim = pot_geometry.direction_deg(rack.cue_ball, target) + rng.uniform(-0.8, 0.8)
        shot = Shot(aim, rng.uniform(3.0, 7.0), side=rng.uniform(-0.4, 0.4), height=rng.uniform(-0.2, 0.3))
        result = physics.simulate(rack, shot)
        table = result.final_table
        if CUE_BALL.value not in table.positions:
            continue
        for ball_id in result.potted_ball_ids:
            colour = colour_of(ball_id)
            if colour != Colour.RED:
                table = table.with_ball_at(colour.value, SPOTS[colour])
        red_ids = table.ball_ids(Colour.RED)
        if len(red_ids) < 12:
            continue
        if table_openness(table, red_ids, physics.pockets, SNOOKER_TABLE) > 0.5:
            continue  # too open: we want tables where splitting matters
        if not pot_geometry.all_pot_options(table, Colour.RED, physics.pockets):
            continue  # no red to start with
        return table
    raise RuntimeError("no usable break-off found")
