"""What a snooker shot must do, for the shot planner.

In a 147 a shot succeeds only if the cue ball hits the ball on first, exactly
one ball on goes in, and the cue ball stays on the table.
"""

from collections.abc import Iterable

from mr147.physics import ShotResult
from mr147.shot_planner.shot_goal import ShotGoal
from mr147.snooker.balls import CUE_BALL, Colour, colour_of
from mr147.snooker.table import SPOTS
from mr147.table_memory.table_state import TableState


def snooker_goal(
    table: TableState,
    ball_on: Colour,
    next_ball_on: Colour | None,
    respotted_colours: Iterable[Colour] = (),
) -> ShotGoal:
    respotted = frozenset(respotted_colours)

    def is_success(result: ShotResult, pocket: int) -> bool:
        if result.first_ball_hit is None or colour_of(result.first_ball_hit) != ball_on:
            return False
        if len(result.potted_ball_ids) != 1 or colour_of(result.potted_ball_ids[0]) != ball_on:
            return False
        return CUE_BALL.value in result.final_table.positions

    def respot(table_after: TableState, result: ShotResult) -> TableState:
        for ball_id in result.potted_ball_ids:
            colour = colour_of(ball_id)
            if colour in respotted:
                table_after = table_after.with_ball_at(colour.value, SPOTS[colour])
        return table_after

    def next_targets(result: ShotResult, table_after: TableState) -> list[str] | None:
        return None if next_ball_on is None else table_after.ball_ids(next_ball_on)

    def balls_to_clear(table_after: TableState) -> list[str]:
        return table_after.ball_ids(Colour.RED)  # the colours sit on their spots: only reds get tied up

    return ShotGoal(frozenset(table.ball_ids(ball_on)), is_success, next_targets, respot, balls_to_clear)
