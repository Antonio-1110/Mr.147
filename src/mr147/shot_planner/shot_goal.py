"""ShotGoal: what a shot has to do. This is the only game-specific input to the planner.

Each game builds its own from its rules:

    mr147.snooker.shot_goal.snooker_goal(...)        pot the ball on (the 147 sequence)
    mr147.pool.shot_goals.eight_ball_goal(game)      8-ball
    mr147.pool.shot_goals.nine_ball_goal(game)       9-ball
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from mr147.physics import ShotResult
from mr147.table_memory.table_state import TableState


def _no_respots(table_after: TableState, result: ShotResult) -> TableState:
    return table_after


@dataclass(frozen=True)
class ShotGoal:
    """
    targets:      ids of the balls we try to pot with this shot
    is_success:   (result, pocket aimed at) -> did the shot do its job?
                  E.g. snooker: exactly the ball on went in. Pool: no foul and we keep the table.
    next_targets: (result, table after) -> ids of the balls we could pot next, to judge
                  where the cue ball stopped. None means there is no next ball.
    respot:       (table after, result) -> the table once the referee has put balls back
    balls_to_clear: (table after) -> ids of the balls still to be cleared later, used to
                  score how open the table is (see table_openness.py). None = don't care.
    """

    targets: frozenset[str]
    is_success: Callable[[ShotResult, int], bool]
    next_targets: Callable[[ShotResult, TableState], Iterable[str] | None]
    respot: Callable[[TableState, ShotResult], TableState] = _no_respots
    balls_to_clear: Callable[[TableState], Iterable[str]] | None = None
