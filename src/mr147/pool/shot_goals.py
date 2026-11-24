"""What a pool shot must do, for the shot planner.

The rules decide. For each simulated shot we replay the outcome on a copy of
the game: the shot is good if it isn't a foul and we either keep the table or
win. So everything the rules know (groups, calling the 8, lowest ball first,
the 9 off a combination) is used automatically.

The planner looks for direct pots of the balls we may hit first. It doesn't
plan safeties, combinations, or where to put the cue ball with ball in hand.
"""

import copy

from mr147.physics import ShotResult
from mr147.pool.balls import ball_id, number_of
from mr147.pool.eight_ball import EightBallGame
from mr147.pool.nine_ball import NineBallGame
from mr147.pool.shot import ShotOutcome
from mr147.shot_planner.shot_goal import ShotGoal
from mr147.table_memory.table_state import TableState

PoolGame = EightBallGame | NineBallGame


def outcome_of(result: ShotResult, called_pocket: int | None = None) -> ShotOutcome:
    """Turn a simulator result into what the pool rules need.

    Pocket numbers are the simulator's (PhysicsSimulator.pockets), for the call and the result alike.
    """
    return ShotOutcome(
        first_ball_hit=None if result.first_ball_hit is None else number_of(result.first_ball_hit),
        potted=tuple(number_of(b) for b in result.potted_ball_ids),
        pockets={number_of(b): p for b, p in zip(result.potted_ball_ids, result.pockets_entered)},
        cushion_after_contact=result.cushion_after_first_hit,
        called_pocket=called_pocket,
    )


def _legal_first_balls(game: PoolGame) -> frozenset[int]:
    if isinstance(game, NineBallGame):
        return frozenset({game.ball_on()})
    return game.legal_first_balls()


def _goal(game: PoolGame) -> ShotGoal:
    shooter = game.shooter

    def game_after(result: ShotResult, pocket: int | None) -> PoolGame:
        after = copy.deepcopy(game)
        after.play(outcome_of(result, called_pocket=pocket))
        return after

    def is_success(result: ShotResult, pocket: int) -> bool:
        after = game_after(result, pocket)
        if after.last_foul is not None:
            return False
        return after.winner == shooter if after.is_over else after.shooter == shooter

    def next_targets(result: ShotResult, table_after: TableState) -> list[str] | None:
        after = game_after(result, None)
        if after.is_over:
            return None
        return [ball_id(n) for n in _legal_first_balls(after) if ball_id(n) in table_after.positions]

    def balls_to_clear(table_after: TableState) -> list[str]:
        return [b for b in table_after.positions if b != "white"]

    targets = frozenset(ball_id(n) for n in _legal_first_balls(game))
    return ShotGoal(targets, is_success, next_targets, balls_to_clear=balls_to_clear)


def eight_ball_goal(game: EightBallGame) -> ShotGoal:
    """Pot one of your own balls (or the 8 into the pocket we aim at) and keep the table."""
    return _goal(game)


def nine_ball_goal(game: NineBallGame) -> ShotGoal:
    """Pot the lowest ball (or the 9 off it) and keep the table."""
    return _goal(game)
