"""The same shot planner and simulator play pool, given pool's table and rules."""

import pytest

from mr147.physics import PhysicsSimulator
from mr147.pool import EightBallGame, Group, NineBallGame
from mr147.pool.shot_goals import eight_ball_goal, nine_ball_goal, outcome_of
from mr147.pool.table import POOL_TABLE, TABLE_LENGTH, TABLE_WIDTH
from mr147.shot_planner import PlannerSettings, ShotPlanner
from mr147.table_memory import TableState

QUICK = PlannerSettings(speeds=(1.0, 1.6), heights=(-0.3, 0.0, 0.3), stress_test_replays=8)


@pytest.fixture(scope="module")
def pool_physics():
    return PhysicsSimulator(POOL_TABLE)


def test_simulator_uses_the_pool_table(pool_physics):
    xs = sorted(abs(x) for x, _ in pool_physics.pockets)
    assert xs[0] == 0.0 and xs[-1] == pytest.approx(TABLE_LENGTH / 2, abs=0.05)
    assert max(abs(y) for _, y in pool_physics.pockets) == pytest.approx(TABLE_WIDTH / 2, abs=0.08)  # middle pockets sit back a little


def test_nine_ball_pots_the_lowest_ball_not_the_easier_nine(pool_physics):
    table = TableState({"white": (0.75, 0.30), "3": (1.0, 0.45), "9": (-0.9, 0.0)})
    game = NineBallGame(on_table={3, 9}, is_break=False)

    plan = ShotPlanner(pool_physics, QUICK).plan_for(table, nine_ball_goal(game))
    assert plan is not None and plan.target == "3"
    assert plan.chance_of_potting > 0.7

    result = pool_physics.simulate(table, plan.shot)
    game.play(outcome_of(result))
    assert game.last_foul is None and game.shooter == 0 and 3 not in game.on_table


def test_eight_ball_calls_the_pocket_it_aims_at_and_wins(pool_physics):
    table = TableState({"white": (0.75, 0.30), "8": (1.0, 0.45), "2": (-0.9, 0.0)})
    game = EightBallGame(on_table={2, 8}, is_break=False, shooter=1, groups={0: Group.SOLIDS, 1: Group.STRIPES})
    assert game.is_on_the_eight()

    plan = ShotPlanner(pool_physics, QUICK).plan_for(table, eight_ball_goal(game))
    assert plan is not None and plan.target == "8"

    result = pool_physics.simulate(table, plan.shot)
    game.play(outcome_of(result, called_pocket=plan.pocket))
    assert game.winner == 1


def test_eight_ball_never_plans_to_pot_the_eight_early(pool_physics):
    table = TableState({"white": (0.75, 0.30), "8": (1.0, 0.45), "10": (-0.9, 0.0)})
    game = EightBallGame(on_table={8, 10}, is_break=False, shooter=1, groups={0: Group.SOLIDS, 1: Group.STRIPES})
    plan = ShotPlanner(pool_physics, QUICK).plan_for(table, eight_ball_goal(game))
    assert plan is None or plan.target == "10"
