"""The pool player: racks, break, ball in hand, and a run-out on the simulated table."""

import random

import pytest

from mr147.physics import PhysicsSimulator
from mr147.pool import NineBallGame
from mr147.pool.ball_in_hand import place_and_plan
from mr147.pool.break_shot import BreakPlanner, BreakSettings
from mr147.pool.layouts import eight_ball_rack, nine_ball_rack, random_pool_table
from mr147.pool.robot import PoolRobot
from mr147.pool.shot_goals import nine_ball_goal
from mr147.pool.simulated_table import SimulatedPoolTable
from mr147.pool.table import BALL_DIAMETER, HEAD_STRING_X, POOL_TABLE, distance
from mr147.shot_planner import CueArmError, PlannerSettings, ShotPlanner, Uncertainty

QUICK = PlannerSettings(
    pot_options_to_try=3, speeds=(1.0, 2.0), heights=(-0.4, 0.0, 0.4), stress_test_replays=4,
    confirm_replays=6, uncertainty=Uncertainty(cue_arm=CueArmError(0, 0, 0), camera_error_m=0.0),
)
QUICK_BREAK = BreakSettings(cue_ys=(-0.25, 0.0), speeds=(9.0,), aim_offsets=(0.0,), racks=3, confirm_top=1, confirm_racks=3)


@pytest.fixture(scope="module")
def physics():
    return PhysicsSimulator(POOL_TABLE)


@pytest.mark.parametrize("rack, balls", [(nine_ball_rack, 9), (eight_ball_rack, 15)])
def test_racks_have_every_ball_and_none_overlap(rack, balls):
    table = rack(random.Random(0))
    assert len(table.positions) == balls + 1
    spots = [p for b, p in table.positions.items() if b != "white"]
    for i, a in enumerate(spots):
        for b in spots[i + 1:]:
            assert distance(a, b) >= BALL_DIAMETER


def test_nine_ball_rack_has_the_1_in_front_and_the_9_in_the_middle():
    table = nine_ball_rack(random.Random(0))
    xs = sorted((p[0], b) for b, p in table.positions.items() if b != "white")
    assert xs[0][1] == "1"
    assert abs(table.positions["9"][1]) < 0.002


def test_break_planner_finds_a_break_that_keeps_the_table(physics):
    plan = BreakPlanner(physics, "nine", QUICK_BREAK).plan(random.Random(0))
    assert plan.cue_ball[0] < HEAD_STRING_X
    assert 0.0 <= plan.score <= plan.keeps_table <= 1.0


def test_ball_in_hand_places_the_cue_ball_for_an_easy_pot(physics):
    table = random_pool_table(random.Random(1), [3, 9])
    planner = ShotPlanner(physics, QUICK)
    spot, plan = place_and_plan(table, nine_ball_goal(NineBallGame(on_table={3, 9}, is_break=False)), planner)
    assert plan.target == "3" and plan.chance_of_potting > 0.5
    assert all(distance(spot, p) >= BALL_DIAMETER for b, p in table.positions.items() if b != "white")


def test_robot_breaks_and_pots_balls_on_a_perfect_table(physics):
    table = SimulatedPoolTable(nine_ball_rack(random.Random(0)), physics, cue_arm_error=None, camera_error_m=0.0)
    robot = PoolRobot(table, table, table, ShotPlanner(physics, QUICK), "nine", BreakPlanner(physics, "nine", QUICK_BREAK))
    report = robot.run_rack(max_shots=4)
    assert report.shots >= 1
    assert report.balls_potted == 9 - len(robot.game.on_table)
