import pytest

from mr147.interfaces.cue_arm import Shot
from mr147.physics import PhysicsSimulator
from mr147.snooker.robot import SnookerRobot
from mr147.shot_planner import PlannerSettings, ShotPlanner
from mr147.snooker.simulated_table import SimulatedTable
from mr147.snooker.balls import Colour
from mr147.snooker.table import BALL_DIAMETER
from mr147.table_memory import TableState

EASY_RED = TableState({"white": (0.8, 0.1), "red_01": (1.3, 0.5), "black": (1.46, 0.0)})


@pytest.fixture(scope="module")
def physics():
    return PhysicsSimulator()


def test_pooltool_coordinates_are_converted_correctly(physics):
    straight_at_black = physics.simulate(
        TableState({"white": (0.0, 0.0), "black": (0.5, 0.0)}), Shot(direction_deg=0, speed=1.0)
    )
    assert straight_at_black.first_ball_hit == "black"
    assert abs(straight_at_black.first_ball_hit_direction_deg) < 0.01

    towards_plus_y = physics.simulate(
        TableState({"white": (0.5, 0.0), "black": (1.46, 0.0)}), Shot(direction_deg=90, speed=1.0)
    )
    x, y = towards_plus_y.cue_ball_path[2]  # where it first meets the cushion
    assert abs(x - 0.5) < 1e-3 and y > 0.85


def test_planned_pot_goes_in_and_cue_ball_stops_where_predicted(physics):
    plan = ShotPlanner(physics).plan(EASY_RED, Colour.RED, next_ball_on=Colour.BLACK)
    assert plan is not None
    assert plan.target == "red_01"
    assert plan.chance_of_potting > 0.8

    result = physics.simulate(EASY_RED, plan.shot)
    assert result.potted_ball_ids == ("red_01",)
    assert result.final_table.cue_ball == plan.cue_ball_stops_at


def test_no_plan_when_the_red_is_hidden_behind_the_black(physics):
    snookered = TableState({"white": (0.0, 0.0), "black": (0.6, 0.0), "red_01": (0.6 + BALL_DIAMETER + 0.001, 0.0)})
    planner = ShotPlanner(physics, PlannerSettings(stress_test_replays=0))
    assert planner.plan(snookered, Colour.RED, next_ball_on=Colour.BLACK) is None


def test_robot_pots_red_then_black_on_simulated_table(physics):
    table = SimulatedTable(EASY_RED, physics, cue_arm_error=None, camera_error_m=0.0)
    quick = PlannerSettings(speeds=(1.0, 2.0), heights=(-0.4, 0.0, 0.4), stress_test_replays=4)
    robot = SnookerRobot(vision=table, cue_arm=table, planner=ShotPlanner(physics, quick))

    robot.look_at_table()
    robot.play_one_shot()
    assert robot.rules.score == 1
    assert robot.memory.number_potted(Colour.RED) == 15

    robot.play_one_shot()  # the black, which gets respotted
    assert robot.rules.score == 8
    assert "black" in robot.memory.table.positions
