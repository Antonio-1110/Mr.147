from mr147.physics import PhysicsSimulator
from mr147.shot_planner import PlannerSettings, ShotPlanner
from mr147.shot_planner.table_openness import table_openness
from mr147.snooker.balls import Colour
from mr147.snooker.table import BALL_DIAMETER, SNOOKER_TABLE, SPOTS
from mr147.table_memory import TableState

D = BALL_DIAMETER
POCKETS = PhysicsSimulator().pockets


def triangle(apex, rows):
    """A tight triangle of reds pointing at -x, like the snooker rack."""
    reds, k = {}, 0
    for row in range(rows):
        for i in range(row + 1):
            k += 1
            reds[f"red_{k:02d}"] = (apex[0] + row * D * 0.866, apex[1] + (i - row / 2) * D)
    return reds


def test_full_rack_is_tied_up():
    table = TableState({"white": (-1.2, 0.0), **triangle((1.0, 0.0), 5)})
    assert table_openness(table, table.ball_ids(Colour.RED), POCKETS, SNOOKER_TABLE) == 0.0


def test_spread_reds_are_open():
    table = TableState({"white": (-1.2, 0.0), "red_01": (0.5, 0.4), "red_02": (-0.5, -0.4), "red_03": (1.0, -0.2)})
    assert table_openness(table, table.ball_ids(Colour.RED), POCKETS, SNOOKER_TABLE) == 1.0


def test_two_reds_nearly_touching_count_as_tied_up():
    table = TableState({"white": (-1.2, 0.0), "red_01": (0.5, 0.4), "red_02": (0.5 + D + 0.002, 0.4)})
    assert table_openness(table, table.ball_ids(Colour.RED), POCKETS, SNOOKER_TABLE) == 0.0


def test_nothing_left_to_clear_is_fully_open():
    assert table_openness(TableState({"white": (0, 0)}), [], POCKETS, SNOOKER_TABLE) == 1.0


def test_planner_reports_openness_after_the_shot():
    physics = PhysicsSimulator()
    table = TableState({"white": (1.15, 0.12), "black": SPOTS[Colour.BLACK], **triangle((0.95, 0.0), 3)})
    quick = PlannerSettings(speeds=(1.0, 2.5), heights=(-0.4, 0.0, 0.4), stress_test_replays=4)
    plan = ShotPlanner(physics, quick).plan(table, Colour.BLACK, next_ball_on=Colour.RED, respotted_colours={Colour.BLACK})
    assert plan is not None
    assert 0.0 <= plan.table_openness <= 1.0
