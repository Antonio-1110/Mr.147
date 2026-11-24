"""Tied-up tables, split shots and the one-shot look-ahead."""

import random

import pytest

from mr147.physics import PhysicsSimulator
from mr147.shot_planner import PlannerSettings, ShotPlanner, pot_geometry
from mr147.shot_planner.table_openness import table_openness
from mr147.snooker.layouts import after_break_off, full_rack
from mr147.snooker.balls import Colour
from mr147.snooker.table import SNOOKER_TABLE

QUICK = PlannerSettings(
    pot_options_to_try=3,
    speeds=(1.0, 2.0),
    heights=(-0.4, 0.0, 0.4),
    split_speeds=(2.0, 3.0),
    split_heights=(-0.4, 0.0, 0.4),
    stress_test_replays=6,
)


@pytest.fixture(scope="module")
def physics():
    return PhysicsSimulator()


def openness(table, physics):
    return table_openness(table, table.ball_ids(Colour.RED), physics.pockets, SNOOKER_TABLE)


def test_full_rack_has_every_ball_and_is_completely_tied_up(physics):
    rack = full_rack(random.Random(0))
    assert len(rack.ball_ids(Colour.RED)) == 15 and len(rack.positions) == 22
    assert openness(rack, physics) == 0.0


def test_break_off_leaves_a_tied_up_table_with_a_red_to_start_on(physics):
    table = after_break_off(random.Random(3), physics)
    assert openness(table, physics) <= 0.5
    assert pot_geometry.all_pot_options(table, Colour.RED, physics.pockets)


def test_planner_looks_one_shot_ahead_on_a_tied_up_table(physics):
    table = after_break_off(random.Random(3), physics)
    plan = ShotPlanner(physics, QUICK).plan(table, Colour.RED, Colour.BLACK)
    assert plan is not None
    assert 0.0 < plan.chance_break_continues <= 1.0
    assert plan.score <= plan.chance_of_potting * plan.chance_break_continues


def test_split_search_finds_shots_into_the_pack(physics):
    table = after_break_off(random.Random(3), physics)
    planner = ShotPlanner(physics, QUICK)
    planner.plan(table, Colour.RED, Colour.BLACK)  # sets up the tied-up balls
    from mr147.snooker.shot_goal import snooker_goal

    goal = snooker_goal(table, Colour.RED, Colour.BLACK)
    pots = pot_geometry.all_pot_options(table, goal.targets, physics.pockets)[:2]
    splits = planner._split_shots(table, pots, goal)
    assert all(s.splits for s in splits)
