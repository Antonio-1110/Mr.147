"""The shot-plan web page: the planner remembers what it tried, and the page carries it."""

import json
import random
import re

from mr147.interfaces.cue_arm import Shot
from mr147.physics import PhysicsSimulator
from mr147.pool import NineBallGame
from mr147.pool.layouts import random_pool_table
from mr147.pool.shot_goals import nine_ball_goal
from mr147.pool.simulated_table import SimulatedPoolTable
from mr147.pool.table import POOL_TABLE
from mr147.shot_planner import CueArmError, PlannerSettings, ShotPlanner, Uncertainty
from mr147.shot_planner.plan_viewer import PlanRecorder

QUICK = PlannerSettings(
    pot_options_to_try=3, speeds=(1.0, 2.0), heights=(-0.4, 0.0, 0.4), stress_test_replays=4,
    confirm_replays=6, uncertainty=Uncertainty(cue_arm=CueArmError(0, 0, 0), camera_error_m=0.0),
)


def test_every_planned_shot_ends_up_on_the_page(tmp_path):
    # No break here: a 9 m/s break is chaotic, so its outcome differs between computers.
    physics = PhysicsSimulator(POOL_TABLE)
    start = random_pool_table(random.Random(2), ("1", "2", "3", "4"))
    table = SimulatedPoolTable(start, physics, cue_arm_error=None, camera_error_m=0.0)
    recorder = PlanRecorder(physics, "test")
    planner = recorder.watch_planner(ShotPlanner(physics, QUICK))
    arm = recorder.watch_cue_arm(table)

    arm.play(Shot(0.0, 0.5))  # a shot nobody planned (like the break) is still shown
    plan = planner.plan_for(table.true_table, nine_ball_goal(NineBallGame(on_table={1, 2, 3, 4}, is_break=False)))
    assert plan is not None
    arm.play(plan.shot)

    page = recorder.write(tmp_path / "plans.html").read_text()
    data = json.loads(re.search(r"const D=(.*?);\n", page).group(1))
    assert [s["kind"] for s in data["shots"]] == ["break", "shot"]
    shot = data["shots"][1]
    assert shot["tried"] and shot["finalists"]
    chosen = shot["finalists"][shot["chosen"]]
    assert chosen["dir"] == shot["played"]["dir"]  # the row marked chosen is the shot that was played
    assert shot["actual"], "what really happened is recorded"


def test_tried_shots_pot_the_ball_they_are_labelled_with():
    physics = PhysicsSimulator(POOL_TABLE)
    table = random_pool_table(random.Random(2), ("1", "2", "3", "4"))
    planner = ShotPlanner(physics, QUICK)
    planner.plan_for(table, nine_ball_goal(NineBallGame(on_table={1, 2, 3, 4}, is_break=False)))
    assert planner.last_search is not None and planner.last_search.tried
    for t in planner.last_search.tried:
        assert t.target in planner.last_search.targets
        result = physics.simulate(table, t.shot)
        assert t.target in result.potted_ball_ids
