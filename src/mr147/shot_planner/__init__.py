"""Choose the shot: pot the ball on, and leave the cue ball well for the next one.

    pot_geometry.py  quick, physics-free checks: aim point, clear paths, how easy a pot is
    uncertainty.py   how imprecise the cue arm and camera are
    shot_goal.py     ShotGoal: what a shot must do (each game folder builds its own)
    shot_planner.py  the search: try shots in the physics simulator, keep the best
    plan_viewer.py   a web page showing every shot it considered and where the cue ball goes

Works for any game: give the simulator the game's table and the planner the game's goal.
"""

from mr147.shot_planner.shot_goal import ShotGoal
from mr147.shot_planner.shot_planner import PlannedShot, PlannerSettings, PlanSearch, ShotPlanner, TriedShot
from mr147.shot_planner.uncertainty import CueArmError, Uncertainty

__all__ = ["CueArmError", "PlannedShot", "PlannerSettings", "PlanSearch", "ShotGoal", "ShotPlanner", "TriedShot", "Uncertainty"]
