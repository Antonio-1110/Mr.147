"""Ball in hand: where to put the cue ball before a shot.

A player puts the cue ball where the next pot is easy and the shot after it is
easy too. We do the same, cheaply first:

  1. For every ball we may pot and every pocket with a clear path, try spots a
     short way behind the ghost ball: straight in, and at small angles (an angle
     lets the cue ball travel on for position).
  2. Keep the spots where geometry says the pot is easiest.
  3. Run the full shot planner from the best few, and keep the best plan.
"""

import math

from mr147.pool.table import HEAD_STRING_X, Position, distance
from mr147.shot_planner import ShotGoal, ShotPlanner, pot_geometry
from mr147.shot_planner.shot_planner import PlannedShot
from mr147.table_memory.table_state import TableState

DISTANCES = (0.18, 0.32, 0.5)  # metres behind the ghost ball
ANGLES_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0)  # cut angle we leave ourselves


def candidate_spots(
    table: TableState, goal: ShotGoal, planner: ShotPlanner, behind_head_string: bool = False
) -> list[tuple[float, Position]]:
    """(geometric ease, cue ball spot), easiest first."""
    game = planner.physics.table
    others = [p for b, p in table.positions.items() if b != game.cue_ball]
    spots = []
    for target in goal.targets:
        if target not in table.positions:
            continue
        ball = table.positions[target]
        for pocket in planner.physics.pockets:
            ghost = pot_geometry.ghost_ball_position(ball, pocket, game.ball_diameter)
            back = math.radians(pot_geometry.direction_deg(pocket, ball))  # from the pocket, through the ball, onwards
            for angle in ANGLES_DEG:
                a = back + math.radians(angle)
                for d in DISTANCES:
                    spot = (ghost[0] + d * math.cos(a), ghost[1] + d * math.sin(a))
                    if not game.is_on_table(spot) or any(distance(spot, o) < game.ball_diameter * 1.05 for o in others):
                        continue
                    if behind_head_string and spot[0] >= HEAD_STRING_X:
                        continue
                    trial = table.with_ball_at(game.cue_ball, spot)
                    option = pot_geometry.pot_option(trial, target, planner.physics.pockets.index(pocket), pocket, game)
                    if option is not None:
                        spots.append((option.ease, spot))
    spots.sort(key=lambda pair: pair[0], reverse=True)
    return spots


def place_and_plan(
    table: TableState,
    goal: ShotGoal,
    planner: ShotPlanner,
    behind_head_string: bool = False,
    spots_to_plan: int = 4,
) -> tuple[Position, PlannedShot] | None:
    """The best place for the cue ball and the shot to play from there (None if nothing works)."""
    best = None
    tried: list[Position] = []
    for _, spot in candidate_spots(table, goal, planner, behind_head_string):
        if any(distance(spot, t) < 0.1 for t in tried):  # don't plan almost the same spot twice
            continue
        tried.append(spot)
        plan = planner.plan_for(table.with_ball_at(planner.physics.table.cue_ball, spot), goal)
        if plan is not None and (best is None or plan.score > best[1].score):
            best = (spot, plan)
        if len(tried) >= spots_to_plan:
            break
    return best
