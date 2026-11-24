"""Pool (pocket billiards): ball numbers, the 9-foot table, and the rules.

    balls.py        ball numbers 0-15, solids and stripes, colour names
    table.py        9-foot table size, pockets, head string, foot spot
    shot.py         ShotOutcome: what happened on a shot, as the rules need it
    eight_ball.py   8-ball rules: groups, fouls, ball in hand, win or lose on the 8
    nine_ball.py    9-ball rules: lowest ball first, three fouls, win on the 9
    shot_goals.py   tells the shot planner what a good pool shot is (uses the rules above)
    layouts.py      9-ball and 8-ball racks (slightly imperfect), random practice tables
    break_shot.py   BreakPlanner: where to put the cue ball and how to break
    ball_in_hand.py where to put the cue ball when we have ball in hand
    simulated_table.py  fake camera + cue arm (no referee: the robot plays alone)
    robot.py        PoolRobot: break, then run out the rack
    play.py         watch it: uv run python -m mr147.pool.play --game nine --seed 0

Pool uses the same coordinates as snooker: metres, (0, 0) at the table centre,
x along the length (head end, where you break from, is negative x), y across.
The rules know nothing about positions: only which balls went where.

To plan a shot: PhysicsSimulator(POOL_TABLE) and ShotPlanner(...).plan_for(table, eight_ball_goal(game)).
"""

from mr147.pool.balls import CUE_BALL, EIGHT_BALL, NINE_BALL, Group
from mr147.pool.eight_ball import EightBallGame
from mr147.pool.nine_ball import NineBallGame
from mr147.pool.shot import ShotOutcome

__all__ = ["CUE_BALL", "EIGHT_BALL", "NINE_BALL", "EightBallGame", "Group", "NineBallGame", "ShotOutcome"]
