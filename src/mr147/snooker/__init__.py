"""Everything snooker-specific.

    balls.py            ball colours, values and ids
    table.py            table size, pockets, spots (SNOOKER_TABLE for the simulator)
    rules.py            the 147 sequence: which ball is on, what's next, respots
    shot_goal.py        tells the shot planner what a good snooker shot is
    layouts.py          open tables, a full rack, tied-up tables after a break-off
    simulated_table.py  fake camera + cue arm + referee
    robot.py            SnookerRobot: look -> plan -> shoot -> check
    play.py             watch it: uv run python -m mr147.snooker.play --seed 0

Shared by every game (one level up): physics/, shot_planner/, table_memory/,
interfaces/, vision/, game_table.py.
"""
