"""Watch the robot break and run out a pool rack on the simulated table.

    uv run python -m mr147.pool.play --game nine --seed 0
    uv run python -m mr147.pool.play --game eight --seed 3 --perfect-hardware
    uv run python -m mr147.pool.play --racks 10          # summary over 10 racks
    uv run python -m mr147.pool.play --seed 0 --show-plans outputs/pool-plans.html
                                    # a web page with every shot considered, rack by rack
"""

import argparse
import logging
import random
import time
from pathlib import Path

from mr147.physics import PhysicsSimulator
from mr147.pool.break_shot import BreakPlanner
from mr147.pool.layouts import eight_ball_rack, nine_ball_rack
from mr147.pool.robot import PoolRobot
from mr147.pool.simulated_table import SimulatedPoolTable
from mr147.pool.table import POOL_TABLE
from mr147.shot_planner import CueArmError, PlannerSettings, ShotPlanner, Uncertainty


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", choices=("nine", "eight"), default="nine")
    parser.add_argument("--seed", type=int, default=0, help="which rack to play (first one, with --racks)")
    parser.add_argument("--racks", type=int, default=1, help="play this many racks and print a summary")
    parser.add_argument("--aim-error-deg", type=float, default=0.03, help="cue arm aim error (1 sd)")
    parser.add_argument("--speed-error-pct", type=float, default=2.0, help="cue arm speed error (1 sd)")
    parser.add_argument("--camera-error-mm", type=float, default=0.5, help="ball position error (1 sd)")
    parser.add_argument("--perfect-hardware", action="store_true", help="no arm or camera error at all")
    parser.add_argument("--quick", action="store_true", help="smaller search: faster but weaker")
    parser.add_argument("--show-plans", type=Path,
                        help="write a web page of every shot considered (.html; one page per rack with --racks)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.racks == 1 else logging.WARNING, format="%(message)s")

    if args.perfect_hardware:
        arm, camera_m = None, 0.0
    else:
        arm = CueArmError(direction_deg=args.aim_error_deg, speed_fraction=args.speed_error_pct / 100, tip_offset=0.01)
        camera_m = args.camera_error_mm / 1000

    physics = PhysicsSimulator(POOL_TABLE)
    rack = nine_ball_rack if args.game == "nine" else eight_ball_rack
    results = []
    for seed in range(args.seed, args.seed + args.racks):
        settings = PlannerSettings(
            uncertainty=Uncertainty(cue_arm=arm or CueArmError(0, 0, 0), camera_error_m=camera_m), random_seed=seed
        )
        if args.quick:
            settings.pot_options_to_try, settings.speeds, settings.heights = 3, (1.0, 2.0), (-0.4, 0.0, 0.4)
            settings.stress_test_replays = 8
        table = SimulatedPoolTable(
            rack(random.Random(seed)), physics, cue_arm_error=arm, camera_error_m=camera_m, random_seed=seed
        )
        planner, cue_arm, recorder = ShotPlanner(physics, settings), table, None
        if args.show_plans:
            from mr147.shot_planner.plan_viewer import PlanRecorder

            name = "9-ball" if args.game == "nine" else "8-ball"
            recorder = PlanRecorder(physics, f"{name}, rack {seed}")
            planner, cue_arm = recorder.watch_planner(planner), recorder.watch_cue_arm(table)
        robot = PoolRobot(table, cue_arm, table, planner, args.game, BreakPlanner(physics, args.game))
        started = time.time()
        try:
            report = robot.run_rack(rng=random.Random(seed))
        finally:
            if recorder is not None:
                page = args.show_plans
                if args.racks > 1:
                    page = page.with_name(f"{page.stem}-rack{seed}{page.suffix or '.html'}")
                print(f"shot plans: {recorder.write(page)}")
        results.append(report)
        print(
            f"rack {seed}: {'CLEARED' if report.won else 'stopped'}  {report.balls_potted} balls in "
            f"{report.shots} shots ({report.note}, {time.time() - started:.0f}s)"
        )
    if args.racks > 1:
        cleared = sum(r.won for r in results)
        print(f"\n{cleared} of {len(results)} racks cleared, {sum(r.balls_potted for r in results)} balls potted")


if __name__ == "__main__":
    main()
