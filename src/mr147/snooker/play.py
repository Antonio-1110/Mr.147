"""Watch the robot play a snooker break on the simulated table.

    uv run python -m mr147.snooker.play --seed 0
    uv run python -m mr147.snooker.play --seed 0 --perfect-hardware
    uv run python -m mr147.snooker.play --seed 0 --aim-error-deg 0.02 --camera-error-mm 0.25

See the robot's eyes at work, and a picture of every planned shot:

    uv run python -m mr147.snooker.play --seed 0 --camera rendered --pictures outputs/break0
    uv run python -m mr147.snooker.play --seed 0 --camera models/snooker.pt --pictures outputs/break0

See every shot the planner considered, on one web page (no 3D needed):

    uv run python -m mr147.snooker.play --seed 0 --show-plans outputs/snooker-plans.html

--camera simple    (default) balls reported straight from the simulator, a few mm off
--camera rendered  the robot's head camera renders the table, a perfect detector finds the
                   boxes, and the vision code works out where the balls are
--camera FILE.pt   the same, but your trained YOLO model finds the boxes
"""

import argparse
import logging
import random
import time
from pathlib import Path

from mr147.physics import PhysicsSimulator
from mr147.snooker.robot import SnookerRobot
from mr147.shot_planner import CueArmError, PlannerSettings, ShotPlanner, Uncertainty
from mr147.snooker.layouts import open_table
from mr147.snooker.simulated_table import SimulatedTable


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="which random layout to play")
    parser.add_argument("--reds", type=int, default=15)
    parser.add_argument("--aim-error-deg", type=float, default=0.08, help="cue arm aim error (1 sd)")
    parser.add_argument("--speed-error-pct", type=float, default=3.0, help="cue arm speed error (1 sd)")
    parser.add_argument("--camera-error-mm", type=float, default=1.0, help="ball position error (1 sd)")
    parser.add_argument("--camera-miss-chance", type=float, default=0.0, help="chance a ball goes unseen")
    parser.add_argument("--perfect-hardware", action="store_true", help="no arm or camera error at all")
    parser.add_argument("--quick", action="store_true", help="smaller search: faster but weaker")
    parser.add_argument("--max-shots", type=int, default=40)
    parser.add_argument("--camera", default="simple", help="simple, rendered, or a YOLO model file (.pt)")
    parser.add_argument("--pictures", type=Path, help="save a picture of every planned shot here")
    parser.add_argument("--show-plans", type=Path, help="write a web page of every shot considered (.html)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.perfect_hardware:
        cue_arm_error, camera_error_m = None, 0.0
    else:
        cue_arm_error = CueArmError(direction_deg=args.aim_error_deg, speed_fraction=args.speed_error_pct / 100)
        camera_error_m = args.camera_error_mm / 1000

    # The planner is told how precise the hardware really is.
    uncertainty = Uncertainty(cue_arm=cue_arm_error or CueArmError(0, 0, 0), camera_error_m=camera_error_m)
    settings = PlannerSettings(uncertainty=uncertainty, random_seed=args.seed)
    if args.quick:
        settings.pot_options_to_try = 3
        settings.speeds = (1.0, 2.0)
        settings.heights = (-0.4, 0.0, 0.4)
        settings.stress_test_replays = 8

    physics = PhysicsSimulator()
    table = SimulatedTable(
        open_table(random.Random(args.seed), args.reds),
        physics,
        cue_arm_error=cue_arm_error,
        camera_error_m=camera_error_m,
        chance_camera_misses_a_ball=args.camera_miss_chance,
        random_seed=args.seed,
    )
    vision = table
    renderer = None
    if args.camera != "simple" or args.pictures:
        from mr147.vision.scene import Renderer

        renderer = Renderer(1280, 720, seed=args.seed)
    if args.camera != "simple":
        vision = robot_eyes(args.camera, renderer, table, args.seed)

    on_plan = None
    if args.pictures:
        import cv2

        from mr147.vision.draw_plan import draw_plan

        args.pictures.mkdir(parents=True, exist_ok=True)

        def on_plan(table_before, plan):
            n = len(list(args.pictures.glob("shot*.jpg"))) + 1
            picture = draw_plan(renderer, table_before, plan, physics.pockets)
            cv2.imwrite(str(args.pictures / f"shot{n:02d}.jpg"), picture[..., ::-1])

    planner, cue_arm, recorder = ShotPlanner(physics, settings), table, None
    if args.show_plans:
        from mr147.shot_planner.plan_viewer import PlanRecorder

        recorder = PlanRecorder(physics, f"Snooker, layout {args.seed}")
        planner, cue_arm = recorder.watch_planner(planner), recorder.watch_cue_arm(table)
    robot = SnookerRobot(vision=vision, cue_arm=cue_arm, planner=planner, on_plan=on_plan)

    started = time.time()
    try:
        score = robot.play_break(args.max_shots)
        print(f"\nbreak: {score}  ({robot.memory.shots_played} shots, {time.time() - started:.0f}s)")
    finally:
        if recorder is not None:
            print(f"shot plans: {recorder.write(args.show_plans)}")


def robot_eyes(camera: str, renderer, table: SimulatedTable, seed: int):
    """The real vision chain, looking at the simulated table through a rendered camera."""
    from mr147.vision.camera_vision import CameraVision
    from mr147.vision.scene import CameraPose
    from mr147.vision.sim_camera import PerfectDetector, SimCamera

    start = CameraPose((-2.3, -1.4, 0.75), (0.0, 0.0, 0.0), 65.0)  # standing at a baulk corner
    sim_camera = SimCamera(renderer, lambda: table.true_table, start, seed=seed)
    if camera == "rendered":
        detector = PerfectDetector(sim_camera, seed=seed)
    else:
        from mr147.vision.detector import YoloDetector

        detector = YoloDetector(camera)
    return CameraVision(sim_camera, detector)


if __name__ == "__main__":
    main()
