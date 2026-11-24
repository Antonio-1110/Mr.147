"""A fake snooker table that acts as camera, cue arm and referee.

Implements both VisionSystem and CueArm, so the robot can't tell it apart from
the real hardware.

  * Cue arm: plays each shot with random error (CueArmError).
  * Camera:  reports every ball a few mm off, and now and then misses one
             (as if the arm were in the way). A better view never misses.
  * Referee: puts colours back on their spots the way snooker does.
"""

import random

from mr147.interfaces.cue_arm import Shot
from mr147.interfaces.vision import Detection, DetectionFrame, ViewRequest
from mr147.physics import PhysicsSimulator
from mr147.shot_planner.uncertainty import CueArmError
from mr147.snooker.balls import CUE_BALL, Colour, colour_of
from mr147.snooker.table import SPOTS
from mr147.table_memory.table_state import TableState


class SimulatedTable:
    def __init__(
        self,
        starting_table: TableState,
        physics: PhysicsSimulator,
        *,
        cue_arm_error: CueArmError | None = CueArmError(),
        camera_error_m: float = 0.001,
        chance_camera_misses_a_ball: float = 0.0,
        random_seed: int = 0,
    ):
        self.true_table = starting_table  # where the balls really are
        self.physics = physics
        self.cue_arm_error = cue_arm_error
        self.camera_error_m = camera_error_m
        self.chance_camera_misses_a_ball = chance_camera_misses_a_ball
        self.rng = random.Random(random_seed)
        self.last_shot_potted_a_red = False
        self.last_result = None  # the simulator's ShotResult for the latest shot
        self.clock = 0.0

    # ---- VisionSystem ----------------------------------------------------------------

    def observe(self, in_play=None) -> DetectionFrame:
        return self._take_picture(self.chance_camera_misses_a_ball, camera_view="main")

    def request_better_view(self, request: ViewRequest) -> DetectionFrame:
        return self._take_picture(0.0, camera_view="closer look")

    # ---- CueArm ----------------------------------------------------------------------

    def play(self, shot: Shot) -> None:
        if self.cue_arm_error is not None:
            shot = self.cue_arm_error.add_to(shot, self.rng)
        result = self.physics.simulate(self.true_table, shot)
        self.last_result = result
        self.true_table = self._referee_respots(result.final_table, result.potted_ball_ids)
        self.clock += 1.0

    # ---- helpers ---------------------------------------------------------------------

    def _referee_respots(self, table: TableState, potted_ball_ids: tuple[str, ...]) -> TableState:
        """Colours come back while reds are left, and right after the last red."""
        potted_colours = [colour_of(ball_id) for ball_id in potted_ball_ids]
        reds_left = bool(table.ball_ids(Colour.RED))
        for colour in potted_colours:
            is_a_colour = colour not in (Colour.RED, CUE_BALL)
            if is_a_colour and (reds_left or self.last_shot_potted_a_red):
                table = table.with_ball_at(colour.value, SPOTS[colour])
        self.last_shot_potted_a_red = Colour.RED in potted_colours
        return table

    def _take_picture(self, chance_to_miss: float, camera_view: str) -> DetectionFrame:
        detections = []
        for ball_id, (x, y) in self.true_table.positions.items():
            if self.rng.random() < chance_to_miss:
                continue
            seen_x = x + self.rng.gauss(0, self.camera_error_m)
            seen_y = y + self.rng.gauss(0, self.camera_error_m)
            detections.append(Detection(colour_of(ball_id), seen_x, seen_y, confidence=0.95))
        return DetectionFrame(tuple(detections), timestamp=self.clock, camera_view=camera_view)
