"""A fake pool table that acts as camera and cue arm, so the pool robot runs on a laptop.

  * Cue arm: plays each shot with random error, and can place a ball by hand
             (the cue ball with ball in hand, or the 8 if it drops on the break).
  * Camera:  reports every ball a little off, named the pool way ("1".."15", "white").

No referee: the robot plays alone and the run ends at the first miss or foul.
"""

import random
from collections.abc import Sequence

from mr147.interfaces.cue_arm import Shot
from mr147.interfaces.vision import Detection, DetectionFrame, ViewRequest
from mr147.physics import PhysicsSimulator
from mr147.pool.table import Position
from mr147.shot_planner.uncertainty import CueArmError
from mr147.table_memory.table_state import TableState


class SimulatedPoolTable:
    def __init__(
        self,
        starting_table: TableState,
        physics: PhysicsSimulator,
        *,
        cue_arm_error: CueArmError | None = CueArmError(),
        camera_error_m: float = 0.001,
        random_seed: int = 0,
    ):
        self.true_table = starting_table
        self.physics = physics
        self.cue_arm_error = cue_arm_error
        self.camera_error_m = camera_error_m
        self.rng = random.Random(random_seed)
        self.shots_played = 0
        self.last_result = None

    # ---- VisionSystem ----------------------------------------------------------------

    def observe(self, in_play: Sequence[str] | None = None) -> DetectionFrame:
        return self._take_picture("main")

    def request_better_view(self, request: ViewRequest) -> DetectionFrame:
        return self._take_picture("closer look")

    # ---- CueArm, plus placing the cue ball ---------------------------------------------

    def place_ball(self, ball: str, position: Position) -> None:
        """Put a ball here by hand: the cue ball with ball in hand, or a ball that is respotted."""
        self.true_table = self.true_table.with_ball_at(ball, position)

    def play(self, shot: Shot) -> None:
        if self.cue_arm_error is not None:
            shot = self.cue_arm_error.add_to(shot, self.rng)
        result = self.physics.simulate(self.true_table, shot)
        self.last_result = result
        self.true_table = result.final_table
        self.shots_played += 1

    # ---- helpers ---------------------------------------------------------------------

    def _take_picture(self, camera_view: str) -> DetectionFrame:
        detections = []
        for name, (x, y) in self.true_table.positions.items():
            seen = (x + self.rng.gauss(0, self.camera_error_m), y + self.rng.gauss(0, self.camera_error_m))
            detections.append(Detection(name, *seen, confidence=0.95))
        return DetectionFrame(tuple(detections), timestamp=float(self.shots_played), camera_view=camera_view, game="pool")
