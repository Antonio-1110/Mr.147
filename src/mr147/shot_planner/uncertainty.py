"""How imprecise the hardware is.

The planner replays each good-looking shot many times with these errors added,
to see how often it would still go in. Set them to match the real hardware.

Snooker is unforgiving. Over a metre, 0.1 degrees of aim error or 1-2 mm of
error in where vision thinks the object ball is can turn a pot into a miss.
"""

import random
from dataclasses import dataclass, field, replace

from mr147.interfaces.cue_arm import Shot
from mr147.table_memory.table_state import TableState


@dataclass(frozen=True)
class CueArmError:
    """How far the cue arm is off from what it's asked to do (1 standard deviation)."""

    direction_deg: float = 0.08
    speed_fraction: float = 0.03  # 0.03 = 3% of the requested speed
    tip_offset: float = 0.02  # side / height error, in ball radii

    def add_to(self, shot: Shot, rng: random.Random) -> Shot:
        """The shot the arm would actually play when asked for `shot`."""
        return replace(
            shot,
            direction_deg=shot.direction_deg + rng.gauss(0, self.direction_deg),
            speed=shot.speed * (1 + rng.gauss(0, self.speed_fraction)),
            side=shot.side + rng.gauss(0, self.tip_offset),
            height=shot.height + rng.gauss(0, self.tip_offset),
        )


@dataclass(frozen=True)
class Uncertainty:
    cue_arm: CueArmError = field(default_factory=CueArmError)
    camera_error_m: float = 0.001  # how far off vision's ball positions are (1 sd, metres)

    def add_camera_error(self, table: TableState, rng: random.Random) -> TableState:
        """A table where every ball is where vision might wrongly have put it."""
        if self.camera_error_m == 0:
            return table
        sd = self.camera_error_m
        return TableState(
            {ball_id: (x + rng.gauss(0, sd), y + rng.gauss(0, sd)) for ball_id, (x, y) in table.positions.items()}
        )
