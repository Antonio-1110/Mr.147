"""The contract with the cue arm: what a shot is, and how to ask for one."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Shot:
    """How the cue strikes the cue ball. Everything is in table coordinates.

    direction_deg: which way the cue ball is sent, in degrees.
                   0 = towards the black end (+x), 90 = towards +y.
    speed:         cue speed at impact, metres per second.
    side:          tip offset left/right of centre, in ball radii (+ = right).
    height:        tip offset above/below centre, in ball radii
                   (+ = top spin / follow, - = back spin / screw).
                   Keep side and height within about +-0.5.
    elevation_deg: how far the cue is tilted up from level.
    """

    direction_deg: float
    speed: float
    side: float = 0.0
    height: float = 0.0
    elevation_deg: float = 0.0


class CueArm(Protocol):
    """Anything that can play a shot."""

    def play(self, shot: Shot) -> None:
        """Play the shot and return once every ball has stopped."""
        ...
