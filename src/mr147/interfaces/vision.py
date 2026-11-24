"""The contract with the vision team.

Vision sends a DetectionFrame: every ball it can see, in table coordinates (see
mr147.snooker.table / mr147.pool.table), taken once the balls have stopped moving.

What a ball is called depends on the game:
    snooker  its colour: "red", "yellow", ..., "black", "white" (see mr147.snooker.balls)
    pool     its number: "1" .. "15", and "white" for the cue ball (see mr147.pool.balls)

If a frame doesn't make sense, we send back a ViewRequest saying what looks
wrong and where to look. Vision moves the camera and sends a new frame.

Who checks what:
    vision  drops anything that isn't on our table (e.g. a ball on the next table),
            never uses a name more often than the game allows, and avoids names of
            balls the memory says are potted (`in_play`).
    memory  the final check, using what only it knows (what was potted, what the
            shot was meant to do). It asks for a better view only when there is
            real doubt on our table, the one time walking round can help.
"""

from collections.abc import Sequence

from dataclasses import dataclass
from typing import Protocol

from mr147.snooker.balls import Colour
from mr147.snooker.table import Position


@dataclass(frozen=True)
class Detection:
    """One ball seen on the table.

    `colour` is the ball's name: a snooker Colour, or a pool ball's number as text
    ("1".."15", "white"). Colour is a str, so both compare as plain text.
    """

    colour: Colour | str
    x: float
    y: float
    confidence: float = 1.0  # 0..1, how sure vision is that this ball is this one
    # Other names this ball might have, with vision's score for each (best first).
    # Lets the reader double-check, e.g. "8, but maybe the 4".
    alternatives: tuple[tuple[str, float], ...] = ()

    @property
    def position(self) -> Position:
        return (self.x, self.y)


@dataclass(frozen=True)
class DetectionFrame:
    """Every ball vision can see right now."""

    detections: tuple[Detection, ...]
    timestamp: float = 0.0
    camera_view: str | None = None  # which camera pose produced this (for debugging)
    game: str = "snooker"  # "snooker" or "pool": says how the balls are named


@dataclass(frozen=True)
class ViewRequest:
    """'Please look again': sent when a frame doesn't add up.

    reasons:     human-readable list of what looks wrong
    look_closer: table positions worth a better look (e.g. where a ball vanished)
    """

    reasons: tuple[str, ...]
    look_closer: tuple[Position, ...] = ()
    in_play: tuple[str, ...] | None = None  # balls still on the table, with repeats ("red" x7)


class VisionSystem(Protocol):
    """Anything that can tell us where the balls are."""

    def observe(self, in_play: Sequence[str] | None = None) -> DetectionFrame:
        """Detect every ball from the current camera position.

        in_play: the balls the memory believes are still on the table (names, with
        repeats: "red" seven times). None at the start, when anything is possible.
        """
        ...

    def request_better_view(self, request: ViewRequest) -> DetectionFrame:
        """Move the camera (or try harder) and detect again."""
        ...
