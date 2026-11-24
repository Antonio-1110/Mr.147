"""Pool balls are known by number: 0 is the cue ball, then 1 to 15.

    1-7   solids (one colour all over)
    8     the black 8 ball
    9-15  stripes (white with a band of the colour of n - 8)

Vision labels them "cue", "1", ..., "15" (see from_vision_label). Inside a
TableState they are "white", "1", ..., "15" (see ball_id and number_of).
"""

from enum import Enum

CUE_BALL = 0
EIGHT_BALL = 8
NINE_BALL = 9
ALL_OBJECT_BALLS = frozenset(range(1, 16))

COLOUR_NAMES = {
    0: "white",
    1: "yellow",
    2: "blue",
    3: "red",
    4: "purple",
    5: "orange",
    6: "green",
    7: "maroon",
    8: "black",
}


class Group(Enum):
    """In 8-ball each player owns one group."""

    SOLIDS = frozenset(range(1, 8))
    STRIPES = frozenset(range(9, 16))

    @property
    def balls(self) -> frozenset[int]:
        return self.value

    @property
    def other(self) -> "Group":
        return Group.STRIPES if self is Group.SOLIDS else Group.SOLIDS


def group_of(ball: int) -> Group | None:
    """1-7 -> SOLIDS, 9-15 -> STRIPES, the cue ball and the 8 -> None."""
    for group in Group:
        if ball in group.balls:
            return group
    return None


def describe(ball: int) -> str:
    """0 -> "cue ball", 3 -> "3 (red)", 11 -> "11 (red stripe)"."""
    if ball == CUE_BALL:
        return "cue ball"
    if ball <= 8:
        return f"{ball} ({COLOUR_NAMES[ball]})"
    return f"{ball} ({COLOUR_NAMES[ball - 8]} stripe)"


def ball_id(number: int) -> str:
    """Ball number -> id in a TableState: 0 -> "white" (every game's cue ball id), 9 -> "9"."""
    return "white" if number == CUE_BALL else str(number)


def number_of(ball_id: str) -> int:
    """Id in a TableState -> ball number: "white" -> 0, "9" -> 9."""
    return CUE_BALL if ball_id == "white" else int(ball_id)


def from_vision_label(label: str) -> int:
    """Vision's label name -> ball number: "cue" -> 0, "9" -> 9."""
    return CUE_BALL if label == "cue" else int(label)
