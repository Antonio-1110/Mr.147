"""Ball colours, values and ids.

Vision only reports colours ("red", "black", "white", ...). Inside our own copy
of the table every ball also gets an id, so the physics simulator can tell the
15 reds apart:

    "red_01" .. "red_15"   for reds
    "yellow", "black", ... for colours (the colour name is the id)
    "white"                for the cue ball

Red ids are handed out fresh each time we look at the table, so "red_03" only
means something within one snapshot.
"""

from enum import StrEnum


class Colour(StrEnum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    BROWN = "brown"
    BLUE = "blue"
    PINK = "pink"
    BLACK = "black"
    WHITE = "white"  # the cue ball


CUE_BALL = Colour.WHITE
NUMBER_OF_REDS = 15

# The order the colours are potted in once all the reds are gone.
COLOURS_IN_CLEARANCE_ORDER = (
    Colour.YELLOW,
    Colour.GREEN,
    Colour.BROWN,
    Colour.BLUE,
    Colour.PINK,
    Colour.BLACK,
)

POINTS = {
    Colour.RED: 1,
    Colour.YELLOW: 2,
    Colour.GREEN: 3,
    Colour.BROWN: 4,
    Colour.BLUE: 5,
    Colour.PINK: 6,
    Colour.BLACK: 7,
}


def make_red_id(index: int) -> str:
    """Id for the red at position `index` (0-based) in a snapshot: 0 -> "red_01"."""
    return f"red_{index + 1:02d}"


def colour_of(ball_id: str) -> Colour:
    """"red_07" -> Colour.RED, "black" -> Colour.BLACK."""
    if ball_id.startswith("red"):
        return Colour.RED
    return Colour(ball_id)
