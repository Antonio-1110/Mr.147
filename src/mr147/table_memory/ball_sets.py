"""Which balls a game starts with, and what can never be true about them.

The table memory uses this to sanity-check what vision reports:

    snooker  15 reds, 6 colours, the white. A colour that is potted is respotted.
    pool     balls 1 to 15 and the white. Every ball is unique, nothing is respotted
             (the rare 8-ball/9-ball respots aren't modelled).

Ball names are the ones in the vision contract (mr147.interfaces.vision).
"""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from mr147.pool import table as pool_table
from mr147.snooker import table as snooker_table
from mr147.snooker.balls import NUMBER_OF_REDS, Colour

Position = tuple[float, float]


@dataclass(frozen=True)
class BallSet:
    game: str
    full_set: Counter  # every object ball at the start (not the cue ball)
    ball_diameter: float
    is_on_table: Callable[..., bool]  # (position, slack=...) -> bool
    cue_ball: str = "white"
    many: str | None = None  # the one name more than one ball can share ("red" in snooker)
    spots: Mapping[str, Position] = field(default_factory=dict)  # where respotted balls go

    def most_allowed(self, name: str) -> int:
        """How many balls with this name can be on the table at once."""
        if name == self.cue_ball:
            return 1
        return self.full_set.get(name, 0)

    def is_known(self, name: str) -> bool:
        return name == self.cue_ball or name in self.full_set


SNOOKER_BALLS = BallSet(
    game="snooker",
    full_set=Counter({Colour.RED: NUMBER_OF_REDS} | {c: 1 for c in Colour if c not in (Colour.RED, Colour.WHITE)}),
    ball_diameter=snooker_table.BALL_DIAMETER,
    is_on_table=snooker_table.is_on_table,
    many=Colour.RED,
    spots=snooker_table.SPOTS,
)

POOL_BALLS = BallSet(
    game="pool",
    full_set=Counter({str(n): 1 for n in range(1, 16)}),
    ball_diameter=pool_table.BALL_DIAMETER,
    is_on_table=pool_table.is_on_table,
)

BALL_SETS = {b.game: b for b in (SNOOKER_BALLS, POOL_BALLS)}
