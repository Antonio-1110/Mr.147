"""What the shot chooser and the simulator need to know about a table, for any game.

Each game folder fills one of these in from its own numbers:

    mr147.snooker.table.SNOOKER_TABLE
    mr147.pool.table.POOL_TABLE

The shot chooser and the simulator only ever read a GameTable, so a new game
(or a different table size) is just a new GameTable.
"""

from dataclasses import dataclass

Position = tuple[float, float]


@dataclass(frozen=True)
class GameTable:
    name: str
    length: float  # playing area between the cushion faces, metres
    width: float
    ball_diameter: float
    corner_pocket_opening: float  # gap between the cushion jaws at the pocket mouth
    middle_pocket_opening: float
    simulator_style: str  # "snooker" or "pool": which table and ball model the simulator uses
    cue_ball: str = "white"  # the cue ball's id in a TableState

    @property
    def ball_radius(self) -> float:
        return self.ball_diameter / 2

    def is_on_table(self, position: Position, slack: float = 0.0) -> bool:
        """Could a ball centre be at this position? `slack` allows for measurement error."""
        x, y = position
        return (
            abs(x) <= self.length / 2 - self.ball_radius + slack
            and abs(y) <= self.width / 2 - self.ball_radius + slack
        )
