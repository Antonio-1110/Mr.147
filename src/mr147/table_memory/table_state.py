"""TableState: a snapshot of where every ball on the table is."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from mr147.interfaces.vision import Detection
from mr147.snooker.balls import CUE_BALL, Colour, make_red_id
from mr147.snooker.table import Position


@dataclass(frozen=True)
class TableState:
    """Ball id -> (x, y) for every ball still on the table.

    Ball ids are explained in mr147.snooker.balls ("red_01", "black", "white", ...)
    and mr147.pool.balls ("1".."15", "white").
    """

    positions: Mapping[str, Position]

    @classmethod
    def from_detections(cls, detections: Iterable[Detection], many: str | None = Colour.RED) -> "TableState":
        """Build a snapshot from what vision saw. Reds are given ids in a fixed order.

        many: the name several balls share (snooker "red"); None for pool, where
        every ball's name is already its id.
        """
        detections = list(detections)
        red_positions = sorted(d.position for d in detections if many is not None and d.colour == many)
        positions = {make_red_id(i): p for i, p in enumerate(red_positions)}
        for d in detections:
            if many is None or d.colour != many:
                positions[str(d.colour)] = d.position
        return cls(positions)

    @property
    def cue_ball(self) -> Position:
        return self.positions[CUE_BALL.value]

    def ball_ids(self, colour: Colour | str) -> list[str]:
        """Ids of every ball of this colour (pool: this number) on the table."""
        return [ball_id for ball_id in self.positions if name_of(ball_id) == colour]

    def count_by_colour(self) -> Counter:
        """How many balls of each colour (pool: each number) are on the table."""
        return Counter(name_of(ball_id) for ball_id in self.positions)

    def positions_except(self, *ball_ids: str) -> list[Position]:
        """Every ball's position except the given ones (the obstacles for a shot)."""
        return [p for ball_id, p in self.positions.items() if ball_id not in ball_ids]

    def with_ball_at(self, ball_id: str, position: Position) -> "TableState":
        """A copy with one ball added or moved."""
        return TableState({**self.positions, ball_id: position})


def name_of(ball_id: str) -> Colour | str:
    """Ball id -> the name vision uses: "red_07" -> Colour.RED, "black" -> Colour.BLACK, "9" -> "9"."""
    if ball_id.startswith("red"):
        return Colour.RED
    try:
        return Colour(ball_id)
    except ValueError:
        return ball_id
