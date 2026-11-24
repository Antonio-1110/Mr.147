"""TableMemory: decide whether to believe vision, and remember what's been potted.

Every frame from vision is checked before we believe it:

  1. Each detection is sane: confident enough, on the table, not overlapping
     another ball, at most one of each colour (pool: one of each number), never
     more reds than can be left.
  2. Balls only ever leave the table. A ball that "appears" is a vision mistake.
  3. The balls that disappeared are exactly what the shot was meant to pot.
     A red that vanished unexpectedly is probably hidden, not potted.
  4. A colour that is respotted is back on its spot.

If any check fails, the frame is rejected with a ViewRequest saying why, and
the robot asks vision for a better look. Memory is only changed by frames that
pass (or when the caller explicitly decides to trust the camera).

A known blind spot: if the black is potted and respotted between two frames,
the camera sees no change. We count it as potted only if the shot was meant to
pot it and it is now sitting on its spot.

The same checks work for snooker and pool: TableMemory(balls=POOL_BALLS) for
pool (see ball_sets.py). Pool balls are named "1".."15" and "white".
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from mr147.interfaces.vision import DetectionFrame, ViewRequest
from mr147.snooker.balls import Colour
from mr147.snooker.table import Position, distance
from mr147.table_memory.ball_sets import SNOOKER_BALLS, BallSet
from mr147.table_memory.table_state import TableState


@dataclass(frozen=True)
class PotRecord:
    colour: Colour | str  # the ball's name (pool: its number)
    shot_number: int  # 0 = already gone before we first looked at the table


@dataclass(frozen=True)
class FrameCheck:
    """The verdict on one frame from vision."""

    accepted: bool
    table: TableState | None = None  # the table we now believe in (if accepted)
    potted: Counter = field(default_factory=Counter)  # colours potted since last frame
    view_request: ViewRequest | None = None  # what to look at again (if rejected)


@dataclass
class MemorySettings:
    min_confidence: float = 0.5
    off_table_slack: float = 0.01  # metres of measurement error we allow at the cushions
    duplicate_distance: float | None = None  # two balls this close = seen twice (default 0.85 ball widths)
    on_spot_tolerance: float = 0.015  # metres: how close a respotted colour must be


class TableMemory:
    def __init__(self, settings: MemorySettings | None = None, balls: BallSet = SNOOKER_BALLS):
        self.settings = settings or MemorySettings()
        self.balls = balls
        if self.settings.duplicate_distance is None:
            self.settings.duplicate_distance = 0.85 * balls.ball_diameter
        self.table: TableState | None = None  # the latest table we believe in
        self.history: list[TableState] = []
        self.pot_records: list[PotRecord] = []
        self.shots_played = 0

    # ---- questions ---------------------------------------------------------------

    def colours_on_table(self) -> Counter:
        return self.table.count_by_colour() if self.table else Counter()

    def in_play(self) -> list[str] | None:
        """Names of the balls that can still be on the table (with repeats), for vision.

        Balls only leave the table, except colours that get respotted. None before
        the first look, when anything is possible.
        """
        if self.table is None:
            return None
        counts = Counter({str(k): v for k, v in self.table.count_by_colour().items()})
        for name in self.balls.spots:  # respotted colours can come back
            if name != self.balls.many:
                counts[str(name)] = max(counts[str(name)], 1)
        return list(counts.elements())

    def number_potted(self, colour: Colour | str | None = None) -> int:
        """How many balls (of this colour, or in total) have been potted so far."""
        return sum(1 for record in self.pot_records if colour is None or record.colour == colour)

    # ---- updates -----------------------------------------------------------------

    def first_look(self, frame: DetectionFrame, *, trust_camera: bool = False) -> FrameCheck:
        """The first frame of the break. Any ball not seen counts as potted earlier."""
        reasons, look_closer = self._problems_with_detections(frame)
        if reasons and not trust_camera:
            return _reject(reasons, look_closer)

        already_gone = self.balls.full_set - _count_colours(frame)

        self._save(frame, already_gone, shot_number=0)
        return FrameCheck(accepted=True, table=self.table, potted=already_gone)

    def after_shot(
        self,
        frame: DetectionFrame,
        *,
        planned_pots: Iterable[Colour | str] | None,
        respotted_colours: Iterable[Colour | str] = (),
        trust_camera: bool = False,
    ) -> FrameCheck:
        """Check the frame taken after a shot.

        planned_pots:      colours the shot was meant to pot (None = don't check).
        respotted_colours: colours that get put back on their spot if potted.
        trust_camera:      save the frame even if checks fail. Use this when several
                           better views all agree: then the shot really did
                           something other than planned.
        """
        assert self.table is not None, "call first_look() before after_shot()"
        reasons, look_closer = self._problems_with_detections(frame)

        before = self.colours_on_table()
        now = _count_colours(frame)
        potted = before - now  # Counter subtraction keeps only the positive part

        # Balls only ever leave the table.
        appeared = now - before
        for colour, count in appeared.items():
            reasons.append(f"{count} more {colour} than can be on the table")
            look_closer += [d.position for d in frame.detections if d.colour == colour]

        # Respotted colours: the camera can't see they were potted, only that they're back.
        planned = Counter(planned_pots) if planned_pots is not None else None
        for colour in respotted_colours:
            if potted[colour]:
                reasons.append(f"{colour} was potted but is not back on its spot yet")
                look_closer.append(self.balls.spots[colour])
            elif planned and planned[colour] and self._is_on_its_spot(frame, colour):
                potted[colour] += 1

        if planned is not None and potted != planned:
            reasons.append(f"expected to pot {_describe(planned)}, but {_describe(potted)} went")
            surprises = (potted - planned) + (planned - potted)
            for colour in surprises:
                look_closer += [self.table.positions[i] for i in self.table.ball_ids(colour)]

        if reasons and not trust_camera:
            return _reject(reasons, look_closer, self.in_play())

        self.shots_played += 1
        self._save(frame, potted, shot_number=self.shots_played)
        return FrameCheck(accepted=True, table=self.table, potted=potted)

    # ---- helpers -----------------------------------------------------------------

    def _problems_with_detections(self, frame: DetectionFrame) -> tuple[list[str], list[Position]]:
        """Check each detection on its own. Returns (reasons, positions to look at)."""
        s = self.settings
        reasons: list[str] = []
        look_closer: list[Position] = []
        detections = frame.detections

        for d in detections:
            if d.confidence < s.min_confidence:
                reasons.append(f"not sure about the {d.colour} (confidence {d.confidence:.2f})")
                look_closer.append(d.position)
            if not self.balls.is_known(d.colour):
                reasons.append(f"'{d.colour}' is not a {self.balls.game} ball")
                look_closer.append(d.position)
            if not self.balls.is_on_table(d.position, slack=s.off_table_slack):
                reasons.append(f"{d.colour} seen off the table at ({d.x:.3f}, {d.y:.3f})")
                look_closer.append(d.position)

        for i, a in enumerate(detections):
            for b in detections[i + 1 :]:
                if distance(a.position, b.position) < s.duplicate_distance:
                    reasons.append(f"{a.colour} and {b.colour} overlap: one ball seen twice?")
                    look_closer.append(a.position)

        counts = _count_colours(frame)
        for colour, count in counts.items():
            limit = self.balls.most_allowed(colour)
            if self.balls.is_known(colour) and count > limit:
                reasons.append(f"{count} {colour} balls seen (there can be at most {limit})")
                look_closer += [d.position for d in detections if d.colour == colour]

        if counts[self.balls.cue_ball] == 0:
            reasons.append("cue ball not found")

        many = self.balls.many
        if many is not None:
            can_be_left = self.balls.full_set[many] - self.number_potted(many)
            if counts[many] > can_be_left:
                reasons.append(
                    f"{counts[many]} {many}s seen, but {self.number_potted(many)} "
                    f"are already potted so at most {can_be_left} can be left"
                )
        return reasons, look_closer

    def _is_on_its_spot(self, frame: DetectionFrame, colour: Colour | str) -> bool:
        spot = self.balls.spots.get(colour)
        return spot is not None and any(
            d.colour == colour and distance(d.position, spot) <= self.settings.on_spot_tolerance
            for d in frame.detections
        )

    def _save(self, frame: DetectionFrame, potted: Counter, shot_number: int) -> None:
        self.table = TableState.from_detections(frame.detections, many=self.balls.many)
        self.history.append(self.table)
        for colour, count in potted.items():
            self.pot_records += [PotRecord(colour, shot_number)] * count


def _reject(reasons: list[str], look_closer: list[Position], in_play: list[str] | None = None) -> FrameCheck:
    request = ViewRequest(tuple(reasons), tuple(look_closer), tuple(in_play) if in_play is not None else None)
    return FrameCheck(accepted=False, view_request=request)


def _count_colours(frame: DetectionFrame) -> Counter:
    return Counter(d.colour for d in frame.detections)


def _describe(counts: Counter) -> str:
    """Counter({RED: 1}) -> "1 red"."""
    return ", ".join(f"{n} {colour}" for colour, n in counts.items() if n) or "nothing"
