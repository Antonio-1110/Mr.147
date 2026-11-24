"""The 147 sequence.

    red, black, red, black, ... (15 times)  -> 15 x (1 + 7) = 120
    yellow, green, brown, blue, pink, black -> 2 + 3 + 4 + 5 + 6 + 7 = 27
                                                                total 147

While reds are left (and for the black right after the last red) a potted
black is put back on its spot. There are no fouls, safeties or opponent here:
the robot pots exactly the ball on every time, or the break is over.

This module knows nothing about ball positions; only the order of play.
"""

from collections import Counter
from dataclasses import dataclass
from enum import Enum, auto

from mr147.snooker.balls import COLOURS_IN_CLEARANCE_ORDER, NUMBER_OF_REDS, POINTS, Colour


class BreakPhase(Enum):
    RED_ON = auto()  # must pot a red
    BLACK_ON = auto()  # just potted a red: must pot the black (it gets respotted)
    CLEARING_COLOURS = auto()  # reds gone: pot the colours in order
    FINISHED = auto()  # 147!
    FAILED = auto()  # potted the wrong thing, or nothing


@dataclass
class PerfectBreak:
    reds_left: int = NUMBER_OF_REDS
    phase: BreakPhase = BreakPhase.RED_ON
    colours_cleared: int = 0  # how many of yellow..black are gone (in CLEARING_COLOURS)
    score: int = 0
    why_failed: str | None = None

    @classmethod
    def starting_with(cls, reds_on_table: int) -> "PerfectBreak":
        """Start a break with this many reds on the table (15 for a real 147)."""
        if reds_on_table == 0:
            return cls(reds_left=0, phase=BreakPhase.CLEARING_COLOURS)
        return cls(reds_left=reds_on_table)

    # ---- questions ---------------------------------------------------------------

    @property
    def is_over(self) -> bool:
        return self.phase in (BreakPhase.FINISHED, BreakPhase.FAILED)

    def ball_on(self) -> Colour | None:
        """The colour the robot must pot with the next shot (None once the break is over)."""
        if self.phase == BreakPhase.RED_ON:
            return Colour.RED
        if self.phase == BreakPhase.BLACK_ON:
            return Colour.BLACK
        if self.phase == BreakPhase.CLEARING_COLOURS:
            return COLOURS_IN_CLEARANCE_ORDER[self.colours_cleared]
        return None

    def ball_on_after_this_one(self) -> Colour | None:
        """The ball on once the current one is potted: what to play position for."""
        return self._after_potting_ball_on().ball_on()

    def colours_respotted_if_potted(self) -> set[Colour]:
        """Colours that go back on their spot if the next shot pots them."""
        if self.phase == BreakPhase.BLACK_ON:
            return {Colour.BLACK}
        return set()

    def best_possible_score(self) -> int:
        if self.phase == BreakPhase.FAILED:
            return self.score
        reds_and_blacks = self.reds_left * (POINTS[Colour.RED] + POINTS[Colour.BLACK])
        black_now = POINTS[Colour.BLACK] if self.phase == BreakPhase.BLACK_ON else 0
        colours = sum(POINTS[c] for c in COLOURS_IN_CLEARANCE_ORDER[self.colours_cleared :])
        return self.score + reds_and_blacks + black_now + colours

    # ---- updates -----------------------------------------------------------------

    def record_shot(self, potted: Counter) -> None:
        """Update the break after a shot, given the colours it potted."""
        if self.is_over:
            raise RuntimeError(f"the break is already over ({self.phase.name})")

        ball_on = self.ball_on()
        if potted != Counter({ball_on: 1}):
            what_went = ", ".join(f"{n} {c}" for c, n in potted.items() if n) or "nothing"
            self.phase = BreakPhase.FAILED
            self.why_failed = f"needed exactly one {ball_on}, potted {what_went}"
            return

        nxt = self._after_potting_ball_on()
        self.reds_left = nxt.reds_left
        self.phase = nxt.phase
        self.colours_cleared = nxt.colours_cleared
        self.score = nxt.score

    def _after_potting_ball_on(self) -> "PerfectBreak":
        """What the break looks like after the ball on is potted (self is unchanged)."""
        ball_on = self.ball_on()
        if ball_on is None:
            return self
        score = self.score + POINTS[ball_on]

        if self.phase == BreakPhase.RED_ON:
            return PerfectBreak(reds_left=self.reds_left - 1, phase=BreakPhase.BLACK_ON, score=score)

        if self.phase == BreakPhase.BLACK_ON:
            next_phase = BreakPhase.RED_ON if self.reds_left > 0 else BreakPhase.CLEARING_COLOURS
            return PerfectBreak(reds_left=self.reds_left, phase=next_phase, score=score)

        # CLEARING_COLOURS
        cleared = self.colours_cleared + 1
        all_gone = cleared == len(COLOURS_IN_CLEARANCE_ORDER)
        return PerfectBreak(
            reds_left=0,
            phase=BreakPhase.FINISHED if all_gone else BreakPhase.CLEARING_COLOURS,
            colours_cleared=cleared,
            score=score,
        )
