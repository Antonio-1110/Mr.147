"""ShotOutcome: what happened on one pool shot, as the rules need it.

The simulator (or vision, watching a real shot) fills this in. The rules then
decide whether it was a foul, who shoots next, and whether someone has won.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from mr147.pool.balls import CUE_BALL


@dataclass(frozen=True)
class ShotOutcome:
    """
    first_ball_hit:          the first ball the cue ball touched (None = touched nothing)
    potted:                  every ball that went down, cue ball (0) included, in order
    pockets:                 which pocket each potted ball went into (index into
                             mr147.pool.table.POCKETS). Only the 8 in 8-ball needs it.
    cushion_after_contact:   did any ball touch a cushion after the first contact?
                             (Needed when nothing is potted, or it's a foul.)
    balls_to_cushion_on_break: how many object balls reached a cushion on the break
    cue_ball_off_table:      the cue ball jumped off the table
    called_pocket:           the pocket called for the 8 (8-ball only)
    """

    first_ball_hit: int | None
    potted: tuple[int, ...] = ()
    pockets: Mapping[int, int] = field(default_factory=dict)
    cushion_after_contact: bool = False
    balls_to_cushion_on_break: int = 0
    cue_ball_off_table: bool = False
    called_pocket: int | None = None

    @property
    def object_balls_potted(self) -> tuple[int, ...]:
        return tuple(b for b in self.potted if b != CUE_BALL)

    @property
    def scratched(self) -> bool:
        """The cue ball went in a pocket or off the table."""
        return CUE_BALL in self.potted or self.cue_ball_off_table


def basic_foul(outcome: ShotOutcome, legal_first_balls: frozenset[int], is_break: bool) -> str | None:
    """The fouls 8-ball and 9-ball share. Returns why it's a foul, or None."""
    from mr147.pool.balls import describe  # local import keeps this file dependency-free

    if outcome.first_ball_hit is None:
        return "the cue ball didn't touch any ball"
    if outcome.first_ball_hit not in legal_first_balls:
        return f"hit the {describe(outcome.first_ball_hit)} first"
    if outcome.cue_ball_off_table:
        return "the cue ball left the table"
    if CUE_BALL in outcome.potted:
        return "scratch: the cue ball went in a pocket"
    if is_break:
        if not outcome.object_balls_potted and outcome.balls_to_cushion_on_break < 4:
            return "illegal break: nothing potted and fewer than 4 balls reached a cushion"
    elif not outcome.object_balls_potted and not outcome.cushion_after_contact:
        return "nothing potted and no ball reached a cushion"
    return None
