"""8-ball rules (World Pool-Billiard Association rules, slightly simplified).

How a game goes:
  * Break from behind the head string. Any ball may be hit first.
  * The table is "open" until someone legally pots a ball after the break:
    that player then owns that group (solids 1-7 or stripes 9-15), the
    opponent owns the other.
  * Each shot must hit one of your own balls first (on an open table, anything
    but the 8). Then something must be potted or touch a cushion.
  * You keep shooting while you legally pot one of your balls (any ball on an
    open table or on the break).
  * A foul gives the opponent ball in hand: they put the cue ball anywhere.
    A scratch on the break puts it behind the head string instead.
  * Once your group is gone, you play the 8 and call its pocket. Potting it
    there legally wins.
  * You lose if you pot the 8 early, into the wrong pocket, or on a foul.
  * The 8 potted on the break is put back on the foot spot; play goes on.

Simplified: an illegal break is just a foul (no re-rack option), and only the
8 needs a called pocket (other balls count wherever they go). If one shot pots
both groups on an open table, the first ball to drop decides the groups.
"""

from dataclasses import dataclass, field

from mr147.pool.balls import ALL_OBJECT_BALLS, EIGHT_BALL, Group, describe, group_of
from mr147.pool.shot import ShotOutcome, basic_foul


@dataclass
class EightBallGame:
    on_table: set[int] = field(default_factory=lambda: set(ALL_OBJECT_BALLS))
    shooter: int = 0  # player 0 or 1
    groups: dict[int, Group] = field(default_factory=dict)  # empty = table is open
    is_break: bool = True
    ball_in_hand: bool = False  # the shooter may place the cue ball first
    ball_in_hand_behind_head_string: bool = False  # ...but only behind the head string
    respot: tuple[int, ...] = ()  # balls to put back on the foot spot after the last shot
    winner: int | None = None
    last_foul: str | None = None
    why_over: str | None = None

    # ---- questions ---------------------------------------------------------------

    @property
    def is_over(self) -> bool:
        return self.winner is not None

    @property
    def table_is_open(self) -> bool:
        return not self.groups

    def legal_first_balls(self) -> frozenset[int]:
        """Balls the shooter may hit first."""
        if self.is_break:
            return frozenset(self.on_table)
        if self.table_is_open:
            return frozenset(self.on_table - {EIGHT_BALL})
        mine = self.groups[self.shooter].balls & self.on_table
        return frozenset(mine) if mine else frozenset({EIGHT_BALL})

    def is_on_the_eight(self) -> bool:
        """True once the shooter's group is cleared: they must call a pocket for the 8."""
        return not self.is_break and self.legal_first_balls() == {EIGHT_BALL}

    def balls_left_for(self, player: int) -> frozenset[int]:
        """The balls this player still has to pot before the 8 (empty while the table is open)."""
        if player not in self.groups:
            return frozenset()
        return frozenset(self.groups[player].balls & self.on_table)

    # ---- updates -----------------------------------------------------------------

    def play(self, outcome: ShotOutcome) -> None:
        """Apply one shot's outcome to the game."""
        if self.is_over:
            raise RuntimeError("the game is already over")

        shooter, opponent = self.shooter, 1 - self.shooter
        was_break, was_on_eight = self.is_break, self.is_on_the_eight()
        legal_targets = self._balls_that_keep_the_turn()
        foul = basic_foul(outcome, self.legal_first_balls(), self.is_break)
        potted = outcome.object_balls_potted

        self.is_break = False
        self.last_foul = foul
        self.respot = ()
        self.on_table -= set(potted)

        if EIGHT_BALL in potted:
            if was_break:
                self.on_table.add(EIGHT_BALL)
                self.respot = (EIGHT_BALL,)
            elif foul:
                return self._end(opponent, f"player {shooter} fouled while potting the 8 ({foul})")
            elif not was_on_eight:
                return self._end(opponent, f"player {shooter} potted the 8 before clearing their group")
            elif outcome.called_pocket is None or outcome.pockets.get(EIGHT_BALL) != outcome.called_pocket:
                return self._end(opponent, f"player {shooter} potted the 8 in an uncalled pocket")
            else:
                return self._end(shooter, f"player {shooter} potted the 8")

        # Choosing groups: the first legal pot after the break.
        if self.table_is_open and not was_break and not foul:
            first_group = next((g for b in potted if (g := group_of(b)) is not None), None)
            if first_group is not None:
                self.groups = {shooter: first_group, opponent: first_group.other}

        keeps_turn = not foul and any(b in legal_targets for b in potted)
        if not keeps_turn:
            self.shooter = opponent
        self.ball_in_hand = foul is not None
        self.ball_in_hand_behind_head_string = foul is not None and was_break and outcome.scratched

    # ---- helpers -----------------------------------------------------------------

    def _balls_that_keep_the_turn(self) -> frozenset[int]:
        if self.is_break or self.table_is_open:
            return frozenset(ALL_OBJECT_BALLS - {EIGHT_BALL})
        return self.groups[self.shooter].balls

    def _end(self, winner: int, why: str) -> None:
        self.winner = winner
        self.why_over = why

    def __str__(self) -> str:
        if self.is_over:
            return f"game over: player {self.winner} wins ({self.why_over})"
        groups = ", ".join(f"player {p}: {g.name.lower()}" for p, g in sorted(self.groups.items())) or "open table"
        hand = " (ball in hand)" if self.ball_in_hand else ""
        targets = ", ".join(describe(b) for b in sorted(self.legal_first_balls()))
        return f"player {self.shooter} to shoot{hand}; {groups}; may hit first: {targets}"
