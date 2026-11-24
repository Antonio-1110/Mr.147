"""9-ball rules (World Pool-Billiard Association rules, slightly simplified).

Only balls 1 to 9 are used. How a game goes:
  * Every shot must hit the lowest-numbered ball on the table first.
  * Then something must be potted or touch a cushion. On the break, something
    must be potted or at least 4 balls must reach a cushion.
  * Any ball potted on a legal shot counts, so a combination onto the 9 wins,
    even on the break.
  * You keep shooting while you legally pot any ball.
  * A foul gives the opponent ball in hand anywhere on the table. A 9 potted on
    a foul is put back on the foot spot; other balls stay down.
  * Three fouls in a row by the same player lose the game.

Simplified: no "push out" after the break.
"""

from dataclasses import dataclass, field

from mr147.pool.balls import NINE_BALL, describe
from mr147.pool.shot import ShotOutcome, basic_foul

FOULS_IN_A_ROW_THAT_LOSE = 3


@dataclass
class NineBallGame:
    on_table: set[int] = field(default_factory=lambda: set(range(1, 10)))
    shooter: int = 0
    is_break: bool = True
    ball_in_hand: bool = False
    fouls_in_a_row: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    respot: tuple[int, ...] = ()  # balls to put back on the foot spot after the last shot
    winner: int | None = None
    last_foul: str | None = None
    why_over: str | None = None

    @property
    def is_over(self) -> bool:
        return self.winner is not None

    def ball_on(self) -> int:
        """The ball that must be hit first: the lowest number on the table."""
        return min(self.on_table)

    def play(self, outcome: ShotOutcome) -> None:
        """Apply one shot's outcome to the game."""
        if self.is_over:
            raise RuntimeError("the game is already over")

        shooter, opponent = self.shooter, 1 - self.shooter
        foul = basic_foul(outcome, frozenset({self.ball_on()}), self.is_break)
        potted = outcome.object_balls_potted

        self.is_break = False
        self.last_foul = foul
        self.respot = ()
        self.on_table -= set(potted)

        if foul:
            if NINE_BALL in potted:
                self.on_table.add(NINE_BALL)
                self.respot = (NINE_BALL,)
            self.fouls_in_a_row[shooter] += 1
            if self.fouls_in_a_row[shooter] >= FOULS_IN_A_ROW_THAT_LOSE:
                self.winner, self.why_over = opponent, f"player {shooter} fouled 3 times in a row"
                return
            self.shooter, self.ball_in_hand = opponent, True
            return

        self.fouls_in_a_row[shooter] = 0
        self.ball_in_hand = False
        if NINE_BALL in potted:
            self.winner, self.why_over = shooter, f"player {shooter} potted the 9"
            return
        if not potted:
            self.shooter = opponent

    def __str__(self) -> str:
        if self.is_over:
            return f"game over: player {self.winner} wins ({self.why_over})"
        hand = " (ball in hand)" if self.ball_in_hand else ""
        return f"player {self.shooter} to shoot{hand}; hit the {describe(self.ball_on())} first"
