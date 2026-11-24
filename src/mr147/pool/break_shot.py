"""The pool break: where to put the cue ball, where to hit the rack, how hard.

The break is chaotic: a fraction of a millimetre of gap in the rack changes
where every ball goes. So a break is never judged on one simulation. Each
candidate break is played on many slightly different racks (the same racks for
every candidate, to compare fairly), and we count how often:

    it is legal and keeps the table (something drops, no scratch), and
    there is then a ball we are allowed to pot from where the cue ball stopped.

The break with the best share of "drops a ball AND leaves a shot" wins. Like the
shot planner, the top few are re-tested on fresh racks so a lucky one isn't
over-rated.

The robot has ball in hand behind the head string for the break, so the cue
ball's position is part of the search.
"""

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from mr147.interfaces.cue_arm import Shot
from mr147.physics import PhysicsSimulator
from mr147.pool.balls import ball_id
from mr147.pool.eight_ball import EightBallGame
from mr147.pool.layouts import eight_ball_rack, nine_ball_rack
from mr147.pool.nine_ball import NineBallGame
from mr147.pool.shot_goals import outcome_of
from mr147.pool.table import FOOT_SPOT, HEAD_STRING_X, Position, distance
from mr147.shot_planner import pot_geometry
from mr147.table_memory.table_state import TableState


@dataclass
class BreakSettings:
    cue_x: float = HEAD_STRING_X - 0.03  # just behind the head string
    cue_ys: tuple[float, ...] = (-0.45, -0.25, 0.0, 0.25, 0.45)  # across the table (the rail is at +-0.6)
    speeds: tuple[float, ...] = (7.0, 9.0)  # metres per second
    aim_offsets: tuple[float, ...] = (-0.4, 0.0, 0.4)  # where on the head ball, in ball radii (0 = full)
    height: float = -0.1  # a touch of screw keeps the cue ball near the middle
    racks: int = 8  # racks each candidate is tried on
    confirm_top: int = 3
    confirm_racks: int = 20
    rack_gap: float = 0.0005  # metres: how imperfect a rack is (largest gap between balls)


@dataclass(frozen=True)
class BreakPlan:
    cue_ball: Position  # where to place the cue ball
    shot: Shot
    keeps_table: float  # share of racks where the break was legal and something dropped
    leaves_a_shot: float  # of those: share with a ball to pot next
    score: float  # keeps_table * leaves_a_shot


def _new_game(game: str):
    return NineBallGame() if game == "nine" else EightBallGame()


def _rack(game: str) -> Callable[[random.Random, float], TableState]:
    return (lambda rng, gap: nine_ball_rack(rng, gap)) if game == "nine" else (lambda rng, gap: eight_ball_rack(rng, gap))


def _legal_targets(game_state) -> frozenset[int]:
    return frozenset({game_state.ball_on()}) if isinstance(game_state, NineBallGame) else game_state.legal_first_balls()


class BreakPlanner:
    def __init__(self, physics: PhysicsSimulator, game: str = "nine", settings: BreakSettings | None = None):
        self.physics = physics
        self.game = game
        self.settings = settings or BreakSettings()

    def plan(self, rng: random.Random | None = None) -> BreakPlan:
        s = self.settings
        rng = rng or random.Random(0)
        racks = [_rack(self.game)(random.Random(rng.random()), s.rack_gap) for _ in range(s.racks)]

        tried = []
        for y in s.cue_ys:
            cue_ball = (s.cue_x, y)
            for offset in s.aim_offsets:
                for speed in s.speeds:
                    shot = self._shot(cue_ball, offset, speed)
                    tried.append(self._judge(cue_ball, shot, racks))
        tried.sort(key=lambda b: b.score, reverse=True)

        fresh = [_rack(self.game)(random.Random(rng.random()), s.rack_gap) for _ in range(s.confirm_racks)]
        confirmed = [self._judge(b.cue_ball, b.shot, fresh) for b in tried[: s.confirm_top]]
        return max(confirmed, key=lambda b: b.score)

    def _shot(self, cue_ball: Position, offset: float, speed: float) -> Shot:
        """Aim at the head ball (on the foot spot), `offset` ball radii to the side of full."""
        head = FOOT_SPOT
        direction = pot_geometry.direction_deg(cue_ball, head)
        r = self.physics.table.ball_radius
        dist = distance(cue_ball, head)
        direction += math.degrees(math.atan2(offset * r, dist))
        return Shot(direction, speed, height=self.settings.height)

    def _judge(self, cue_ball: Position, shot: Shot, racks: list[TableState]) -> BreakPlan:
        keeps = 0
        leaves = 0
        for rack in racks:
            table = rack.with_ball_at("white", cue_ball)
            result = self.physics.simulate(table, shot)
            game = _new_game(self.game)
            game.play(outcome_of(result))
            if game.last_foul is not None or not result.potted_ball_ids:
                continue
            if game.is_over:  # e.g. the 9 on the break
                keeps += 1
                leaves += 1
                continue
            if game.shooter != 0:
                continue
            keeps += 1
            after = result.final_table
            targets = [ball_id(n) for n in _legal_targets(game) if ball_id(n) in after.positions]
            if pot_geometry.next_pot_ease(after, targets, self.physics.pockets, self.physics.table) > 0:
                leaves += 1
        n = len(racks)
        keeps_table = keeps / n
        leaves_a_shot = leaves / keeps if keeps else 0.0
        return BreakPlan(cue_ball, shot, keeps_table, leaves_a_shot, keeps_table * leaves_a_shot)
