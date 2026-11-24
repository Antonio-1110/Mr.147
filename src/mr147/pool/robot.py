"""PoolRobot: break, then run out the rack. Look -> plan -> shoot -> check, like the snooker robot.

It plays alone, to show good technique and clear the table: the run is over as
soon as a shot misses or fouls.

    BreakPlanner   (break_shot)     where to put the cue ball and how to break
    TableMemory    (table_memory)   whether to believe vision, what has been potted
    NineBallGame / EightBallGame    which ball may be hit, who has won
    ShotPlanner    (shot_planner)   the shot, with the game's own ShotGoal
    place_and_plan (ball_in_hand)   where to put the cue ball when we have ball in hand

The camera can't see which ball the cue ball hit first or whether a ball touched
a cushion, so the rules are told what the plan intended for those (the ball we
aimed at, and for the 8 the pocket we aimed at). What went down is what vision saw.
"""

import logging
import random
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from mr147.interfaces.cue_arm import CueArm
from mr147.interfaces.vision import DetectionFrame, VisionSystem
from mr147.pool.ball_in_hand import place_and_plan
from mr147.pool.balls import CUE_BALL, EIGHT_BALL, ball_id, describe, number_of
from mr147.pool.break_shot import BreakPlanner
from mr147.pool.eight_ball import EightBallGame
from mr147.pool.nine_ball import NineBallGame
from mr147.pool.shot import ShotOutcome
from mr147.pool.shot_goals import eight_ball_goal, nine_ball_goal
from mr147.pool.table import Position, respot_position
from mr147.shot_planner import ShotPlanner
from mr147.table_memory import POOL_BALLS, FrameCheck, TableMemory

log = logging.getLogger(__name__)


class PlacesBalls(Protocol):
    def place_ball(self, ball: str, position: Position) -> None:
        """Put a ball here by hand: the cue ball with ball in hand, or the 8 if it drops on the break."""
        ...


@dataclass
class RunReport:
    balls_potted: int
    shots: int
    won: bool  # potted the 9 / the 8 legally
    note: str


class PoolRobot:
    def __init__(
        self,
        vision: VisionSystem,
        cue_arm: CueArm,
        hand: PlacesBalls,
        planner: ShotPlanner,
        game: str = "nine",  # "nine" or "eight"
        break_planner: BreakPlanner | None = None,
        better_views_before_giving_up: int = 3,
    ):
        self.vision = vision
        self.cue_arm = cue_arm
        self.hand = hand
        self.planner = planner
        self.game_kind = game
        self.break_planner = break_planner or BreakPlanner(planner.physics, game)
        self.better_views_before_giving_up = better_views_before_giving_up
        self.memory = TableMemory(balls=POOL_BALLS)
        self.game = NineBallGame() if game == "nine" else EightBallGame()
        self.balls_potted = 0
        self.shots = 0

    def run_rack(self, max_shots: int = 30, rng: random.Random | None = None) -> RunReport:
        """Break and try to clear the table. Returns how it went."""
        rng = rng or random.Random(0)
        self._look(first=True)

        plan = self.break_planner.plan(rng)
        log.info(
            "break from (%.2f, %.2f) at %.1f m/s: keeps the table %.0f%%, then leaves a shot %.0f%%",
            *plan.cue_ball, plan.shot.speed, 100 * plan.keeps_table, 100 * plan.leaves_a_shot,
        )
        self.hand.place_ball("white", plan.cue_ball)
        self.cue_arm.play(plan.shot)
        note = self._after_shot(first_ball_hit=None, aimed_pocket=None, is_break=True)

        while note is None and self.shots < max_shots:
            note = self._play_one_shot()
        report = RunReport(self.balls_potted, self.shots, self.game.winner == 0, note or "shot limit")
        log.info("run over: %d balls in %d shots (%s)", report.balls_potted, report.shots, report.note)
        return report

    # ---- one shot ----------------------------------------------------------------------

    def _goal(self):
        return nine_ball_goal(self.game) if isinstance(self.game, NineBallGame) else eight_ball_goal(self.game)

    def _play_one_shot(self) -> str | None:
        table = self.memory.table
        goal = self._goal()
        if self.game.ball_in_hand:
            placed = place_and_plan(table, goal, self.planner)
            if placed is None:
                return "ball in hand, but no pot anywhere"
            spot, plan = placed
            self.hand.place_ball("white", spot)
        else:
            plan = self.planner.plan_for(table, goal)
            if plan is None:
                return "no pot from here"

        log.info(
            "shot %d: %s into pocket %d  goes in %.0f%%  next ball there %.0f%%",
            self.shots + 1, describe(number_of(plan.target)), plan.pocket,
            100 * plan.chance_of_potting, 100 * plan.chance_break_continues,
        )
        self.cue_arm.play(plan.shot)
        return self._after_shot(first_ball_hit=number_of(plan.target), aimed_pocket=plan.pocket, is_break=False)

    def _after_shot(self, first_ball_hit: int | None, aimed_pocket: int | None, is_break: bool) -> str | None:
        """Look at the table, tell the rules what happened. Returns why the run is over, or None."""
        self.shots += 1
        check = self._look(first=False)
        potted = [number_of(str(name)) for name, n in check.potted.items() for _ in range(n)]
        cue_ball_gone = "white" not in check.table.positions
        object_balls = [b for b in potted if b != CUE_BALL]
        if cue_ball_gone:
            potted.append(CUE_BALL)

        outcome = ShotOutcome(
            first_ball_hit=first_ball_hit if not is_break else min(self.game.on_table),
            potted=tuple(potted),
            pockets={EIGHT_BALL: aimed_pocket} if EIGHT_BALL in potted and aimed_pocket is not None else {},
            cushion_after_contact=True,  # can't be seen; a planned pot always reaches a cushion or a pocket
            balls_to_cushion_on_break=4 if is_break else 0,
            called_pocket=aimed_pocket,
        )
        self.game.play(outcome)
        for ball in self.game.respot:  # the 8 on the break: no referee, so the robot puts it back
            spot = respot_position([p for b, p in self.memory.table.positions.items()])
            self.hand.place_ball(ball_id(ball), spot)
            self._look(first=False)  # memory sees it come back (not counted as a pot)
            object_balls.remove(ball)
        self.balls_potted += len(object_balls)
        if self.game.is_over:
            return self.game.why_over
        if self.game.last_foul:
            return f"foul: {self.game.last_foul}"
        if self.game.shooter != 0:
            return "missed" if not object_balls else "potted, but not a ball that keeps the table"
        return None

    # ---- looking -----------------------------------------------------------------------

    def _look(self, first: bool) -> FrameCheck:
        def check(frame: DetectionFrame, trust: bool) -> FrameCheck:
            if first:
                return self.memory.first_look(frame, trust_camera=trust)
            return self.memory.after_shot(frame, planned_pots=None, trust_camera=trust)

        frame = self.vision.observe(in_play=self.memory.in_play())
        for _ in range(self.better_views_before_giving_up):
            result = check(frame, False)
            if result.accepted:
                return result
            frame = self.vision.request_better_view(result.view_request)
        return check(frame, True)
