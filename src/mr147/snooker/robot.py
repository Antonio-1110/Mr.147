"""SnookerRobot: look -> plan -> shoot -> check, until 147 or a miss.

This file only connects the parts. Every decision belongs to one of them:

    VisionSystem   (interfaces)    what is on the table
    TableMemory    (table_memory)  whether to believe it, and what has been potted
    PerfectBreak   (snooker.rules)       which ball is on, and which comes next
    ShotPlanner    (shot_planner)  which shot to play, and where the cue ball should stop
    CueArm         (interfaces)    playing it
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from mr147.interfaces.cue_arm import CueArm
from mr147.interfaces.vision import DetectionFrame, VisionSystem
from mr147.snooker.rules import PerfectBreak
from mr147.shot_planner import PlannedShot, ShotPlanner
from mr147.snooker.balls import CUE_BALL, Colour
from mr147.table_memory import FrameCheck, TableMemory, TableState

log = logging.getLogger(__name__)

# A function that checks a frame: (frame, trust_camera) -> verdict.
FrameChecker = Callable[[DetectionFrame, bool], FrameCheck]


class NeedsHumanHelp(RuntimeError):
    """Vision can't produce a believable table, even after looking again."""


@dataclass
class ShotReport:
    plan: PlannedShot | None  # None if no shot was played
    check: FrameCheck | None  # what the table memory made of the frame afterwards
    score: int
    note: str = ""


class SnookerRobot:
    def __init__(
        self,
        vision: VisionSystem,
        cue_arm: CueArm,
        planner: ShotPlanner,
        better_views_before_giving_up: int = 3,
        on_plan: Callable[[TableState, PlannedShot], None] | None = None,
    ):
        self.vision = vision
        self.cue_arm = cue_arm
        self.planner = planner
        self.better_views_before_giving_up = better_views_before_giving_up
        self.on_plan = on_plan  # e.g. save a picture of each planned shot
        self.memory = TableMemory()
        self.rules: PerfectBreak | None = None

    def play_break(self, max_shots: int = 40) -> int:
        """Play until the break is over. Returns the score."""
        self.look_at_table()
        for _ in range(max_shots):
            report = self.play_one_shot()
            if self.rules.is_over or report.plan is None:
                log.info("break over: %d (%s)", report.score, report.note or self.rules.phase.name)
                break
        return self.rules.score

    def look_at_table(self) -> None:
        """First look at the table before the break starts."""

        def check(frame: DetectionFrame, trust_camera: bool) -> FrameCheck:
            return self.memory.first_look(frame, trust_camera=trust_camera)

        result = self._look_until_believable(check)
        reds = result.table.count_by_colour()[Colour.RED]
        self.rules = PerfectBreak.starting_with(reds)
        log.info("%d reds on the table, best possible break %d", reds, self.rules.best_possible_score())

    def play_one_shot(self) -> ShotReport:
        assert self.rules is not None, "call look_at_table() first"
        ball_on = self.rules.ball_on()
        if ball_on is None:
            return ShotReport(None, None, self.rules.score, self.rules.why_failed or "break finished")

        # Plan.
        respotted = self.rules.colours_respotted_if_potted()
        plan = self.planner.plan(self.memory.table, ball_on, self.rules.ball_on_after_this_one(), respotted)
        if plan is None:
            return ShotReport(None, None, self.rules.score, f"no way to pot the {ball_on}")
        log.info(
            "shot %d: %s into pocket %d  chance %.0f%%  position %.2f  cue ball stops at (%.2f, %.2f)",
            self.memory.shots_played + 1,
            plan.target,
            plan.pocket,
            100 * plan.chance_of_potting,
            plan.position_quality,
            *plan.cue_ball_stops_at,
        )

        if self.on_plan is not None:
            self.on_plan(self.memory.table, plan)

        # Shoot.
        self.cue_arm.play(plan.shot)

        # Check what happened.
        def check(frame: DetectionFrame, trust_camera: bool) -> FrameCheck:
            return self.memory.after_shot(
                frame, planned_pots=[ball_on], respotted_colours=respotted, trust_camera=trust_camera
            )

        result = self._look_until_believable(check)
        self.rules.record_shot(result.potted)
        return ShotReport(plan, result, self.rules.score, self.rules.why_failed or "")

    def _look_until_believable(self, check: FrameChecker) -> FrameCheck:
        """Ask vision for better views until the table memory accepts a frame.

        If it still disagrees after a few tries, believe the camera: most likely
        the shot simply didn't do what the plan said.
        """
        frame = self.vision.observe(in_play=self.memory.in_play())
        for attempt in range(1, self.better_views_before_giving_up + 1):
            result = check(frame, False)
            if result.accepted:
                return result
            log.info("frame rejected (%d): %s", attempt, "; ".join(result.view_request.reasons))
            frame = self.vision.request_better_view(result.view_request)

        result = check(frame, True)
        if CUE_BALL.value not in result.table.positions:
            raise NeedsHumanHelp("can't find the cue ball")
        return result
