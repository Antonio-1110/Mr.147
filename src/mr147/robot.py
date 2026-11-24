"""Moved: the snooker robot now lives in mr147.snooker.robot (the pool one in mr147.pool.robot).
This name keeps old imports working."""

from mr147.snooker.robot import FrameChecker, NeedsHumanHelp, ShotReport, SnookerRobot

__all__ = ["FrameChecker", "NeedsHumanHelp", "ShotReport", "SnookerRobot"]
