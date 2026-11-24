"""Snooker files moved into mr147.snooker. The old names must keep working."""


def test_old_names_still_import():
    from mr147.play_simulated_break import main, robot_eyes  # noqa: F401
    from mr147.robot import NeedsHumanHelp, ShotReport, SnookerRobot  # noqa: F401
    from mr147.rules import BreakPhase, PerfectBreak
    from mr147.rules.perfect_break import PerfectBreak as P2
    from mr147.simulated_hardware import SimulatedTable, after_break_off, full_rack, open_table  # noqa: F401
    from mr147.simulated_hardware.random_layouts import open_table as o2  # noqa: F401
    from mr147.simulated_hardware.simulated_table import SimulatedTable as S2
    from mr147.snooker.rules import PerfectBreak as P3
    from mr147.snooker.simulated_table import SimulatedTable as S3

    assert PerfectBreak is P2 is P3 and S2 is S3 and BreakPhase
