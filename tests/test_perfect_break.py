from collections import Counter

from mr147.snooker.rules import BreakPhase, PerfectBreak
from mr147.snooker.balls import COLOURS_IN_CLEARANCE_ORDER, Colour


def test_potting_the_ball_on_every_time_scores_147_in_36_shots():
    rules = PerfectBreak()
    assert rules.best_possible_score() == 147
    shots = 0
    while not rules.is_over:
        rules.record_shot(Counter({rules.ball_on(): 1}))
        shots += 1
    assert rules.phase == BreakPhase.FINISHED
    assert rules.score == 147
    assert shots == 36


def test_red_then_black_and_the_black_is_respotted():
    rules = PerfectBreak()
    assert rules.ball_on() == Colour.RED
    assert rules.ball_on_after_this_one() == Colour.BLACK
    assert rules.colours_respotted_if_potted() == set()

    rules.record_shot(Counter({Colour.RED: 1}))
    assert rules.ball_on() == Colour.BLACK
    assert rules.ball_on_after_this_one() == Colour.RED
    assert rules.colours_respotted_if_potted() == {Colour.BLACK}


def test_black_after_the_last_red_is_respotted_then_colours_are_cleared():
    rules = PerfectBreak.starting_with(reds_on_table=1)
    rules.record_shot(Counter({Colour.RED: 1}))
    assert rules.colours_respotted_if_potted() == {Colour.BLACK}
    assert rules.ball_on_after_this_one() == Colour.YELLOW

    rules.record_shot(Counter({Colour.BLACK: 1}))
    assert rules.phase == BreakPhase.CLEARING_COLOURS
    assert rules.ball_on() == COLOURS_IN_CLEARANCE_ORDER[0]
    assert rules.colours_respotted_if_potted() == set()


def test_anything_except_exactly_the_ball_on_ends_the_break():
    for potted in (Counter(), Counter({Colour.PINK: 1}), Counter({Colour.RED: 2})):
        rules = PerfectBreak()
        rules.record_shot(potted)
        assert rules.phase == BreakPhase.FAILED
        assert rules.score == 0
        assert rules.ball_on() is None
