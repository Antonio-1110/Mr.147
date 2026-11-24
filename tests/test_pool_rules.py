from mr147.pool import CUE_BALL, EIGHT_BALL, NINE_BALL, EightBallGame, Group, NineBallGame, ShotOutcome

# ---- 8-ball --------------------------------------------------------------------------

GOOD_BREAK = ShotOutcome(first_ball_hit=1, balls_to_cushion_on_break=5)


def eight_ball_after_break(**kwargs) -> EightBallGame:
    game = EightBallGame(**kwargs)
    game.play(GOOD_BREAK)  # nothing potted: player 1's turn, table still open
    assert game.shooter == 1 and game.table_is_open and not game.ball_in_hand
    return game


def test_break_with_a_pot_keeps_the_turn_and_leaves_the_table_open():
    game = EightBallGame()
    game.play(ShotOutcome(first_ball_hit=1, potted=(3,)))
    assert game.shooter == 0
    assert game.table_is_open  # pots on the break don't choose groups
    assert 3 not in game.on_table


def test_weak_break_is_a_foul():
    game = EightBallGame()
    game.play(ShotOutcome(first_ball_hit=1, balls_to_cushion_on_break=2))
    assert game.last_foul and game.shooter == 1 and game.ball_in_hand


def test_scratch_on_the_break_gives_ball_in_hand_behind_the_head_string():
    game = EightBallGame()
    game.play(ShotOutcome(first_ball_hit=1, potted=(CUE_BALL,), balls_to_cushion_on_break=6))
    assert game.shooter == 1 and game.ball_in_hand and game.ball_in_hand_behind_head_string


def test_eight_on_the_break_is_respotted():
    game = EightBallGame()
    game.play(ShotOutcome(first_ball_hit=1, potted=(EIGHT_BALL,)))
    assert not game.is_over
    assert EIGHT_BALL in game.on_table and game.respot == (EIGHT_BALL,)


def test_first_legal_pot_chooses_the_groups():
    game = eight_ball_after_break()
    game.play(ShotOutcome(first_ball_hit=12, potted=(12,)))
    assert game.groups == {1: Group.STRIPES, 0: Group.SOLIDS}
    assert game.shooter == 1
    assert game.legal_first_balls() == Group.STRIPES.balls - {12}


def test_hitting_the_eight_first_on_an_open_table_is_a_foul():
    game = eight_ball_after_break()
    game.play(ShotOutcome(first_ball_hit=EIGHT_BALL, cushion_after_contact=True))
    assert game.last_foul and game.shooter == 0 and game.ball_in_hand


def test_hitting_the_opponents_ball_first_is_a_foul():
    game = eight_ball_after_break()
    game.play(ShotOutcome(first_ball_hit=12, potted=(12,)))  # player 1 = stripes
    game.play(ShotOutcome(first_ball_hit=2, potted=(2,)))
    assert game.last_foul == "hit the 2 (blue) first"
    assert game.shooter == 0 and game.ball_in_hand
    assert 2 not in game.on_table  # a ball potted on a foul stays down


def test_no_pot_and_no_cushion_is_a_foul():
    game = eight_ball_after_break()
    game.play(ShotOutcome(first_ball_hit=4))
    assert game.last_foul and game.ball_in_hand


def test_potting_only_the_opponents_ball_ends_the_turn():
    game = eight_ball_after_break()
    game.play(ShotOutcome(first_ball_hit=12, potted=(12,)))  # player 1 = stripes
    game.play(ShotOutcome(first_ball_hit=13, potted=(5,)))
    assert game.last_foul is None and game.shooter == 0 and not game.ball_in_hand


def test_potting_the_eight_early_loses():
    game = eight_ball_after_break()
    game.play(ShotOutcome(first_ball_hit=12, potted=(12, EIGHT_BALL)))
    assert game.winner == 0 and "before clearing" in game.why_over


def stripes_cleared() -> EightBallGame:
    game = EightBallGame(on_table={1, 2, 3, EIGHT_BALL, 9}, is_break=False, shooter=1)
    game.play(ShotOutcome(first_ball_hit=9, potted=(9,)))  # player 1 = stripes, now on the 8
    assert game.is_on_the_eight() and game.shooter == 1
    return game


def test_potting_the_eight_in_the_called_pocket_wins():
    game = stripes_cleared()
    game.play(ShotOutcome(first_ball_hit=8, potted=(8,), pockets={8: 2}, called_pocket=2))
    assert game.winner == 1


def test_eight_in_the_wrong_pocket_loses():
    game = stripes_cleared()
    game.play(ShotOutcome(first_ball_hit=8, potted=(8,), pockets={8: 3}, called_pocket=2))
    assert game.winner == 0


def test_eight_with_a_scratch_loses():
    game = stripes_cleared()
    game.play(ShotOutcome(first_ball_hit=8, potted=(8, CUE_BALL), pockets={8: 2}, called_pocket=2))
    assert game.winner == 0 and "fouled" in game.why_over


# ---- 9-ball --------------------------------------------------------------------------


def test_must_hit_the_lowest_ball_first():
    game = NineBallGame(on_table={3, 5, 9}, is_break=False)
    assert game.ball_on() == 3
    game.play(ShotOutcome(first_ball_hit=5, potted=(5,)))
    assert game.last_foul and game.shooter == 1 and game.ball_in_hand
    assert 5 not in game.on_table


def test_combination_onto_the_nine_wins():
    game = NineBallGame(on_table={3, 5, 9}, is_break=False)
    game.play(ShotOutcome(first_ball_hit=3, potted=(NINE_BALL,)))
    assert game.winner == 0


def test_nine_on_the_break_wins():
    game = NineBallGame()
    game.play(ShotOutcome(first_ball_hit=1, potted=(NINE_BALL,)))
    assert game.winner == 0


def test_nine_potted_on_a_foul_is_respotted():
    game = NineBallGame(on_table={3, 9}, is_break=False)
    game.play(ShotOutcome(first_ball_hit=3, potted=(NINE_BALL, CUE_BALL)))
    assert not game.is_over and NINE_BALL in game.on_table and game.respot == (NINE_BALL,)
    assert game.shooter == 1 and game.ball_in_hand


def test_legal_pot_keeps_the_turn_and_a_miss_passes_it():
    game = NineBallGame(on_table={1, 2, 9}, is_break=False)
    game.play(ShotOutcome(first_ball_hit=1, potted=(1,)))
    assert game.shooter == 0 and game.ball_on() == 2
    game.play(ShotOutcome(first_ball_hit=2, cushion_after_contact=True))
    assert game.shooter == 1 and game.last_foul is None and not game.ball_in_hand


def test_three_fouls_in_a_row_lose():
    game = NineBallGame(on_table={4, 9}, is_break=False)
    for _ in range(3):
        game.shooter = 0  # the opponent misses safely in between (not modelled here)
        game.play(ShotOutcome(first_ball_hit=None))
    assert game.winner == 1 and "3 times" in game.why_over


def test_a_legal_shot_resets_the_foul_count():
    game = NineBallGame(on_table={4, 9}, is_break=False)
    game.play(ShotOutcome(first_ball_hit=None))
    game.play(ShotOutcome(first_ball_hit=4, cushion_after_contact=True))  # player 1, legal
    game.play(ShotOutcome(first_ball_hit=None))  # player 0 fouls again
    assert game.fouls_in_a_row == {0: 2, 1: 0}
