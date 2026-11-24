"""snooker/ and pool/ own the table numbers. Vision keeps its own copy for now:
these checks make sure the two never drift apart."""

import pytest

from mr147.pool import table as pool
from mr147.snooker import table as snooker
from mr147.vision import table as vision


def test_snooker_numbers_match_vision():
    assert snooker.TABLE_LENGTH == vision.SNOOKER.length
    assert snooker.TABLE_WIDTH == vision.SNOOKER.width
    assert snooker.BALL_DIAMETER == vision.SNOOKER.ball_d
    assert snooker.POCKETS == [(x, y) for x, y, _ in vision.SNOOKER.pockets]
    spots = {c.value: p for c, p in snooker.SPOTS.items()}
    assert spots.keys() == vision.SPOTS.keys()
    for name, spot in vision.SPOTS.items():
        assert spots[name] == pytest.approx(spot)


def test_pool_numbers_match_vision():
    assert pool.TABLE_LENGTH == vision.POOL.length
    assert pool.TABLE_WIDTH == vision.POOL.width
    assert pool.BALL_DIAMETER == vision.POOL.ball_d
    assert pool.POCKETS == [(x, y) for x, y, _ in vision.POOL.pockets]
    assert pool.FOOT_SPOT == (vision.POOL_FOOT_X, 0.0)


def test_respotted_pool_ball_goes_behind_a_ball_on_the_foot_spot():
    x, y = pool.respot_position([pool.FOOT_SPOT])
    assert y == 0.0 and x >= pool.FOOT_SPOT[0] + pool.BALL_DIAMETER
