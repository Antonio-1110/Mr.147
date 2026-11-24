"""Pool ball sets vary (shades, stripes, numbers, cue ball) but always follow the same plan."""

import numpy as np

from mr147.vision.pool_balls import CLASSIC, PoolBallStyle, pool_ball_image


def test_random_sets_differ_but_keep_the_plan():
    rng = np.random.default_rng(0)
    a, b = PoolBallStyle.random(rng), PoolBallStyle.random(rng)
    assert a.colours != b.colours
    for style in (CLASSIC, a, b):
        solid, stripe = pool_ball_image(3, style), pool_ball_image(11, style)
        h, w = solid.shape[:2]
        # Solid: coloured at the poles. Stripe: white at the poles, same colour round the middle.
        assert np.allclose(solid[2, 5], np.array(style.colours[3]) * 255, atol=2)
        caps = style.stripe_caps if style.stripe_caps is not None else style.white
        assert np.allclose(stripe[2, 5], np.array(caps) * 255, atol=2)
        assert np.allclose(stripe[h // 2, 5], solid[h // 2, 5], atol=2)


def test_cue_ball_marks():
    rng = np.random.default_rng(3)
    styles = [PoolBallStyle.random(rng) for _ in range(30)]
    assert {s.cue for s in styles} == {"plain", "logo", "dots"}
    plain = next(s for s in styles if s.cue == "plain")
    img = pool_ball_image(0, plain)
    assert (img == img[0, 0]).all()  # one colour all over


def test_some_sets_have_black_stripe_caps():
    rng = np.random.default_rng(4)
    styles = [PoolBallStyle.random(rng) for _ in range(40)]
    black = [s for s in styles if s.stripe_caps is not None]
    assert 0 < len(black) < len(styles) and all(max(s.stripe_caps) < 0.1 for s in black)
    img = pool_ball_image(12, black[0])
    assert img[2, 5].max() < 30  # dark at the poles
    h = img.shape[0]
    assert img[h // 2, img.shape[1] // 4].min() > 150  # the number circle stays white
