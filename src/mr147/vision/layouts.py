"""Random but believable ball layouts."""

import math

import numpy as np

from .table import (BALL_D, BAULK_X, D_RADIUS, POOL, POOL_FOOT_X, POOL_HEAD_X, SNOOKER, SPOTS, Ball,
                    TableSpec)

COLOURS = ["yellow", "green", "brown", "blue", "pink", "black"]


def _free(x: float, y: float, placed: list[Ball], gap: float = 0.0005, spec: TableSpec = SNOOKER) -> bool:
    if abs(x) > spec.length / 2 - spec.ball_r or abs(y) > spec.width / 2 - spec.ball_r:
        return False
    for px, py, pr in spec.pockets:  # not sitting in a pocket mouth
        if math.hypot(x - px, y - py) < pr * 0.9:
            return False
    return all(math.hypot(x - b.x, y - b.y) >= spec.ball_d + gap for b in placed)


def _random_spot(rng: np.random.Generator, placed: list[Ball], spec: TableSpec = SNOOKER) -> tuple[float, float]:
    """A free position. Mostly uniform, sometimes near a cushion or next to another ball."""
    HALF_L, HALF_W, BALL_D = spec.length / 2 - spec.ball_r, spec.width / 2 - spec.ball_r, spec.ball_d
    for _ in range(500):
        mode = rng.random()
        if mode < 0.6 or not placed:
            x, y = rng.uniform(-HALF_L, HALF_L), rng.uniform(-HALF_W, HALF_W)
        elif mode < 0.8:  # hugging a cushion
            if rng.random() < 0.5:
                x = rng.uniform(-HALF_L, HALF_L)
                y = math.copysign(HALF_W - rng.uniform(0, 0.03), rng.uniform(-1, 1))
            else:
                y = rng.uniform(-HALF_W, HALF_W)
                x = math.copysign(HALF_L - rng.uniform(0, 0.03), rng.uniform(-1, 1))
        else:  # touching or almost touching another ball (clusters)
            other = placed[rng.integers(len(placed))]
            a = rng.uniform(0, 2 * math.pi)
            d = BALL_D + rng.uniform(0.0006, 0.01)
            x, y = other.x + d * math.cos(a), other.y + d * math.sin(a)
        if _free(x, y, placed, spec=spec):
            return x, y
    raise RuntimeError("table too crowded")


def opening_rack(rng: np.random.Generator) -> list[Ball]:
    """Start of frame: reds in a triangle behind pink, colours on spots, cue ball in the D."""
    balls = [Ball(c, *SPOTS[c]) for c in COLOURS]
    px = SPOTS["pink"][0] + BALL_D + 0.003
    row_dx = BALL_D * math.sqrt(3) / 2
    k = 0
    for row in range(5):
        for j in range(row + 1):
            y = (j - row / 2) * (BALL_D + 0.0012)
            jx, jy = rng.uniform(-0.0003, 0.0003, 2)  # small gaps, never overlapping
            balls.append(Ball("red", px + row * (row_dx + 0.0012) + jx, y + jy, k))
            k += 1
    for _ in range(100):
        a, r = rng.uniform(math.pi / 2, 3 * math.pi / 2), D_RADIUS * math.sqrt(rng.random())
        x, y = BAULK_X + r * math.cos(a), r * math.sin(a)
        if _free(x, y, balls):
            balls.append(Ball("white", x, y))
            break
    return balls


def mid_frame(rng: np.random.Generator) -> list[Ball]:
    """Some reds and colours left, scattered, with clusters and balls on cushions."""
    n_reds = int(rng.choice([0, 1, 2, 3, 4, 6, 8, 10, 12, 15]))
    colours = COLOURS[:] if n_reds > 0 or rng.random() < 0.5 else COLOURS[rng.integers(0, 6):]
    balls: list[Ball] = []
    for c in colours:  # a colour often still sits on its spot
        if rng.random() < 0.4 and _free(*SPOTS[c], balls):
            balls.append(Ball(c, *SPOTS[c]))
        else:
            balls.append(Ball(c, *_random_spot(rng, balls)))
    if n_reds >= 8 and rng.random() < 0.5:  # a broken-open pack near the pink
        cx, cy = SPOTS["pink"][0] + 0.25, rng.uniform(-0.2, 0.2)
        for k in range(n_reds):
            for _ in range(200):
                x, y = cx + rng.normal(0, 0.12), cy + rng.normal(0, 0.15)
                if _free(x, y, balls):
                    balls.append(Ball("red", x, y, k))
                    break
            else:
                balls.append(Ball("red", *_random_spot(rng, balls), k))
    else:
        for k in range(n_reds):
            balls.append(Ball("red", *_random_spot(rng, balls), k))
    balls.append(Ball("white", *_random_spot(rng, balls)))
    return balls


def random_layout(rng: np.random.Generator, spec: TableSpec = SNOOKER) -> list[Ball]:
    if spec.game == "pool":
        return random_pool_layout(rng)
    return opening_rack(rng) if rng.random() < 0.15 else mid_frame(rng)


# --------------------------------------------------------------------------- pool


def _pool_cue_behind_head(rng: np.random.Generator, balls: list[Ball]) -> Ball:
    P = POOL
    for _ in range(200):
        x = rng.uniform(-P.length / 2 + P.ball_r, POOL_HEAD_X)
        y = rng.uniform(-P.width / 2 + P.ball_r, P.width / 2 - P.ball_r)
        if _free(x, y, balls, spec=P):
            return Ball("cue", x, y)
    return Ball("cue", *_random_spot(rng, balls, P))


def _rack(rng: np.random.Generator, rows: list[list[int]]) -> list[Ball]:
    """Balls in touching rows from the foot spot towards the foot cushion."""
    d = POOL.ball_d + 0.0012
    dx = d * math.sqrt(3) / 2
    balls = []
    for r, row in enumerate(rows):
        for j, num in enumerate(row):
            y = (j - (len(row) - 1) / 2) * d
            jx, jy = rng.uniform(-0.0003, 0.0003, 2)
            balls.append(Ball(str(num), POOL_FOOT_X + r * dx + jx, y + jy))
    return balls


def pool_rack_8ball(rng: np.random.Generator) -> list[Ball]:
    """15-ball triangle: 8 in the middle, a solid and a stripe in the back corners."""
    nums = [n for n in range(1, 16) if n != 8]
    rng.shuffle(nums)
    solids = [n for n in nums if n < 8]
    stripes = [n for n in nums if n > 8]
    corners = [solids.pop(), stripes.pop()]
    rng.shuffle(corners)
    rest = solids + stripes
    rng.shuffle(rest)
    sizes = [1, 2, 3, 4, 5]
    rows, k = [], 0
    for r, n in enumerate(sizes):
        row = []
        for j in range(n):
            if r == 2 and j == 1:
                row.append(8)
            elif r == 4 and j == 0:
                row.append(corners[0])
            elif r == 4 and j == 4:
                row.append(corners[1])
            else:
                row.append(rest[k])
                k += 1
        rows.append(row)
    balls = _rack(rng, rows)
    return balls + [_pool_cue_behind_head(rng, balls)]


def pool_rack_9ball(rng: np.random.Generator) -> list[Ball]:
    """Diamond: 1 at the front, 9 in the middle."""
    others = list(range(2, 9))
    rng.shuffle(others)
    rows = [[1], others[0:2], [others[2], 9, others[3]], others[4:6], [others[6]]]
    balls = _rack(rng, rows)
    return balls + [_pool_cue_behind_head(rng, balls)]


def pool_scatter(rng: np.random.Generator) -> list[Ball]:
    """A game in progress: some numbered balls left, spread out with clusters and rail balls."""
    # 9-ball tables have no stripes; kept rare so the stripes (the hardest to read) are
    # on the table about as often as the solids.
    pool = list(range(1, 10)) if rng.random() < 0.1 else list(range(1, 16))  # 9-ball or 8-ball
    n = int(rng.integers(1, len(pool) + 1))
    nums = rng.choice(pool, size=n, replace=False)
    balls: list[Ball] = []
    for num in nums:
        balls.append(Ball(str(int(num)), *_random_spot(rng, balls, POOL)))
    if rng.random() < 0.95:
        balls.append(Ball("cue", *_random_spot(rng, balls, POOL)))
    return balls


def random_pool_layout(rng: np.random.Generator) -> list[Ball]:
    r = rng.random()
    if r < 0.12:
        return pool_rack_8ball(rng)
    if r < 0.16:
        return pool_rack_9ball(rng)
    return pool_scatter(rng)
