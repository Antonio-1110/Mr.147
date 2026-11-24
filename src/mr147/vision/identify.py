"""Which ball is which: give every ball a name, using each name only as often as it exists.

The detector scores each ball for every name it might have ("8: 0.6, 4: 0.3").
Taken one at a time, three dark balls can all come out as "8". But a pool table
has only one 8 ball (and snooker only one black, at most 15 reds), so we choose
the names for all balls together:

    the most likely overall set of names, with each name used no more often than allowed.

The ball the detector is surest is the 8 keeps it; the others get their next-best
name. A ball left with a name the detector barely considered gets a low
confidence, so the table memory asks for a closer look at it.

If the table memory says which balls are still in play, a potted ball's name is
strongly discouraged too: a dark ball that looks a bit like the potted 4 gets
its next-best live number. Only a ball that really is a near-certain 4 keeps the
name (and the memory then sees a "potted ball is back" and asks for another look).

This is the classic "assignment problem" (solved exactly with scipy's
linear_sum_assignment).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from mr147.table_memory.ball_sets import BALL_SETS

UNLIKELY = 0.01  # score for a name the detector didn't suggest at all
SURPRISE = 0.05  # score multiplier for a name the memory says is no longer in play


def name_for_memory(label: str, game: str) -> str:
    """Detector label -> the name in the vision contract (pool "cue" -> "white")."""
    return "white" if game == "pool" and label == "cue" else label


def name_limits(game: str) -> dict[str, int]:
    """How many balls can carry each name in this game."""
    balls = BALL_SETS[game]
    limits = {str(name): n for name, n in balls.full_set.items()}
    limits[balls.cue_ball] = 1
    return limits


def assign_names(scores: Sequence[Mapping[str, float]], limits: Mapping[str, int],
                 expected: Mapping[str, int] | None = None) -> list[tuple[str, float]]:
    """Give each ball one name, never using a name more often than its limit.

    scores: for each ball, name -> how likely (0..1). Names not listed count as UNLIKELY.
    expected: how many of each name the memory believes are still on the table
              (None = no idea). Copies beyond that are scored down by SURPRISE.
    Returns (name, score) per ball. If there are more balls than names allow, the
    leftover balls keep their own best name (the table memory will then complain
    about the count and ask for a better look).
    """
    if not scores:
        return []
    names = [(n, k) for n in limits for k in range(limits[n])]  # one column per allowed copy
    cost = np.empty((len(scores), len(names)))
    for i, s in enumerate(scores):
        for j, (n, k) in enumerate(names):
            factor = 1.0 if expected is None or k < expected.get(n, 0) else SURPRISE
            cost[i, j] = -math.log(max(s.get(n, 0.0), UNLIKELY) * factor)
    rows, cols = linear_sum_assignment(cost)
    out = [_best(s) for s in scores]
    for i, j in zip(rows, cols):
        n, k = names[j]
        score = max(scores[i].get(n, 0.0), UNLIKELY)
        if expected is not None and k >= expected.get(n, 0):
            score *= SURPRISE  # "you said it was potted": low confidence, memory takes a look
        out[i] = (n, score)
    return out


def _best(s: Mapping[str, float]) -> tuple[str, float]:
    return max(s.items(), key=lambda kv: kv[1]) if s else ("?", 0.0)


def name_once_in_picture(found: list, game: str) -> list:
    """The same rule inside one picture: rename PixelDetections so no name repeats too often.

    Pockets are left alone. Used by the video check to show what the robot would believe.
    """
    from .detector import PixelDetection

    balls = [d for d in found if d.name != "pocket"]
    scores = [{name_for_memory(n, game): v for n, v in d.scores.items()} for d in balls]
    named = assign_names(scores, name_limits(game))
    out = [d for d in found if d.name == "pocket"]
    for d, (name, score) in zip(balls, named):
        label = "cue" if game == "pool" and name == "white" else name
        out.append(PixelDetection(label, d.box, score, d.alternatives))
    return out
