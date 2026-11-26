"""Never confidently wrong: name a ball only when several looks agree and the reader is sure.

A single picture can fool the reader (a stripe seen from its cap, a ball cut off by
the picture's edge). A person walking round the table looks again; so does the robot:

  1. follow each ball through the nearby video frames (it moves only a little between
     frames, so the nearest ball in the next frame is the same ball),
  2. average what the reader said about it in all of them,
  3. give the names for the whole picture together (one of each, identify.py),
  4. leave a ball unnamed ("?") when the averaged score for its name is below the bar.

An unnamed ball isn't a mistake: the table memory asks for another look at it. A
wrong name is a mistake, because the robot would plan a shot on the wrong ball.
"""

from __future__ import annotations

from .detector import PixelDetection
from .identify import name_once_in_picture

UNNAMED = "?"


def _width(d: PixelDetection) -> float:
    return max(d.box[2] - d.box[0], d.box[3] - d.box[1], 1.0)


def follow(frames: list[list[PixelDetection]], centre: int, max_jump: float = 1.0) -> list[list[PixelDetection]]:
    """For each ball in frames[centre]: the same ball in the other frames (where it could be followed).

    Going outwards from the centre frame, a ball is the nearest ball in the next frame
    if that one is within `max_jump` ball widths; otherwise it is lost in that direction.
    """
    balls = [d for d in frames[centre] if d.name != "pocket"]
    tracks = [[d] for d in balls]
    for step in (1, -1):
        last = list(balls)
        alive = [True] * len(balls)
        k = centre + step
        while 0 <= k < len(frames):
            here = [d for d in frames[k] if d.name != "pocket"]
            taken: set[int] = set()
            for i, prev in enumerate(last):
                if not alive[i]:
                    continue
                px, py = prev.centre
                best, dist = None, max_jump * _width(prev)
                for j, d in enumerate(here):
                    if j in taken:
                        continue
                    cx, cy = d.centre
                    dd = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    if dd < dist:
                        best, dist = j, dd
                if best is None:
                    alive[i] = False
                    continue
                taken.add(best)
                last[i] = here[best]
                tracks[i].append(here[best])
            k += step
    return tracks


def averaged(track: list[PixelDetection]) -> dict[str, float]:
    """What the reader said about one ball, averaged over every frame it was seen in."""
    total: dict[str, float] = {}
    for d in track:
        for name, p in d.scores.items():
            total[name] = total.get(name, 0.0) + p
    return {name: p / len(track) for name, p in total.items()}


def with_bar(named: list[PixelDetection], bar: float) -> list[PixelDetection]:
    """Balls whose name scored below the bar become UNNAMED (pockets untouched)."""
    return [d if d.name == "pocket" or d.confidence >= bar
            else PixelDetection(UNNAMED, d.box, d.confidence, d.alternatives) for d in named]


def name_balls(frames: list[list[PixelDetection]], centre: int, game: str = "pool", bar: float = 0.0,
               max_jump: float = 1.0) -> list[PixelDetection]:
    """Final names for the balls in frames[centre], using all the frames given (one frame = no vote)."""
    pockets = [d for d in frames[centre] if d.name == "pocket"]
    voted = []
    for track in follow(frames, centre, max_jump):
        scores = averaged(track)
        ranked = tuple(sorted(scores.items(), key=lambda kv: -kv[1])[:5])
        voted.append(PixelDetection(ranked[0][0], track[0].box, ranked[0][1], ranked))
    return with_bar(name_once_in_picture(pockets + voted, game), bar)
