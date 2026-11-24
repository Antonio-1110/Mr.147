"""Quick, physics-free shot geometry.

Used to find which pots are worth simulating, and to judge where the cue ball
stops: a good position is one where the next pot is easy.

Works for any game: table and ball size come from a GameTable (snooker by default).

Ghost ball: to send an object ball towards a pocket, the cue ball must touch it
from the opposite side. At contact the cue ball's centre is one ball diameter
behind the object ball, on the line from the pocket through the object ball.
That spot is the "ghost ball", and the cue ball is aimed at it.
"""

import math
from dataclasses import dataclass

from collections.abc import Iterable

from mr147.game_table import GameTable
from mr147.snooker.balls import Colour
from mr147.snooker.table import SNOOKER_TABLE, Position, distance
from mr147.table_memory.table_state import TableState

# Pots harder than these are not considered.
MAX_CUT_ANGLE_DEG = 80.0
MAX_MIDDLE_POCKET_ANGLE_DEG = 50.0  # angle away from straight-in for a middle pocket
MAX_CORNER_POCKET_ANGLE_DEG = 65.0  # angle away from the corner's diagonal


@dataclass(frozen=True)
class PotOption:
    """One way to pot one ball: which ball, which pocket, and how hard it looks."""

    target: str  # ball id
    pocket: int  # index into PhysicsSimulator.pockets
    aim_direction_deg: float  # cue ball towards the ghost ball
    cut_angle_deg: float  # 0 = dead straight
    ease: float  # 0..1, higher is easier


# ---- small geometry helpers ----------------------------------------------------------


def direction_deg(start: Position, end: Position) -> float:
    """Direction from start to end, in degrees (0 = +x, 90 = +y)."""
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))


def angle_between_deg(a: float, b: float) -> float:
    """Signed smallest difference a - b, in degrees between -180 and 180."""
    return (a - b + 180.0) % 360.0 - 180.0


def ghost_ball_position(object_ball: Position, pocket: Position, ball_diameter: float = SNOOKER_TABLE.ball_diameter) -> Position:
    """Where the cue ball's centre must be at contact to send object_ball at pocket."""
    to_pocket = direction_deg(object_ball, pocket)
    rad = math.radians(to_pocket)
    return (
        object_ball[0] - ball_diameter * math.cos(rad),
        object_ball[1] - ball_diameter * math.sin(rad),
    )


def path_is_clear(
    start: Position, end: Position, obstacles: list[Position], ball_diameter: float = SNOOKER_TABLE.ball_diameter
) -> bool:
    """Can a ball roll in a straight line from start to end without touching any obstacle?"""
    min_gap = ball_diameter - 0.001  # two ball centres closer than this would touch
    return all(_distance_to_segment(o, start, end) >= min_gap for o in obstacles)


def _distance_to_segment(point: Position, start: Position, end: Position) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return distance(point, start)
    # How far along the segment the closest point is (0 = start, 1 = end).
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    t = max(0.0, min(1.0, t))
    closest = (start[0] + t * dx, start[1] + t * dy)
    return distance(point, closest)


def _can_enter_pocket(object_ball: Position, pocket: Position, game: GameTable) -> bool:
    """Pockets can't be entered from too sharp an angle."""
    approach = direction_deg(object_ball, pocket)
    is_middle_pocket = abs(pocket[0]) < 0.1 * game.length
    if is_middle_pocket:
        straight_in = 90.0 if pocket[1] > 0 else -90.0
        return abs(angle_between_deg(approach, straight_in)) <= MAX_MIDDLE_POCKET_ANGLE_DEG
    corner_diagonal = math.degrees(
        math.atan2(math.copysign(game.width, pocket[1]), math.copysign(game.length, pocket[0]))
    )
    return abs(angle_between_deg(approach, corner_diagonal)) <= MAX_CORNER_POCKET_ANGLE_DEG


def _cue_ball_near_cushion(cue_ball: Position, game: GameTable) -> bool:
    """A cue ball tight on a cushion is hard to strike cleanly."""
    gap_x = game.length / 2 - abs(cue_ball[0]) - game.ball_radius
    gap_y = game.width / 2 - abs(cue_ball[1]) - game.ball_radius
    return min(gap_x, gap_y) < game.ball_radius


# ---- pot options ---------------------------------------------------------------------


def pot_option(
    table: TableState, target: str, pocket_index: int, pocket: Position, game: GameTable = SNOOKER_TABLE
) -> PotOption | None:
    """How to pot `target` into this pocket from where the cue ball is. None if impossible."""
    cue_ball = table.positions[game.cue_ball]
    object_ball = table.positions[target]
    obstacles = table.positions_except(game.cue_ball, target)
    d = game.ball_diameter

    if not _can_enter_pocket(object_ball, pocket, game):
        return None

    ghost = ghost_ball_position(object_ball, pocket, d)
    aim = direction_deg(cue_ball, ghost)
    cut = abs(angle_between_deg(aim, direction_deg(object_ball, pocket)))
    if cut > MAX_CUT_ANGLE_DEG:
        return None

    if not path_is_clear(cue_ball, ghost, obstacles, d):
        return None
    if not path_is_clear(object_ball, pocket, obstacles, d):
        return None

    # Thin cuts and long distances are harder. A rough score, not a probability.
    cue_to_ghost = distance(cue_ball, ghost)
    object_to_pocket = distance(object_ball, pocket)
    ease = math.cos(math.radians(cut)) ** 1.5 / (1.0 + 0.5 * cue_to_ghost + 1.0 * object_to_pocket)
    if _cue_ball_near_cushion(cue_ball, game):
        ease *= 0.6
    return PotOption(target, pocket_index, aim, cut, ease)


def all_pot_options(
    table: TableState, targets: Colour | Iterable[str], pockets: list[Position], game: GameTable = SNOOKER_TABLE
) -> list[PotOption]:
    """Every sensible pot of any of these balls, easiest first.

    `targets` is a list of ball ids, or (snooker) a colour meaning every ball of that colour.
    """
    if isinstance(targets, Colour):
        targets = table.ball_ids(targets)
    options = []
    for target in targets:
        if target not in table.positions:
            continue
        for pocket_index, pocket in enumerate(pockets):
            option = pot_option(table, target, pocket_index, pocket, game)
            if option is not None:
                options.append(option)
    options.sort(key=lambda option: option.ease, reverse=True)
    return options


def next_pot_ease(
    table: TableState,
    next_targets: Colour | Iterable[str] | None,
    pockets: list[Position],
    game: GameTable = SNOOKER_TABLE,
) -> float:
    """How good the cue ball's position is for potting one of `next_targets` (0..1).

    This is how we judge where the cue ball stops: 0 means no pot at all.
    """
    if next_targets is None:  # the last ball: nothing to play position for
        return 1.0
    options = all_pot_options(table, next_targets, pockets, game)
    return options[0].ease if options else 0.0
