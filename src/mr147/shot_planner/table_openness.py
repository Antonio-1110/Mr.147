"""How open is the table? A score from 0 (every ball tied up) to 1 (every ball free).

A player clearing the table wants balls that are on their own and can each go
in a pocket. For every ball still to be cleared:

    1.0  free: not touching another ball, and pottable into at least one pocket
    0.5  loose, but no pocket can be reached from where it sits
    0.0  tied up: touching (or nearly touching) another ball

The score is the average. The shot chooser uses it to prefer shots that split a
cluster of reds (or a pool rack) over shots that leave it tied up.
"""

from collections.abc import Iterable

from mr147.game_table import GameTable, Position
from mr147.shot_planner.pot_geometry import _can_enter_pocket, ghost_ball_position, path_is_clear
from mr147.snooker.table import distance
from mr147.table_memory.table_state import TableState

TIED_UP_GAP = 0.005  # metres: balls closer than this (edge to edge) count as touching


def ball_is_tied_up(ball: Position, others: list[Position], game: GameTable) -> bool:
    return any(distance(ball, o) < game.ball_diameter + TIED_UP_GAP for o in others)


def ball_is_pottable(ball: Position, others: list[Position], pockets: list[Position], game: GameTable) -> bool:
    """Is there a pocket the ball could go into, if the cue ball were in the right place?"""
    for pocket in pockets:
        if not _can_enter_pocket(ball, pocket, game):
            continue
        if not path_is_clear(ball, pocket, others, game.ball_diameter):
            continue
        ghost = ghost_ball_position(ball, pocket, game.ball_diameter)
        room_for_cue_ball = game.is_on_table(ghost) and all(
            distance(ghost, o) >= game.ball_diameter for o in others
        )
        if room_for_cue_ball:
            return True
    return False


def table_openness(
    table: TableState, balls_to_clear: Iterable[str], pockets: list[Position], game: GameTable
) -> float:
    """0..1, see the module docstring. 1.0 when there is nothing left to clear."""
    balls = [b for b in balls_to_clear if b in table.positions]
    if not balls:
        return 1.0
    total = 0.0
    for ball_id in balls:
        position = table.positions[ball_id]
        others = table.positions_except(ball_id, game.cue_ball)
        if ball_is_tied_up(position, others, game):
            continue
        total += 1.0 if ball_is_pottable(position, others, pockets, game) else 0.5
    return total / len(balls)
