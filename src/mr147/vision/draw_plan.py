"""Pictures of what the robot is about to do, made with the same 3D table.

    picture = draw_plan(renderer, table_before, plan, pockets)

Two views side by side:
  * from above, so a person can read the whole shot at a glance
    (only for us to look at; the robot itself never sees the table from above)
  * from behind the cue ball, the way a player lines up the shot

Drawn on the cloth:
  * yellow line:  the ball being potted, to its pocket
  * white line:   the cue ball's planned path, including after it hits the ball
  * white ring:   where the cue ball is expected to stop
  * small ring:   the "ghost ball", where the cue ball touches the object ball
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from mr147.shot_planner.pot_geometry import ghost_ball_position
from mr147.table_memory.table_state import TableState

from .scene import CameraPose, Look, Renderer
from .sim_camera import balls_from_table
from .table import SNOOKER, TableSpec


def presentation_look() -> Look:
    """Even, bright lighting so the drawing is easy to read."""
    lights = [(x, 0.0, 1.1, 1.2, (1.0, 1.0, 1.0)) for x in (-1.2, 0.0, 1.2)]
    return Look((0.05, 0.40, 0.10), (0.30, 0.14, 0.06), (0.15, 0.15, 0.15), (0.05, 0.05, 0.06),
                lights, 0.35, 0.0)


def overhead_view(spec: TableSpec = SNOOKER) -> CameraPose:
    return CameraPose((0.0, -0.25, 3.3), (0.0, 0.0, 0.0), 66.0)


def behind_cue_ball_view(cue: tuple[float, float], direction_deg: float) -> CameraPose:
    a = math.radians(direction_deg)
    back = 1.1
    pos = (cue[0] - back * math.cos(a), cue[1] - back * math.sin(a), 0.55)
    ahead = (cue[0] + 0.8 * math.cos(a), cue[1] + 0.8 * math.sin(a), 0.0)
    return CameraPose(pos, ahead, 60.0)


def _caption(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    for i, text in enumerate(lines):
        y = 32 + 30 * i
        cv2.putText(out, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_plan(renderer: Renderer, table: TableState, plan, pockets: list[tuple[float, float]],
              spec: TableSpec = SNOOKER, look: Look | None = None) -> np.ndarray:
    """RGB picture of the planned shot (overhead view | behind the cue ball)."""
    renderer.build(balls_from_table(table), look or presentation_look(), spec)
    path = list(plan.cue_ball_path) or [table.cue_ball]
    target = table.positions[plan.target]
    pocket = pockets[plan.pocket]
    r = spec.ball_r

    renderer.add_path([target, pocket], rgb=(1.0, 0.85, 0.1))
    renderer.add_path(path, rgb=(1, 1, 1))
    ghost = ghost_ball_position(target, pocket)  # where the cue ball meets the object ball
    renderer.add_ring(*ghost, r, rgb=(0.8, 0.8, 0.8), thickness=2)
    renderer.add_ring(*plan.cue_ball_stops_at, r * 1.3, rgb=(1, 1, 1), thickness=4)

    views = []
    for pose in (overhead_view(spec), behind_cue_ball_view(table.cue_ball, plan.shot.direction_deg)):
        renderer.set_camera(pose)
        rgb, _ = renderer.render_frame()
        views.append(rgb)
    picture = np.hstack(views)
    s = plan.shot
    spin = "follow" if s.height > 0.05 else "screw" if s.height < -0.05 else "plain ball"
    return _caption(picture, [
        f"pot {plan.target} into pocket {plan.pocket + 1}   chance {100 * plan.chance_of_potting:.0f}%",
        f"speed {s.speed:.1f} m/s, {spin}   cue ball stops at ({plan.cue_ball_stops_at[0]:+.2f}, "
        f"{plan.cue_ball_stops_at[1]:+.2f}) m   position {plan.position_quality:.2f}",
    ])
