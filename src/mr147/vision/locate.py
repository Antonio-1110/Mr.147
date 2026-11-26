"""From boxes in a picture to positions on the table.

The detector gives boxes in pixels. To turn a box into a spot on the table we
need to know exactly where the camera is and which way it points:

  1. The robot's body gives a rough camera position (a few cm, a degree or two).
  2. The pockets are at known places on the table. We nudge the rough camera
     position until the pockets we see line up with where they should appear.
  3. With the camera pinned down, each ball box gives one table position: the
     spot where a ball would have to sit to fill exactly that box.

Pockets alone can't tell one end of the table from the other (the table looks
the same turned round), which is why step 1 is needed.

Camera axes follow OpenCV: x to the right of the picture, y down, z forward.
Pixel coordinates run from 0 to width / height (0 is the left/top edge).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .table import SNOOKER, TableSpec, off_cloth


@dataclass(frozen=True)
class PinholeCamera:
    """A camera: where it is (R, t) and its lens (fx, fy, cx, cy)."""

    R: np.ndarray  # 3x3, world -> camera
    t: np.ndarray  # 3, world -> camera
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    # ---- building ---------------------------------------------------------------

    @classmethod
    def looking_at(cls, pos, look_at, fov_h_deg: float, width: int, height: int,
                   roll_deg: float = 0.0) -> "PinholeCamera":
        """Same convention as the renderer's CameraPose (z up, roll about the view axis)."""
        pos = np.asarray(pos, float)
        f = np.asarray(look_at, float) - pos
        f /= np.linalg.norm(f)
        right = np.cross(f, [0.0, 0.0, 1.0])
        if np.linalg.norm(right) < 1e-9:  # looking straight down
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        r = -math.radians(roll_deg)  # Panda rolls the other way
        right, up = math.cos(r) * right + math.sin(r) * up, -math.sin(r) * right + math.cos(r) * up
        R = np.stack([right, -up, f])
        fx = (width / 2) / math.tan(math.radians(fov_h_deg) / 2)
        return cls(R, -R @ pos, fx, fx, width / 2, height / 2, width, height)

    @classmethod
    def from_pose(cls, pose, width: int, height: int) -> "PinholeCamera":
        """From a vision.scene.CameraPose."""
        return cls.looking_at(pose.pos, pose.look_at, pose.fov_h, width, height, pose.roll)

    def with_extrinsics(self, rvec: np.ndarray, t: np.ndarray) -> "PinholeCamera":
        return PinholeCamera(Rotation.from_rotvec(rvec).as_matrix(), np.asarray(t, float),
                             self.fx, self.fy, self.cx, self.cy, self.width, self.height)

    @property
    def rvec(self) -> np.ndarray:
        return Rotation.from_matrix(self.R).as_rotvec()

    @property
    def position(self) -> np.ndarray:
        return -self.R.T @ self.t

    # ---- projecting -------------------------------------------------------------

    def to_camera(self, pts: np.ndarray) -> np.ndarray:
        return np.atleast_2d(pts) @ self.R.T + self.t

    def project(self, pts: np.ndarray) -> np.ndarray:
        """World points (N x 3) to pixels (N x 2). Points behind the camera give NaN."""
        c = self.to_camera(pts)
        z = np.where(c[:, 2] > 1e-6, c[:, 2], np.nan)
        return np.stack([self.fx * c[:, 0] / z + self.cx, self.fy * c[:, 1] / z + self.cy], 1)

    def ray(self, pixel) -> np.ndarray:
        """Unit direction, in world coordinates, of the line of sight through a pixel."""
        u, v = pixel
        d = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        d = self.R.T @ d
        return d / np.linalg.norm(d)

    def sees(self, pt, margin: float = 0.0) -> bool:
        """Is this world point inside the picture (and in front of the camera)?"""
        p = self.project(np.asarray(pt, float))[0]
        return bool(np.all(np.isfinite(p)) and -margin <= p[0] <= self.width + margin
                    and -margin <= p[1] <= self.height + margin)

    # ---- shapes -----------------------------------------------------------------

    def sphere_outline(self, centre, radius: float, n: int = 48) -> np.ndarray | None:
        """Outline of a ball in the picture: the circle where lines of sight graze the ball."""
        c = np.asarray(centre, float)
        d = c - self.position
        dist = np.linalg.norm(d)
        if dist <= radius * 1.01:
            return None
        u = d / dist
        mid = c - (radius**2 / dist) * u
        rad = radius * math.sqrt(1 - (radius / dist) ** 2)
        a = np.cross(u, [0, 0, 1.0])
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(u, [1.0, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(u, a)
        th = np.linspace(0, 2 * math.pi, n, endpoint=False)
        px = self.project(mid + rad * (np.cos(th)[:, None] * a + np.sin(th)[:, None] * b))
        return None if np.isnan(px).any() else px

    def ball_box(self, x: float, y: float, radius: float) -> np.ndarray | None:
        """Box (x0, y0, x1, y1) a ball centred at (x, y) on the cloth fills, before clipping."""
        o = self.sphere_outline((x, y, radius), radius)
        return None if o is None else np.concatenate([o.min(0), o.max(0)])

    def pocket_box(self, px: float, py: float, pr: float, spec: TableSpec) -> np.ndarray | None:
        """Box of a pocket: the hole in the cloth plus its cut-out in the rail (as labelled)."""
        th = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        # Same shape the renderer draws and labels: the rail-level cut-out stops at the
        # cushion line (it doesn't hover over the cloth), see table.off_cloth.
        snap = off_cloth((px, py), spec.length / 2, spec.width / 2)
        hole = [(px + pr * c, py + pr * s, 0.0) for c, s in zip(np.cos(th), np.sin(th))]
        rail = [(px + x, py + y, spec.rail_h)
                for x, y in (snap(1.1 * pr * c, 1.1 * pr * s) for c, s in zip(np.cos(th), np.sin(th)))]
        p = self.project(np.array(hole + rail))
        return None if np.isnan(p).any() else np.concatenate([p.min(0), p.max(0)])


# --------------------------------------------------------------------------- boxes


def _clip(box: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.array([max(box[0], 0), max(box[1], 0), min(box[2], w), min(box[3], h)])


def _touches_edge(box, w: int, h: int, margin: float = 1.0) -> bool:
    return box[0] <= margin or box[1] <= margin or box[2] >= w - margin or box[3] >= h - margin


def ball_position(cam: PinholeCamera, box, radius: float) -> tuple[float, float] | None:
    """Where on the table a ball must sit to fill this box.

    Uses the box edges that aren't cut off by the picture's border, so a ball
    half out of the picture still gets a fair position.
    """
    box = np.asarray(box, float)
    w, h = cam.width, cam.height
    keep = np.array([box[0] > 1, box[1] > 1, box[2] < w - 1, box[3] < h - 1])
    if keep.sum() < 2:
        return None
    # First guess: the line of sight through the box centre meets the plane of ball centres.
    ray = cam.ray(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
    if ray[2] >= -1e-6:
        return None
    s = (radius - cam.position[2]) / ray[2]
    guess = cam.position + s * ray

    def residual(xy):
        b = cam.ball_box(xy[0], xy[1], radius)
        if b is None:
            return np.full(int(keep.sum()), 1e3)
        return (b - box)[keep]

    res = least_squares(residual, guess[:2], method="lm" if keep.sum() >= 2 else "trf", x_scale=0.05)
    return float(res.x[0]), float(res.x[1])


# --------------------------------------------------------------------------- camera pose


@dataclass
class PoseFit:
    camera: PinholeCamera
    pockets_used: int
    pixel_error: float  # average mismatch after fitting, in pixels
    matched: dict  # observed pocket index -> table pocket index
    balls_used: int = 0  # balls with a known position that helped pin the camera

    @property
    def landmarks_used(self) -> int:
        return self.pockets_used + self.balls_used


@dataclass(frozen=True)
class BodySense:
    """How well the robot's own body knows where its head camera is (one standard deviation).

    Height above the cloth and roll (head tipped sideways) come from the robot's joints
    and balance sensor, so they are known well. With them, two pockets are enough to
    place the table: 2 pockets give 4 clues for the 4 numbers left (left/right,
    forward/back, turn, tilt). Further pockets let the pictures overrule the body.
    """

    height_m: float = 0.02
    roll_deg: float = 0.5
    floor_m: float = 0.05  # where on the floor the robot stands (weak: the pockets decide)
    aim_deg: float = 2.0  # which way the head points (weak)


def refine_pose(rough: PinholeCamera, pocket_boxes: list, spec: TableSpec = SNOOKER,
                max_match_px: float | None = None,
                known_balls: list[tuple[tuple, tuple[float, float]]] = (),
                body: BodySense | None = BodySense(), min_landmarks: int = 2) -> PoseFit:
    """Nudge a rough camera pose so known things land where we see them.

    Known things are the pockets (always at the same place on the table), and
    optionally balls whose table position we already measured from a better
    look (known_balls: pairs of (box seen now, (x, y) on the table)).

    The body's own sense of height and roll (BodySense) is weighed in too, so 2
    pockets are enough. Needs at least `min_landmarks` of them in view, with boxes
    not cut off by the picture's edge; otherwise the rough pose is returned
    unchanged. body=None and min_landmarks=3 give the old pockets-only fit.
    """
    w, h = rough.width, rough.height
    max_match_px = max_match_px or 0.12 * w
    obs = [np.asarray(b, float) for b in pocket_boxes]
    obs = [b for b in obs if not _touches_edge(b, w, h)]

    def predicted(cam, k):
        px, py, pr = spec.pockets[k]
        return cam.pocket_box(px, py, pr, spec)

    def centre(b):
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])

    # Match each observed pocket to the table pocket it most likely is.
    pred = {k: predicted(rough, k) for k in range(len(spec.pockets))}
    pred = {k: b for k, b in pred.items() if b is not None}
    best_cost, matched = math.inf, {}
    ks = list(pred)
    for perm in itertools.permutations(ks, min(len(obs), len(ks))):
        cost = sum(np.linalg.norm(centre(obs[i]) - centre(pred[k])) for i, k in enumerate(perm))
        if cost < best_cost:
            best_cost, matched = cost, dict(enumerate(perm))
    matched = {i: k for i, k in matched.items()
               if np.linalg.norm(centre(obs[i]) - centre(pred[k])) < max_match_px}
    balls = [(np.asarray(b, float), xy) for b, xy in known_balls if not _touches_edge(b, w, h)]
    if len(matched) + len(balls) < max(min_landmarks, 2 if body is not None else 3):
        return PoseFit(rough, len(matched), math.nan, matched, len(balls))

    x0 = np.concatenate([rough.rvec, rough.t])

    def residual(x):
        cam = rough.with_extrinsics(x[:3], x[3:])
        out = []
        for i, k in matched.items():
            b = predicted(cam, k)
            out.append(np.full(4, 1e3) if b is None else b - obs[i])
        for box, (bx, by) in balls:
            b = cam.ball_box(bx, by, spec.ball_r)
            out.append(np.full(4, 1e3) if b is None else b - box)
        if body is not None:
            # Stay near what the body says, in units of its uncertainty (one "pixel" per sigma).
            R = Rotation.from_rotvec(x[:3]).as_matrix()
            moved = -R.T @ x[3:] - rough.position
            turn = Rotation.from_matrix(R @ rough.R.T).as_rotvec()  # camera axes: tilt, turn, roll
            out.append(np.array([
                moved[0] / body.floor_m, moved[1] / body.floor_m, moved[2] / body.height_m,
                math.degrees(turn[0]) / body.aim_deg, math.degrees(turn[1]) / body.aim_deg,
                math.degrees(turn[2]) / body.roll_deg]))
        return np.concatenate(out)

    res = least_squares(residual, x0, x_scale=np.array([0.01] * 3 + [0.02] * 3), loss="soft_l1", f_scale=3.0)
    cam = rough.with_extrinsics(res.x[:3], res.x[3:])
    err = float(np.median(np.abs(res.fun[:-6] if body is not None else res.fun)))
    return PoseFit(cam, len(matched), err, matched, len(balls))
