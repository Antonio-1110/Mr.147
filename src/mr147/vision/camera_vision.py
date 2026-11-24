"""The robot's vision: camera + detector + geometry, speaking the vision contract.

CameraVision implements mr147.interfaces.vision.VisionSystem:

    observe()                     take a picture, find the balls, say where they are
    request_better_view(request)  walk to a better spot for the places memory is unsure
                                  about, look again, and merge both looks

It works the same with the real robot and in simulation: anything with a
`capture()` and `move_to()` is a camera (see RobotCamera), and anything with a
`detect()` is a detector (see vision/detector.py).

It works for snooker and pool (pass spec=POOL). Ball names are chosen for all
balls together, so no name is used more often than the game allows (identify.py).

Vision filters what it can decide on its own, so the robot doesn't walk round for
nothing: balls that aren't on our table (the next table over) are dropped, and
names of balls the memory says are potted are avoided.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from mr147.interfaces.vision import Detection, DetectionFrame, ViewRequest
from mr147.snooker.balls import Colour
from mr147.table_memory.ball_sets import BALL_SETS

from .detector import Detector, PixelDetection
from .identify import assign_names, name_for_memory, name_limits
from .locate import PinholeCamera, PoseFit, ball_position, refine_pose
from .scene import CameraPose
from .table import SNOOKER, TableSpec


class RobotCamera(Protocol):
    """The robot's head camera."""

    def capture(self) -> tuple[np.ndarray, CameraPose]:
        """An RGB picture, and where the robot's body thinks the camera is (roughly)."""
        ...

    def move_to(self, pose: CameraPose) -> None:
        """Walk / turn the head so the camera ends up at this pose."""
        ...

    @property
    def size(self) -> tuple[int, int]:
        """Picture width and height in pixels."""
        ...


@dataclass
class VisionSettings:
    min_pockets_for_pose: int = 3  # fewer pockets than this: trust the robot's own pose
    no_pose_fix_penalty: float = 0.6  # confidence multiplier when the pose couldn't be checked
    far_distance: float = 3.0  # metres; balls further away get a little less confidence
    same_ball_distance: float = 0.045  # metres; two looks this close are the same ball
    # (two different balls can never be closer than one ball width, 52.5 mm)
    standing_eye_height: float = 1.6
    crouched_eye_height: float = 1.15
    step_back: float = 0.35  # how far from the rail the robot stands when looking closer
    # Each observe() looks from this many sides of the table and merges them, so a
    # ball hidden behind another from one side is still found.
    views_per_observe: int = 2
    # A ball whose centre is further than this outside the cushions isn't on our
    # table (positions are good to ~1 cm, so 4 cm is safely beyond any error).
    off_table_margin: float = 0.04


@dataclass
class Look:
    """Everything from one picture, kept for debugging and for drawing."""

    image: np.ndarray
    pixel_detections: list[PixelDetection]
    pose_fit: PoseFit
    frame: DetectionFrame
    pinned: bool  # are the positions tied to the table (pockets, or lined up with a look that was)?


class CameraVision:
    def __init__(self, camera: RobotCamera, detector: Detector, spec: TableSpec = SNOOKER,
                 settings: VisionSettings | None = None):
        self.camera = camera
        self.detector = detector
        self.spec = spec
        self.settings = settings or VisionSettings()
        self.last_look: Look | None = None
        self.clock = 0.0
        self._last_pose: CameraPose | None = None
        self.looks: list[Look] = []
        self.in_play: Counter | None = None  # what the memory says is still on the table
        self.dropped_off_table = 0  # balls ignored in the last look because they weren't on our table

    # ---- VisionSystem ----------------------------------------------------------------

    def observe(self, in_play: Sequence[str] | None = None) -> DetectionFrame:
        self.in_play = Counter(map(str, in_play)) if in_play is not None else None
        self.last_look = None  # a new observation starts fresh
        look = self._look("main")
        self.looks = [look]
        frame = look.frame
        for k in range(1, self.settings.views_per_observe):
            # Walk round: the same view from the other side of the table, then merge.
            angle = 2 * math.pi * k / self.settings.views_per_observe
            frame = self._look_again(_walk_round(self.camera_pose, angle), "other side")
        return frame

    def request_better_view(self, request: ViewRequest) -> DetectionFrame:
        if request.in_play is not None:
            self.in_play = Counter(map(str, request.in_play))
        previous = self.last_look
        known = [d.position for d in previous.frame.detections] if previous else []
        pose = choose_view(request.look_closer, known, self.spec, self.settings)
        return self._look_again(pose, "closer look")

    @property
    def camera_pose(self) -> CameraPose:
        return getattr(self.camera, "pose", None) or self._last_pose

    def _look_again(self, pose: CameraPose, name: str) -> DetectionFrame:
        previous = self.last_look
        self.camera.move_to(pose)
        # The balls haven't moved since the last look, so the ones we already placed
        # help pin down where the camera is now, even if few pockets are in view.
        reference = previous.frame if previous is not None and previous.pinned else None
        look = self._look(name, reference)
        self.looks.append(look)
        if previous is None:
            return look.frame
        merged = merge_frames(previous.frame, previous.pose_fit.camera, look.frame,
                              look.pose_fit.camera, self.spec, self.settings, self.in_play)
        self.last_look = Look(look.image, look.pixel_detections, look.pose_fit, merged,
                              pinned=previous.pinned or look.pinned)
        return merged

    # ---- one picture -----------------------------------------------------------------

    def _look(self, name: str, reference: DetectionFrame | None = None) -> Look:
        image, rough_pose = self.camera.capture()
        self._last_pose = rough_pose
        w, h = image.shape[1], image.shape[0]
        found = self.detector.detect(image)
        rough = PinholeCamera.from_pose(rough_pose, w, h)
        known = match_known_balls(rough, found, reference, self.spec) if reference else []
        fit = refine_pose(rough, [d.box for d in found if d.name == "pocket"], self.spec, known_balls=known)
        pinned = fit.landmarks_used >= self.settings.min_pockets_for_pose
        frame = self.to_frame(found, fit, name, pinned)
        self.clock += 1.0
        look = Look(image, found, fit, frame, pinned=pinned)
        self.last_look = look
        return look

    def to_frame(self, found: list[PixelDetection], fit: PoseFit, name: str = "",
                 pose_ok: bool = True) -> DetectionFrame:
        s = self.settings
        cam = fit.camera
        placed = []  # (x, y, name -> score)
        on_table = BALL_SETS[self.spec.game].is_on_table
        self.dropped_off_table = 0
        for d in found:
            if d.name == "pocket":
                continue
            xy = ball_position(cam, d.box, self.spec.ball_r)
            if xy is None:
                continue
            if not on_table(xy, slack=s.off_table_margin):  # e.g. a ball on the next table
                self.dropped_off_table += 1
                continue
            conf = d.confidence * (1.0 if pose_ok else s.no_pose_fix_penalty)
            dist = float(np.linalg.norm(cam.position - np.array([xy[0], xy[1], self.spec.ball_r])))
            if dist > s.far_distance:  # far balls: a pixel covers more cloth
                conf *= max(0.8, s.far_distance / dist)
            factor = conf / d.confidence if d.confidence > 0 else 0.0
            scores = {name_for_memory(n, self.spec.game): v * factor for n, v in d.scores.items()}
            placed.append((xy[0], xy[1], scores))
        detections = name_balls(placed, self.spec.game, self.in_play)
        return DetectionFrame(tuple(detections), timestamp=self.clock, camera_view=name, game=self.spec.game)


def match_known_balls(rough: PinholeCamera, found: list[PixelDetection], reference: DetectionFrame,
                      spec: TableSpec, max_px: float | None = None) -> list:
    """Pair balls in this picture with balls already placed on the table.

    Only pairs that are clearly the same ball (same colour, the nearest by a wide
    margin) are kept, because a wrong pair would pull the camera the wrong way.
    """
    max_px = max_px or 0.06 * rough.width
    pred = []
    for d in reference.detections:
        b = rough.ball_box(d.x, d.y, spec.ball_r)
        if b is not None:
            pred.append((str(d.colour), np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), d.position))
    pairs = []
    for f in found:
        if f.name == "pocket":
            continue
        c = np.array(f.centre)
        f_name = name_for_memory(f.name, spec.game)
        dists = sorted((np.linalg.norm(c - pc), xy) for name, pc, xy in pred if name == f_name)
        if not dists or dists[0][0] > max_px:
            continue
        if len(dists) > 1 and dists[1][0] < 2.5 * dists[0][0] + 5:  # ambiguous
            continue
        pairs.append((f.box, dists[0][1]))
    return pairs


def _walk_round(pose: CameraPose, angle: float) -> CameraPose:
    """The same view, from a spot turned `angle` radians round the table centre."""
    c, s = math.cos(angle), math.sin(angle)

    def turn(p):
        return (c * p[0] - s * p[1], s * p[0] + c * p[1], p[2])

    return CameraPose(turn(pose.pos), turn(pose.look_at), pose.fov_h, pose.roll)


# --------------------------------------------------------------------------- merging


def merge_frames(old: DetectionFrame, old_cam: PinholeCamera, new: DetectionFrame,
                 new_cam: PinholeCamera, spec: TableSpec = SNOOKER,
                 settings: VisionSettings | None = None, in_play: Counter | None = None) -> DetectionFrame:
    """Combine two looks at the same (still) table into one.

    * Seen in both: average the two positions, trusting the more confident look more,
      and average the two looks' scores for each name (a ball called "8" once and
      "4" once ends up unsure, so memory will ask for another look).
    * Only in the new look: keep it.
    * Only in the old look: keep it if the new camera couldn't see that spot (out of
      the picture, or behind another ball). If the new camera had a clear view and
      didn't see it, it was probably a mistake.

    Balls are matched by position only (two balls can't be closer than a ball
    width), then all balls are named again together.
    """
    s = settings or VisionSettings()
    used_old: set[int] = set()
    out: list[tuple[float, float, dict]] = []
    for n in new.detections:
        best, best_d = None, s.same_ball_distance
        for i, o in enumerate(old.detections):
            if i in used_old:
                continue
            d = math.dist(o.position, n.position)
            if d < best_d:
                best, best_d = i, d
        if best is None:
            out.append((n.x, n.y, _scores(n)))
            continue
        used_old.add(best)
        o = old.detections[best]
        wo, wn = max(o.confidence, 1e-3), max(n.confidence, 1e-3)
        x = (o.x * wo + n.x * wn) / (wo + wn)
        y = (o.y * wo + n.y * wn) / (wo + wn)
        so, sn = _scores(o), _scores(n)
        out.append((x, y, {k: (so.get(k, 0.0) + sn.get(k, 0.0)) / 2 for k in so.keys() | sn.keys()}))
    new_balls = [np.array([d.x, d.y, spec.ball_r]) for d in new.detections]
    for i, o in enumerate(old.detections):
        if i in used_old:
            continue
        p = np.array([o.x, o.y, spec.ball_r])
        in_view = new_cam.sees(p, margin=-20)
        hidden = in_view and _line_blocked(new_cam.position, p, new_balls, spec.ball_r)
        if not in_view or hidden:
            out.append((o.x, o.y, _scores(o)))
    game = new.game or spec.game
    return DetectionFrame(tuple(name_balls(out, game, in_play)), timestamp=new.timestamp, camera_view="merged", game=game)


def _scores(d: Detection) -> dict[str, float]:
    return dict(d.alternatives) if d.alternatives else {str(d.colour): d.confidence}


def name_balls(placed: list[tuple[float, float, dict]], game: str,
               in_play: Counter | None = None) -> list[Detection]:
    """Placed balls with name scores -> Detections, each name used no more than allowed.

    in_play: what the memory believes is still on the table; other names are avoided.
    """
    expected = None
    if in_play is not None:
        expected = Counter(in_play)
        expected[BALL_SETS[game].cue_ball] = 1  # the cue ball is always in play
    names = assign_names([sc for _, _, sc in placed], name_limits(game), expected)
    out = []
    for (x, y, sc), (name, score) in zip(placed, names):
        alternatives = tuple(sorted(((k, round(v, 3)) for k, v in sc.items()), key=lambda kv: -kv[1]))
        label = Colour(name) if game == "snooker" and name in Colour.__members__.values() else name
        out.append(Detection(label, x, y, round(score, 3), alternatives))
    return out


# --------------------------------------------------------------------------- where to stand


def candidate_views(spec: TableSpec, settings: VisionSettings) -> list[tuple[float, float]]:
    """Spots around the table where the robot can stand (every 40 cm along each side)."""
    s = settings
    half_l = spec.length / 2 + spec.cushion_w + spec.rail_w + s.step_back
    half_w = spec.width / 2 + spec.cushion_w + spec.rail_w + s.step_back
    spots = []
    for x in np.arange(-half_l + 0.2, half_l - 0.19, 0.4):
        spots += [(float(x), -half_w), (float(x), half_w)]
    for y in np.arange(-half_w + 0.2, half_w - 0.19, 0.4):
        spots += [(-half_l, float(y)), (half_l, float(y))]
    return spots


def _line_blocked(eye: np.ndarray, target: np.ndarray, balls: list[np.ndarray], r: float) -> bool:
    """Does the line of sight from the eye to the target pass through another ball?"""
    d = target - eye
    length = np.linalg.norm(d)
    d /= length
    for b in balls:
        if np.linalg.norm(b - target) < 1e-3:
            continue
        t = float(np.dot(b - eye, d))
        if 0 < t < length - r and np.linalg.norm(eye + t * d - b) < r * 1.05:
            return True
    return False


def choose_view(look_closer, known_balls, spec: TableSpec = SNOOKER,
                settings: VisionSettings | None = None, fov_h: float = 65.0) -> CameraPose:
    """Pick where to stand to see the uncertain spots clearly.

    Prefers spots close to the places in question, from which no other ball
    blocks the line of sight.
    """
    s = settings or VisionSettings()
    targets = [np.array([p[0], p[1], spec.ball_r]) for p in look_closer] or [np.array([0.0, 0.0, 0.0])]
    centre = np.mean(targets, axis=0)
    balls = [np.array([p[0], p[1], spec.ball_r]) for p in known_balls]
    pockets = [np.array([px, py, 0.0]) for px, py, _ in spec.pockets]
    best, best_score = None, math.inf
    for x, y in candidate_views(spec, s):
        for eye_h in (s.standing_eye_height, s.crouched_eye_height):
            eye = np.array([x, y, eye_h - spec.cloth_height])
            blocked = sum(_line_blocked(eye, t, balls, spec.ball_r) for t in targets)
            dist = np.mean([np.linalg.norm(eye - t) for t in targets])
            # At least 3 pockets in the picture let us check the camera position.
            cam = PinholeCamera.looking_at(eye, centre, fov_h, 1280, 720)
            n_pockets = sum(cam.sees(p, margin=-40) and np.linalg.norm(p - eye) < 3.0 for p in pockets)
            # Looking along the cloth at a low angle squashes the balls together: prefer
            # standing unless crouching is clearly closer.
            score = (10 * blocked + dist + (0.5 if eye_h < s.standing_eye_height else 0.0)
                     + (1.0 if n_pockets < 3 else 0.0))
            if score < best_score:
                best, best_score = eye, score
    return CameraPose(tuple(float(v) for v in best), tuple(float(v) for v in centre), fov_h)
