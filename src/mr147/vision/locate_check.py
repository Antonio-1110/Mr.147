"""How well does the robot know where the balls are?  (mr147-locate-check)

Renders random tables seen from the robot's head, where we know exactly where
every ball is. Then it does what the robot does:

    picture -> boxes -> camera pose from the pockets -> each box to a spot on the table

and measures how far each spot is from the true one, in millimetres.

It reports three numbers per ball, so we can see where the error comes from:

    robot     what the robot really uses (pose fixed by the pockets it sees)
    perfect   if the robot knew its camera pose exactly (only the box is wrong)
    no fix    if the robot trusted its body's rough pose (no pocket correction)

    uv run mr147-locate-check --game pool --tables 4                # quick try
    uv run mr147-locate-check --game pool --tables 60               # real run
    uv run mr147-locate-check --game pool --model models/pool.pt    # trained finder, not simulator boxes

Writes results/<name>/summary.txt, balls.csv, error-map.jpg (table from above)
and views.jpg (a few pictures with true and estimated ball outlines).
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import table as T
from .augment import camera_effects
from .generate import label_view
from .layouts import random_layout
from .locate import PinholeCamera, ball_position, fuse_looks, refine_pose
from .scene import CameraPose, random_camera, random_look


@dataclass
class Settings:
    game: str = "pool"
    tables: int = 4
    views: int = 4  # camera spots per table
    pose_error_m: float = 0.02  # how wrong the robot's body is about where its head is (per axis)
    roll_error_deg: float = 0.5
    box_noise_px: float = 0.5  # simulator boxes: wobble, like a good detector
    seed: int = 0
    model: str | None = None  # trained YOLO finder; None = boxes straight from the simulator
    device: str | None = None
    refine: bool = False  # tighten each box to the ball's real edge (refine.py)


@dataclass
class Result:
    rows: list[dict] = field(default_factory=list)  # one per ball found
    views: list[dict] = field(default_factory=list)  # one per picture
    refined: int = 0  # boxes the edge refinement tightened (the rest kept the finder's box)
    missed: int = 0  # balls in clear view the finder didn't find
    false_finds: int = 0  # "balls" that aren't there
    examples: list[np.ndarray] = field(default_factory=list)
    # Balls seen (3+ pockets) in two or more pictures of the same table, combined:
    # (error of plain average mm, error weighing each look along its line of sight mm, best single look mm)
    combined: list[tuple[float, float, float]] = field(default_factory=list)


def rough_pose(pose: CameraPose, rng: np.random.Generator, e: float, roll: float) -> CameraPose:
    """What the robot's body thinks: the true pose plus a little error (as SimCamera does)."""
    return CameraPose(tuple(np.array(pose.pos) + rng.normal(0, e, 3)),
                      tuple(np.array(pose.look_at) + rng.normal(0, e, 3)),
                      pose.fov_h, pose.roll + rng.normal(0, roll))


def simulator_boxes(info: list[dict], pockets: list[dict], noise: float, w: int, h: int,
                    rng: np.random.Generator) -> tuple[list, list]:
    """(ball index, box) pairs and pocket boxes, as labelled, with a little noise."""
    def wobble(b):
        return tuple(np.clip(np.array(b) + rng.normal(0, noise, 4), 0, [w, h, w, h]))

    balls = [(i, wobble(e["box"])) for i, e in enumerate(info) if e["box"] is not None]
    return balls, [wobble(p["box"]) for p in pockets if p["box"] is not None]


def match_to_truth(found_boxes: list, info: list[dict]) -> tuple[list, int, int]:
    """Pair each found ball box with the true ball it sits on.

    Returns ((ball index, box) pairs, missed, false finds). Missed only counts
    balls the labels say are in clear view.
    """
    truth = [i for i, e in enumerate(info) if "centre_px" in e]
    if not found_boxes or not truth:
        return [], sum(e["box"] is not None for e in info), len(found_boxes)
    fc = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in found_boxes])
    tc = np.array([info[i]["centre_px"] for i in truth])
    cost = np.linalg.norm(fc[:, None] - tc[None], axis=2)
    r, c = linear_sum_assignment(cost)
    pairs, used = [], set()
    for a, b in zip(r, c):
        size = max(found_boxes[a][2] - found_boxes[a][0], found_boxes[a][3] - found_boxes[a][1])
        if cost[a, b] < 0.6 * size:
            pairs.append((truth[b], found_boxes[a]))
            used.add(truth[b])
    missed = sum(1 for i, e in enumerate(info) if e["box"] is not None and i not in used)
    return pairs, missed, len(found_boxes) - len(pairs)


def run(renderer, s: Settings, detector=None, keep_examples: int = 6) -> Result:
    spec = T.SPECS[s.game]
    rng = np.random.default_rng(s.seed)
    w, h = renderer.w, renderer.h
    res = Result()
    for ti in range(s.tables):
        seen: dict[int, list] = {}  # ball index -> [(position, camera position), ...] from trusted looks
        balls = random_layout(rng, spec)
        renderer.build(balls, random_look(rng, spec), spec)
        for vi in range(s.views):
            pose = random_camera(rng, spec)
            renderer.set_camera(pose)
            rgb, ids = renderer.render_frame()
            _, info, pocket_info = label_view(renderer, balls, ids, spec)
            image = camera_effects(rgb, rng)
            if detector is None:
                found, pocket_boxes = simulator_boxes(info, pocket_info, s.box_noise_px, w, h, rng)
                missed = false = 0
            else:
                dets = detector.detect(image)
                pocket_boxes = [d.box for d in dets if d.name == "pocket"]
                found, missed, false = match_to_truth([d.box for d in dets if d.name != "pocket"], info)
            res.missed += missed
            res.false_finds += false
            if s.refine:
                from .refine import refine_ball_box

                tightened = [(i, *refine_ball_box(image, box)) for i, box in found]
                res.refined += sum(ok for _, _, ok in tightened)
                found = [(i, tuple(b)) for i, b, _ in tightened]

            true_cam = PinholeCamera.from_pose(pose, w, h)
            rough_cam = PinholeCamera.from_pose(rough_pose(pose, rng, s.pose_error_m, s.roll_error_deg), w, h)
            fit = refine_pose(rough_cam, pocket_boxes, spec)
            robot_cam = fit.camera  # unchanged rough pose when fewer than 3 pockets are usable
            pose_err = float(np.linalg.norm(robot_cam.position - true_cam.position))
            res.views.append(dict(table=ti, view=vi, pockets=fit.pockets_used, balls=len(found),
                                  eye_height=pose.pos[2] + spec.cloth_height,
                                  pose_error_mm=1000 * pose_err,
                                  rough_error_mm=1000 * float(np.linalg.norm(rough_cam.position - true_cam.position))))
            shown = []
            for i, box in found:
                b = balls[i]
                xy = {k: ball_position(cam, box, spec.ball_r)
                      for k, cam in (("robot", robot_cam), ("perfect", true_cam), ("nofix", rough_cam))}
                if xy["robot"] is None:
                    continue
                err = {k: (1000 * math.dist(v, (b.x, b.y)) if v is not None else math.nan) for k, v in xy.items()}
                e = info[i]
                res.rows.append(dict(
                    table=ti, view=vi, ball=b.label, true_x=round(b.x, 4), true_y=round(b.y, 4),
                    est_x=round(xy["robot"][0], 4), est_y=round(xy["robot"][1], 4),
                    error_mm=round(err["robot"], 2), error_perfect_pose_mm=round(err["perfect"], 2),
                    error_no_fix_mm=round(err["nofix"], 2),
                    distance_m=round(float(np.linalg.norm(true_cam.position - [b.x, b.y, spec.ball_r])), 3),
                    box_px=round(float(box[2] - box[0]), 1), visible=e["visible_frac"],
                    in_picture=e.get("in_frame_frac", 1.0), pockets=fit.pockets_used))
                shown.append((b, xy["robot"]))
                if fit.pockets_used >= 3:
                    seen.setdefault(i, []).append((xy["robot"], robot_cam.position))
            if len(res.examples) < keep_examples:
                res.examples.append(draw_view(image, true_cam, robot_cam, shown, spec, ti, vi, fit.pockets_used))
        for i, looks in seen.items():
            if len(looks) < 2:
                continue
            true_xy = (balls[i].x, balls[i].y)
            mean = np.mean([p for p, _ in looks], axis=0)
            res.combined.append((1000 * math.dist(mean, true_xy),
                                 1000 * math.dist(fuse_looks(looks, spec.ball_r), true_xy),
                                 1000 * min(math.dist(p, true_xy) for p, _ in looks)))
    return res


# --------------------------------------------------------------------------- report


def _stats(v: np.ndarray) -> str:
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "   (none)"
    return (f"{np.median(v):6.1f} {np.percentile(v, 90):6.1f} {v.max():7.1f}"
            f"   {100 * np.mean(v < 5):5.0f}%   {100 * np.mean(v < 10):5.0f}%   {len(v):5d}")


HEAD = "                              middle  9 in 10  worst  under 5mm  under 10mm  balls"


def summary(res: Result, s: Settings, w: int, h: int, seconds: float) -> str:
    rows = res.rows
    col = {k: np.array([r[k] for r in rows], float) for k in rows[0] if k != "ball"} if rows else {}
    finder = f"trained finder {s.model}" if s.model else f"simulator boxes (+{s.box_noise_px} px wobble)"
    if s.refine:
        finder += ", then tightened to the ball's edge"
    out = [
        f"Ball position check - {s.game}, {s.tables} tables x {s.views} views, {w}x{h} pictures",
        f"Boxes from: {finder}",
        f"Robot's own idea of its head: off by ~{s.pose_error_m * 100:.0f} cm per direction, "
        f"{s.roll_error_deg} deg tilt",
        f"Took {seconds / 60:.1f} min",
        "",
        "Error = distance between where the robot thinks a ball is and where it really is (mm).",
        "A pool ball is 57 mm wide.",
        "",
    ]
    if not rows:
        return "\n".join(out + ["No balls were placed."])
    good = col["pockets"] >= 3
    out += [HEAD,
            "3+ pockets (robot trusts)    " + _stats(col["error_mm"][good]),
            "  if its pose were perfect   " + _stats(col["error_perfect_pose_mm"][good]),
            "All pictures                 " + _stats(col["error_mm"]),
            "  without the pocket fix     " + _stats(col["error_no_fix_mm"]),
            ""]
    if s.refine:
        out += [f"Edge tightening worked on {res.refined} of {len(rows)} balls (the rest kept their box).", ""]
    if s.model:
        out += [f"Finder: missed {res.missed} balls in clear view, {res.false_finds} false balls.", ""]
    if res.combined:
        c = np.array(res.combined)
        out += [f"Balls seen (3+ pockets) in 2 or more pictures of the same table, combined:",
                "  plain average              " + _stats(c[:, 0]),
                "  weighing lines of sight    " + _stats(c[:, 1]),
                "  best single look (cheat)   " + _stats(c[:, 2]), ""]
    out.append("By distance from the camera (3+ pockets):")
    d = col["distance_m"]
    for lo, hi in ((0, 1.5), (1.5, 2.5), (2.5, 9)):
        m = (d >= lo) & (d < hi) & good
        out.append(f"  {lo:.1f}-{hi:.1f} m".ljust(30) + _stats(col["error_mm"][m]) if hi < 9
                   else f"  over {lo:.1f} m".ljust(30) + _stats(col["error_mm"][m]))
    out.append("By pockets used to fix the pose:")
    p = col["pockets"]
    for lo, hi, name in ((0, 3, "0-2 (would look again)"), (3, 4, "3"), (4, 9, "4 or more")):
        m = (p >= lo) & (p < hi)
        out.append(f"  {name}".ljust(30) + _stats(col["error_mm"][m]))
    out.append("By how much of the ball shows (3+ pockets):")
    vis, inpic = col["visible"], col["in_picture"]
    for m, name in ((vis >= 0.95, "fully in view"), ((vis < 0.95) & (inpic >= 0.95), "partly behind another ball"),
                    (inpic < 0.95, "cut by the picture edge")):
        out.append(f"  {name}".ljust(30) + _stats(col["error_mm"][m & good]))
    v = res.views
    pe = np.array([x["pose_error_mm"] for x in v])
    fixed = np.array([x["pockets"] >= 3 for x in v])
    out += ["",
            f"Pictures: {len(v)}. Pose fixed by pockets in {fixed.sum()} ({100 * fixed.mean():.0f}%).",
            f"Camera position error after the fix: middle {np.median(pe[fixed]) if fixed.any() else math.nan:.1f} mm"
            f" (before: {np.median([x['rough_error_mm'] for x in v]):.0f} mm)."]
    worst = sorted((r for r in rows if r["pockets"] >= 3), key=lambda r: -r["error_mm"])[:5]
    out += ["", "Worst balls (3+ pockets):"] + [
        f"  table {r['table']} view {r['view']} ball {r['ball']}: {r['error_mm']:.0f} mm, {r['distance_m']:.1f} m away, "
        f"{r['pockets']} pockets, {100 * r['visible']:.0f}% showing" for r in worst]
    return "\n".join(out)


def draw_view(image, true_cam, robot_cam, shown, spec, ti, vi, pockets) -> np.ndarray:
    """The picture with each ball's true outline (green) and where the robot put it (red)."""
    img = np.ascontiguousarray(image[..., ::-1])
    for b, (x, y) in shown:
        for cam, xy, colour in ((true_cam, (b.x, b.y), (0, 255, 0)), (true_cam, (x, y), (0, 0, 255))):
            o = cam.sphere_outline((xy[0], xy[1], spec.ball_r), spec.ball_r)
            if o is not None:
                cv2.polylines(img, [np.round(o).astype(np.int32)], True, colour, 1, cv2.LINE_AA)
    cv2.putText(img, f"table {ti} view {vi}: {pockets} pockets", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def error_map(rows: list[dict], spec: T.TableSpec, scale: int = 300) -> np.ndarray:
    """Table from above: each ball as a dot coloured by its error (green < 5 mm, yellow < 10, red more)."""
    m = 40
    W, H = int(spec.length * scale) + 2 * m, int(spec.width * scale) + 2 * m + 40
    img = np.full((H, W, 3), (40, 90, 40), np.uint8)

    def px(x, y):
        return int(m + (x + spec.length / 2) * scale), int(m + (spec.width / 2 - y) * scale)

    cv2.rectangle(img, px(-spec.length / 2, spec.width / 2), px(spec.length / 2, -spec.width / 2), (200, 200, 200), 1)
    for x, y, r in spec.pockets:
        cv2.circle(img, px(x, y), int(r * scale), (0, 0, 0), -1)
    for r in sorted(rows, key=lambda r: r["error_mm"]):
        e = r["error_mm"]
        colour = (80, 220, 80) if e < 5 else (40, 220, 240) if e < 10 else (60, 60, 240)
        cv2.circle(img, px(r["true_x"], r["true_y"]), 4, colour, -1)
    cv2.putText(img, "green under 5 mm, yellow 5-10 mm, red over 10 mm", (m, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def sheet(images: list[np.ndarray], cols: int = 2, width: int = 960) -> np.ndarray:
    small = [cv2.resize(i, (width, int(i.shape[0] * width / i.shape[1]))) for i in images]
    while len(small) % cols:
        small.append(np.zeros_like(small[0]))
    return np.vstack([np.hstack(small[k:k + cols]) for k in range(0, len(small), cols)])


def write(res: Result, s: Settings, out: Path, w: int, h: int, seconds: float) -> str:
    out.mkdir(parents=True, exist_ok=True)
    text = summary(res, s, w, h, seconds)
    (out / "summary.txt").write_text(text + "\n")
    if res.rows:
        with open(out / "balls.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(res.rows[0]))
            wr.writeheader()
            wr.writerows(res.rows)
        cv2.imwrite(str(out / "error-map.jpg"), error_map(res.rows, T.SPECS[s.game]))
    if res.examples:
        cv2.imwrite(str(out / "views.jpg"), sheet(res.examples), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return text


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", choices=sorted(T.SPECS), default="pool")
    ap.add_argument("--tables", type=int, default=4)
    ap.add_argument("--views", type=int, default=4, help="camera spots per table")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--pose-error", type=float, default=0.02, help="metres; robot's error about its head")
    ap.add_argument("--roll-error", type=float, default=0.5, help="degrees")
    ap.add_argument("--box-noise", type=float, default=0.5, help="pixels; simulator boxes only")
    ap.add_argument("--model", help="trained finder (YOLO .pt); default: boxes from the simulator")
    ap.add_argument("--device", help="e.g. 0 for the graphics card, cpu")
    ap.add_argument("--refine", action="store_true", help="tighten each box to the ball's edge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="default: results/locate-<game>[-model]")
    a = ap.parse_args(argv)

    from .scene import Renderer

    s = Settings(a.game, a.tables, a.views, a.pose_error, a.roll_error, a.box_noise, a.seed, a.model, a.device,
                 a.refine)
    detector = None
    if a.model:
        from .detector import YoloDetector

        detector = YoloDetector(a.model, device=a.device)
    out = a.out or Path(f"results/locate-{a.game}" + ("-model" if a.model else ""))
    r = Renderer(a.width, a.height, seed=a.seed)
    t0 = time.time()
    res = run(r, s, detector)
    print(write(res, s, out, a.width, a.height, time.time() - t0))
    print(f"\nWrote {out}/summary.txt, balls.csv, error-map.jpg, views.jpg")


if __name__ == "__main__":
    main()
