"""Ball close-up studio: training close-ups for the reader, rendered directly.

The old way rendered whole 1920x1080 tables and then cut 128-pixel close-ups out of
them, so most of the rendering was thrown away and every new idea cost about an
hour of pictures. Here the camera stands where the robot's head would be and zooms
straight onto one ball, so each render is a small picture that is all ball:

    uv run mr147-studio --out data/studio --layouts 400
    uv run --extra train mr147-train-reader --data data/studio

What stays the same as the full-table pictures: the same tables, cloths, lamps,
ball sets and camera effects, and the ball is seen from head height. Each close-up is
cut the way the reader cuts a real detector box (a little off-centre, a little too
big or small), at the ball's real size in pixels (about 20-90 px wide, as in the
robot's video), then enlarged to the reader's size.

Balls that are hard to tell apart (the 8 and the black-capped stripes) get more
close-ups (--focus). A dark stripe with black caps seen from its cap, with almost none
of the stripe showing, looks just like the 8. It is labelled "unsure": a person can't
tell it either, and the robot then takes another look instead of guessing.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from . import table as T
from .augment import camera_effects, jpeg_quality
from .layouts import random_layout
from .pool_balls import appearance
from .reader import PAD, blank_surroundings, crop_ball
from .scene import CameraPose, Renderer, random_camera, random_look

UNSURE = "unsure"  # reader class: a stripe ball showing (almost) only its caps
STUDIO_PX = 192  # render size; the ball is about STUDIO_PX / FRAME pixels wide in it
FRAME = 2.2  # render width / ball width (room for the crop's shift and size jitter)
MIN_VISIBLE = 0.30  # like the full-table labels: at least 30% of the ball in view


def stripe_share(cam: np.ndarray, centre: np.ndarray, radius: float, pole: np.ndarray,
                 stripe_half: float, n: int = 4000, rng: np.random.Generator | None = None,
                 inner: float = 0.75) -> float:
    """Share of the middle of the ball's face (as the camera sees it) covered by the stripe.

    The stripe is the band round the ball's equator, `stripe_half` of pole-to-pole on
    each side (see pool_balls.pool_ball_image). Seen end-on, a stripe ball shows its
    cap and only a thin, squashed, dark ring of stripe right at its edge, so only the
    middle `inner` of the ball's width is counted.
    """
    rng = rng or np.random.default_rng(0)
    p = rng.normal(size=(n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)  # points all over the ball
    v = (cam - centre) / np.linalg.norm(cam - centre)
    facing = p @ v
    seen = facing > math.sqrt(1 - inner**2)  # inside the middle of the silhouette
    w = facing[seen]  # area each point covers in the picture shrinks as it turns away
    lat = np.arcsin(np.clip(p[seen] @ (pole / np.linalg.norm(pole)), -1, 1))
    in_stripe = np.abs(lat) < stripe_half * math.pi
    return float((w * in_stripe).sum() / w.sum())


def look_label(num: int, style, share: float | None, unsure_below: float) -> str:
    """Reader class for a ball: its look, or UNSURE for a stripe that can't be told from the 8.

    That is a stripe with black caps and a dark stripe colour (blue, purple, green,
    maroon, brown) seen from its cap: only a thin dark ring of colour shows at the edge.
    A bright ring (yellow, orange, red, pink) on a black cap, or any ring on a white
    cap, still shows which ball it is, so those keep their look.
    """
    if not 9 <= num <= 15 or share is None or share >= unsure_below or style.stripe_caps is None:
        return appearance(num, style.looks)
    black_caps = max(style.stripe_caps) < 0.15
    dark_stripe = max(style.colours[num - 8]) < 0.75
    return UNSURE if black_caps and dark_stripe else appearance(num, style.looks)


def _ball_number(name: str) -> int:
    return 0 if name == "cue" else int(name)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Render ball close-ups for the reader (fast).")
    ap.add_argument("--out", type=Path, default=Path("data/studio"))
    ap.add_argument("--layouts", type=int, default=400, help="tables (each with its own cloth, lamps and ball set)")
    ap.add_argument("--shots", type=int, default=2, help="close-ups of each ball per table")
    ap.add_argument("--focus", default="8,10,13,14", help="balls that get extra close-ups")
    ap.add_argument("--focus-times", type=int, default=3, help="how many times more close-ups they get")
    ap.add_argument("--size", type=int, default=64, help="close-up size the reader learns on (pixels)")
    ap.add_argument("--unsure-below", type=float, default=0.10,
                    help="a stripe ball is 'unsure' when less than this share of what you see is stripe")
    ap.add_argument("--blank", action="store_true", help="paint the cloth round the ball grey")
    ap.add_argument("--val", type=float, default=0.1, help="share of tables kept for checking")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    spec = T.SPECS["pool"]
    focus = {int(x) for x in a.focus.split(",") if x.strip()}
    rng = np.random.default_rng(a.seed)
    if a.out.exists():
        shutil.rmtree(a.out)  # always a fresh set
    a.out.mkdir(parents=True)
    (a.out / "crops.json").write_text(json.dumps({"blank_surroundings": a.blank, "size": a.size,
                                                  "studio": True, "unsure_below": a.unsure_below}))
    r = Renderer(STUDIO_PX, STUDIO_PX, seed=a.seed)
    counts: dict[str, int] = {}
    t0 = time.time()
    k = 0
    for li in range(a.layouts):
        split = "val" if rng.random() < a.val else "train"
        balls = random_layout(rng, spec)
        look = random_look(rng, spec)
        r.build(balls, look, spec)
        # Expose for the whole table first, like the robot's camera does, then keep it.
        r.set_camera(random_camera(rng, spec))
        r.render_frame(auto_exposure=True)
        r.freeze_shadows()  # nothing moves from here on (a ball turning on the spot casts the same shadow)
        R = spec.ball_r
        for i, b in enumerate(balls):
            num = _ball_number(b.name)
            shots = a.shots * (a.focus_times if num in focus else 1)
            node = r.ball_np[i]
            centre = np.array([b.x, b.y, R])
            for _ in range(shots):
                node.setHpr(*rng.uniform(0, 360, 3))  # a fresh resting angle for each shot
                pose = random_camera(rng, spec)
                cam = np.array(pose.pos)
                dist = float(np.linalg.norm(cam - centre))
                half = math.tan(math.asin(R / dist)) * FRAME  # tan of half the view, so the ball is 1/FRAME wide
                r.set_camera(CameraPose(pose.pos, tuple(centre), math.degrees(2 * math.atan(half)), pose.roll))
                rgb, ids = r.render_frame(auto_exposure=False, frames=1)
                outline = r.ball_outline(b)
                if outline is None:
                    continue
                whole = float(cv2.contourArea(outline.astype(np.float32)))
                if whole <= 0 or (ids == i).sum() / whole < MIN_VISIBLE:
                    continue  # hidden behind another ball from here
                pole = np.array(node.getQuat(r.render).getUp())
                share = stripe_share(cam, centre, R, pole, look.pool_balls.stripe_half, rng=rng) if num >= 9 else None
                name = look_label(num, look.pool_balls, share, a.unsure_below)
                # Shrink to the ball's real size in the robot's pictures, add camera effects there.
                x0, y0 = outline.min(0)
                x1, y1 = outline.max(0)
                ball_px = float(np.clip(rng.lognormal(math.log(44), 0.35), 18, 110))
                f = ball_px / max(x1 - x0, y1 - y0)
                small = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
                small = camera_effects(small, rng)
                ok, enc = cv2.imencode(".jpg", small[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality(rng)])
                bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                # Cut it like a detector box: a little off-centre, a little too big or small.
                crop = crop_ball(bgr, (x0 * f, y0 * f, x1 * f, y1 * f), pad=PAD, size=a.size,
                                 shift=tuple(rng.normal(0, 0.06, 2)), scale=float(rng.uniform(0.9, 1.15)))
                if a.blank:
                    crop = blank_surroundings(crop)
                folder = a.out / split / name
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folder / f"{a.seed:02d}{k:07d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                counts[f"{split}/{name}"] = counts.get(f"{split}/{name}", 0) + 1
                k += 1
        if (li + 1) % 10 == 0 or li + 1 == a.layouts:
            print(f"{li + 1}/{a.layouts} tables, {k} close-ups, {k / (time.time() - t0):.1f} per second", flush=True)
    train = sum(v for c, v in counts.items() if c.startswith("train/"))
    val = sum(v for c, v in counts.items() if c.startswith("val/"))
    print(f"{train} training close-ups, {val} check close-ups in {a.out}")
    print("per look: " + ", ".join(f"{n} {counts.get('train/' + n, 0)}"
                                   for n in sorted({c.split('/')[1] for c in counts})))


if __name__ == "__main__":
    main()
