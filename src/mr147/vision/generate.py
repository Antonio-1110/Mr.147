"""Make a labelled training set of robot's-eye snooker or pool images.

    uv run mr147-gen --game snooker --layouts 500 --views 4
    uv run mr147-gen --game pool --layouts 500 --views 4

Writes a YOLO-style dataset to data/synth-<game>/ (or --out):
    data/synth-<game>/images/{train,val}/000123.jpg
    data/synth-<game>/labels/{train,val}/000123.txt   class cx cy w h (0..1)
    data/synth-<game>/meta/000123.json                ball 3D positions, camera, visibility
    data/synth-<game>/data.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import table as T
from .augment import camera_effects, jpeg_quality
from .layouts import random_layout
from .scene import Renderer, random_camera, random_look

MIN_VISIBLE = 0.30  # keep a label if at least 30% of the ball can be seen
MIN_PIXELS = 12  # ...and at least this many pixels of it


def _polygon_area_in_image(poly: np.ndarray, w: int, h: int) -> float:
    """Area of the outline after clipping to the image (via a small mask)."""
    x0, y0 = np.floor(poly.min(0)).astype(int)
    x1, y1 = np.ceil(poly.max(0)).astype(int)
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1c <= x0c or y1c <= y0c:
        return 0.0
    mask = np.zeros((y1c - y0c, x1c - x0c), np.uint8)
    cv2.fillPoly(mask, [np.round(poly - [x0c, y0c]).astype(np.int32)], 1)
    return float(mask.sum())


def _label(outline, count: int, cls: int, w: int, h: int, entry: dict, rows: list) -> None:
    """Fill in visibility and box for one object, and add a YOLO row if it's visible enough."""
    entry.update(visible_px=int(count), visible_frac=0.0, box=None)
    if outline is None:
        return
    whole = float(cv2.contourArea(outline.astype(np.float32)))
    if whole <= 0:
        return
    entry["visible_frac"] = round(min(count / whole, 1.0), 3)
    entry["in_frame_frac"] = round(min(_polygon_area_in_image(outline, w, h) / whole, 1.0), 3)
    x0, y0 = outline.min(0)
    x1, y1 = outline.max(0)
    entry["centre_px"] = [round(float((x0 + x1) / 2), 2), round(float((y0 + y1) / 2), 2)]
    bx0, by0, bx1, by1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)  # whole object, clipped
    if count >= MIN_PIXELS and entry["visible_frac"] >= MIN_VISIBLE and bx1 > bx0 and by1 > by0:
        entry["box"] = [round(float(v), 2) for v in (bx0, by0, bx1, by1)]
        rows.append(f"{cls} {(bx0 + bx1) / 2 / w:.6f} {(by0 + by1) / 2 / h:.6f} "
                    f"{(bx1 - bx0) / w:.6f} {(by1 - by0) / h:.6f}")


def label_view(r: Renderer, balls, ids: np.ndarray, spec: T.TableSpec = T.SNOOKER):
    """Boxes for every ball and pocket that is visible enough.

    Returns (yolo rows, per-ball info, per-pocket info).
    """
    w, h = r.w, r.h
    pockets = spec.pockets
    counts = np.bincount(ids[ids >= 0].ravel(), minlength=len(balls) + len(pockets))
    rows, info, pocket_info = [], [], []
    for i, b in enumerate(balls):
        entry = {"label": b.label, "class": b.name, "xyz": [b.x, b.y, spec.ball_r]}
        _label(r.ball_outline(b), counts[i], spec.class_id[b.name], w, h, entry, rows)
        info.append(entry)
    for k, (px, py, pr) in enumerate(pockets):
        entry = {"label": f"pocket{k + 1}", "class": "pocket", "xyz": [px, py, 0.0], "radius": pr}
        _label(r.pocket_outline(k), counts[len(balls) + k], spec.class_id["pocket"], w, h, entry, rows)
        pocket_info.append(entry)
    return rows, info, pocket_info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", choices=sorted(T.SPECS), default="snooker")
    ap.add_argument("--out", type=Path, help="default: data/synth-<game>")
    ap.add_argument("--layouts", type=int, default=100, help="number of ball layouts")
    ap.add_argument("--views", type=int, default=4, help="camera views per layout")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--val", type=float, default=0.1, help="share of layouts kept for validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true", help="save without blur/noise effects too")
    a = ap.parse_args(argv)

    spec = T.SPECS[a.game]
    a.out = a.out or Path(f"data/synth-{a.game}")
    rng = np.random.default_rng(a.seed)
    r = Renderer(a.width, a.height, seed=a.seed)
    for split in ("train", "val"):
        (a.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (a.out / "meta").mkdir(parents=True, exist_ok=True)
    (a.out / "data.yaml").write_text(
        f"path: {a.out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(spec.classes))
    )

    t0 = time.time()
    k = 0
    for li in range(a.layouts):
        split = "val" if rng.random() < a.val else "train"  # split by layout, not by image
        balls = random_layout(rng, spec)
        look = random_look(rng, spec)
        r.build(balls, look, spec)
        for _ in range(a.views):
            pose = random_camera(rng, spec)
            r.set_camera(pose)
            rgb, ids = r.render_frame()
            rows, info, pocket_info = label_view(r, balls, ids, spec)
            img = rgb if a.clean else camera_effects(rgb, rng)
            name = f"{a.seed:02d}{k:06d}"
            cv2.imwrite(str(a.out / "images" / split / f"{name}.jpg"), img[..., ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality(rng)])
            (a.out / "labels" / split / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            meta = {"game": a.game, "layout": li, "split": split, "camera": r.camera_info(),
                    "camera_pose": pose.__dict__, "look": {**look.__dict__, "pool_balls": look.pool_balls.__dict__}, "balls": info,
                    "pockets": pocket_info}
            (a.out / "meta" / f"{name}.json").write_text(json.dumps(meta, default=float, indent=1))
            k += 1
        if (li + 1) % 10 == 0 or li + 1 == a.layouts:
            print(f"{li + 1}/{a.layouts} layouts, {k} images, {k / (time.time() - t0):.1f} img/s", flush=True)


if __name__ == "__main__":
    main()
