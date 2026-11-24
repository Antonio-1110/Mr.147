"""Check the trained detector on real videos and photos, by eye.

    uv run mr147-check-video match.mp4 --model models/snooker.pt
    uv run mr147-check-video clips/*.MOV photos/*.JPG --model models/pool.pt --game pool --frames 4
    uv run mr147-check-video match.mp4 --model models/snooker.pt --frames 30 --save-for-fixing data/real-snooker

Picks random frames from each video (--frames per video; photos count once), draws the model's boxes and writes one contact sheet
(video-check.jpg). Each frame also gets a count of what was found, and frames
where the count is impossible (16 reds, two blacks...) are marked in red, so
the bad ones stand out.

Ball names are chosen for all balls in the picture together, so each number is
used only once (identify.py): the surest "8" keeps the name and the others get
their next-best one. Renamed balls show the old name, e.g. "4 (was 8)"; a ball
left with a name the model barely considered is drawn with a "?". Use --raw to see
the model's own picks instead.

--save-for-fixing also writes each frame with the model's boxes as label files.
Open that folder in a box-editing app (X-AnyLabeling or Label Studio), fix only
the wrong boxes, and the result is real training data and a real score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .detector import YoloDetector
from .identify import name_once_in_picture
from .preview import DRAW

MAX_COUNT = {"snooker": {"red": 15, "pocket": 6}, "pool": {"pocket": 6}}


def impossible(counts: Counter, game: str) -> list[str]:
    limits = MAX_COUNT[game]
    problems = []
    for name, n in counts.items():
        limit = limits.get(name, 1)
        if n > limit:
            problems.append(f"{n} {name}")
    return problems


def pick_frames(n_total: int, n: int, rng: np.random.Generator) -> list[int]:
    """Random frames, spread out: one from each of n equal slices of the video."""
    edges = np.linspace(0, n_total, n + 1).astype(int)
    return [int(rng.integers(a, max(a + 1, b))) for a, b in zip(edges, edges[1:])]


PHOTO_TYPES = {".jpg", ".jpeg", ".png", ".heic", ".bmp"}


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Shrink to fit a w x h tile (photos and videos have different shapes)."""
    s = min(w / img.shape[1], h / img.shape[0])
    small = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    tile = np.zeros((h, w, 3), np.uint8)
    tile[:small.shape[0], :small.shape[1]] = small
    return tile


def read_frames(sources: list[Path], per_video: int, rng: np.random.Generator):
    """Yield (source, frame number, label, BGR image) for every photo and random video frames."""
    for source in sources:
        if source.suffix.lower() in PHOTO_TYPES:
            bgr = cv2.imread(str(source))  # also turns the photo upright
            if bgr is not None:
                yield source, 0, source.name, bgr
            continue
        cap = cv2.VideoCapture(str(source))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        if total <= 0:
            print(f"can't read {source}")
            continue
        for idx in pick_frames(total, per_video, rng):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if ok:
                t = idx / fps
                yield source, idx, f"{source.name} {int(t // 60)}:{t % 60:04.1f}", bgr


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", type=Path, nargs="+", help="videos and/or photos")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--reader", type=Path, help="close-up ball reader (models/pool-reader.pt): names balls from full-size crops")
    ap.add_argument("--game", choices=["snooker", "pool"], default="snooker")
    ap.add_argument("--ball-set", choices=["traditional", "modern"], default="traditional",
                    help="pool ball colours: traditional (4 purple, 5 orange) or modern (4 pink, 5 purple)")
    ap.add_argument("--frames", type=int, default=12, help="random frames per video")
    ap.add_argument("--min-confidence", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("video-check.jpg"))
    ap.add_argument("--raw", action="store_true", help="show the model's own names (no one-of-each rule)")
    ap.add_argument("--save-for-fixing", type=Path, help="also save frames + boxes here, YOLO format")
    a = ap.parse_args(argv)

    detector = YoloDetector(a.model, min_confidence=a.min_confidence, reader=a.reader, ball_set=a.ball_set)
    names = [detector.names[i] for i in sorted(detector.names)]
    rng = np.random.default_rng(a.seed)

    if a.save_for_fixing:
        for sub in ("images", "labels"):
            (a.save_for_fixing / sub).mkdir(parents=True, exist_ok=True)
        (a.save_for_fixing / "classes.txt").write_text("\n".join(names) + "\n")

    tiles = []
    for source, idx, label, bgr in read_frames(a.sources, a.frames, rng):
        raw = detector.detect(bgr[..., ::-1])
        found = raw if a.raw else name_once_in_picture(raw, a.game)
        h, w = bgr.shape[:2]
        tile = bgr.copy()
        for d, r in zip(found, raw if a.raw else _same_order(found, raw)):
            x0, y0, x1, y1 = (int(round(v)) for v in d.box)
            colour = DRAW.get(d.name, (255, 255, 0))
            text = f"{d.name} {d.confidence:.2f}"
            if r is not None and r.name != d.name:
                text = f"{d.name} (was {r.name})"
            if d.name != "pocket" and d.confidence < 0.1:
                text += " ?"
                colour = (0, 0, 255)
            k = max(1.0, w / 1280)  # big photos: thicker lines so they survive shrinking
            cv2.rectangle(tile, (x0, y0), (x1, y1), colour, int(2 * k))
            cv2.putText(tile, text, (x0, max(int(12 * k), y0 - int(4 * k))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k, colour, max(1, int(k)), cv2.LINE_AA)
        counts = Counter(d.name for d in found)
        bad = impossible(counts, a.game)
        renamed = sum(r is not None and r.name != d.name for d, r in zip(found, _same_order(found, raw)))
        unsure = sum(d.name != "pocket" and d.confidence < 0.1 for d in found)
        summary = f"{label}  " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
        if not a.raw:
            summary += f"  [renamed {renamed}, unsure {unsure}]"
        colour = (0, 0, 255) if bad else (255, 255, 255)
        k = max(1.0, w / 1280)
        cv2.rectangle(tile, (0, 0), (w, int(34 * k)), (0, 0, 0), -1)
        cv2.putText(tile, summary + ("   IMPOSSIBLE: " + ", ".join(bad) if bad else ""), (8, int(24 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, colour, int(2 * k), cv2.LINE_AA)
        tiles.append(_fit(tile, 960, 720))
        print(summary + (f"   <- impossible: {', '.join(bad)}" if bad else ""))

        if a.save_for_fixing:
            stem = f"{source.stem}_{idx:06d}"
            cv2.imwrite(str(a.save_for_fixing / "images" / f"{stem}.jpg"), bgr)
            rows = [f"{names.index(d.name)} {(d.box[0] + d.box[2]) / 2 / w:.6f} {(d.box[1] + d.box[3]) / 2 / h:.6f} "
                    f"{(d.box[2] - d.box[0]) / w:.6f} {(d.box[3] - d.box[1]) / h:.6f}" for d in found]
            (a.save_for_fixing / "labels" / f"{stem}.txt").write_text("\n".join(rows) + "\n")

    if not tiles:
        raise SystemExit("no frames could be read")
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % a.cols)
    sheet = np.vstack([np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)])
    cv2.imwrite(str(a.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\ncontact sheet: {a.out}")


def _same_order(found: list, raw: list) -> list:
    """For each box in `found`, the model's original box (matched by position), or None."""
    by_box = {tuple(r.box): r for r in raw}
    return [by_box.get(tuple(d.box)) for d in found]


if __name__ == "__main__":
    main()
