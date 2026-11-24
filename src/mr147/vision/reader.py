"""The close-up ball reader: names each ball from a full-size crop of the picture.

The detector looks at the whole picture shrunk to about 1,000 pixels wide, which
is plenty to find a ball but often too few pixels to tell a 10 from an 8. So we
use two steps:

    1. the detector finds the balls (and pockets) in the shrunk picture
    2. for each ball, a small square around it is cut from the FULL-SIZE picture
       and the reader (a small image classifier) says which ball it is

Make the reader's training pictures from a generated set, then train it:

    uv run mr147-crops data/synth-pool2                  # -> data/synth-pool2-crops
    uv run mr147-train-reader --data data/synth-pool2-crops --game pool   # -> models/pool-reader.pt

and use it:  YoloDetector("models/pool.pt", reader="models/pool-reader.pt")
or:          mr147-check-video ... --model models/pool.pt --reader models/pool-reader.pt

Only the ball, not the cloth: everything outside the ball's circle is painted plain
grey before the reader sees it. A reader that also saw the cloth learnt that a blue
ring round a black ball means "the blue 10 seen from its black cap", so on blue
cloth the black 8 kept coming out as a 10. Crops are made with the grey surround
(`crops.json` in the crops folder says so) and the trained reader carries a
`pool-reader.json` beside it, so it gets the same kind of crop when it is used.
Readers trained before this have no such file and still get the cloth.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

CROP_SIZE = 128  # pixels; the reader looks at 128 x 128 crops
PAD = 0.3  # crop = the ball's box plus this share of its size on every side
SURROUND = (128, 128, 128)  # what replaces the cloth around the ball
EDGE = 1.04  # the kept circle is a touch wider than the box: a slightly small box keeps the ball whole, little cloth shows


def blank_surroundings(crop: np.ndarray, pad: float = PAD) -> np.ndarray:
    """Paint everything outside the ball's circle grey (the circle the box says, in the middle of the crop)."""
    size = crop.shape[0]
    r = size / (2 * (1 + 2 * pad)) * EDGE
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    out = crop.copy()
    out[(xx - c) ** 2 + (yy - c) ** 2 > r * r] = SURROUND
    return out


def _settings_file(weights: str | Path) -> Path:
    return Path(weights).with_suffix(".json")


def reader_blanks_surroundings(weights: str | Path) -> bool:
    """Was this reader trained on crops with a grey surround? (older readers: no settings file, no)"""
    f = _settings_file(weights)
    return f.exists() and bool(json.loads(f.read_text()).get("blank_surroundings"))


def crop_ball(image: np.ndarray, box, pad: float = PAD, size: int = CROP_SIZE,
              shift: tuple[float, float] = (0.0, 0.0), scale: float = 1.0) -> np.ndarray:
    """Square crop centred on the box (edges repeated where it leaves the picture), resized."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0, 2.0) * (1 + 2 * pad) * scale
    cx = (x0 + x1) / 2 + shift[0] * side
    cy = (y0 + y1) / 2 + shift[1] * side
    # Map the square onto `size` x `size` pixels in one step (no huge padded copies).
    s = size / side
    m = np.array([[s, 0, size / 2 - s * cx], [0, s, size / 2 - s * cy]], np.float32)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.warpAffine(image, m, (size, size), flags=interp, borderMode=cv2.BORDER_REPLICATE)


class BallReader:
    """A trained reader (an ultralytics image classifier). Needs the `ultralytics` package."""

    def __init__(self, weights: str | Path, device: str | None = None, blank: bool | None = None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.names = self.model.names
        self.device = device
        self.blank = reader_blanks_surroundings(weights) if blank is None else blank

    def read(self, image: np.ndarray, boxes: list) -> list[dict[str, float]]:
        """For each box in the RGB image: ball name -> probability."""
        if not boxes:
            return []
        bgr = np.ascontiguousarray(image[..., ::-1])
        crops = [crop_ball(bgr, b) for b in boxes]
        if self.blank:
            crops = [blank_surroundings(c) for c in crops]
        results = self.model.predict(crops, imgsz=CROP_SIZE, device=self.device, verbose=False)
        return [{self.names[i]: float(p) for i, p in enumerate(r.probs.data.tolist())} for r in results]


def reread(found: list, image: np.ndarray, reader: BallReader, keep: float = 0.01,
           ball_set: str = "traditional") -> list:
    """Replace each ball's name and alternatives with what the reader says (pockets untouched).

    The reader says how a ball looks ("purple", "pink-stripe"); `ball_set` turns that
    into a number (pool_balls.LOOK_TO_NUMBER). Looks that mean the same number add up.
    """
    from .detector import PixelDetection
    from .pool_balls import appearance_to_number

    balls = [d for d in found if d.name != "pocket"]
    out = [d for d in found if d.name == "pocket"]
    for d, probs in zip(balls, reader.read(image, [d.box for d in balls])):
        numbers: dict[str, float] = {}
        for look, p in probs.items():
            n = appearance_to_number(look, ball_set)
            numbers[n] = numbers.get(n, 0.0) + p
        ranked = sorted(((n, p) for n, p in numbers.items() if p >= keep), key=lambda kv: -kv[1])
        if not ranked:
            out.append(d)
            continue
        out.append(PixelDetection(ranked[0][0], d.box, ranked[0][1], tuple(ranked[:5])))
    return out


# --------------------------------------------------------------------------- training pictures


def make_crops(data: Path, out: Path, size: int = CROP_SIZE, per_ball: int = 1, seed: int = 0,
               blank: bool = True) -> dict:
    """Cut every labelled ball out of a generated set: out/{train,val}/<look>/*.jpg.

    The folder is how the ball looks in that picture's ball set ("purple", "pink-stripe",
    "black", "cue"; see pool_balls.appearance), read from the picture's meta file.
    Each crop is moved and resized a little at random, like the detector's boxes.
    With `blank`, everything outside the ball is grey (see blank_surroundings).
    """
    from .check_data import read_names
    from .pool_balls import appearance

    names = read_names(data)
    rng = np.random.default_rng(seed)
    counts: dict[str, int] = {}
    out.mkdir(parents=True, exist_ok=True)
    (out / "crops.json").write_text(json.dumps({"blank_surroundings": blank}))
    for split in ("train", "val"):
        for label in sorted((data / "labels" / split).glob("*.txt")):
            img_path = data / "images" / split / f"{label.stem}.jpg"
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            meta = data / "meta" / f"{label.stem}.json"
            looks = None
            if meta.exists():
                pb = json.loads(meta.read_text()).get("look", {}).get("pool_balls") or {}
                looks = {int(k): v for k, v in (pb.get("looks") or {}).items()} or None
            for i, line in enumerate(label.read_text().split("\n")):
                parts = line.split()
                if len(parts) != 5:
                    continue
                name = names[int(parts[0])]
                if name == "pocket":
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:])
                box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
                if looks is not None:  # older sets without looks keep the number as the class
                    name = appearance(0 if name == "cue" else int(name), looks)
                folder = out / split / name
                folder.mkdir(parents=True, exist_ok=True)
                for k in range(per_ball):
                    jitter = split == "train"
                    crop = crop_ball(bgr, box, size=size,
                                     shift=tuple(rng.normal(0, 0.06, 2)) if jitter else (0.0, 0.0),
                                     scale=float(rng.uniform(0.9, 1.15)) if jitter else 1.0)
                    if blank:  # the circle stays where the (jittered) box says, like a real detector box
                        crop = blank_surroundings(crop)
                    cv2.imwrite(str(folder / f"{label.stem}_{i}_{k}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    counts[f"{split}/{name}"] = counts.get(f"{split}/{name}", 0) + 1
    return counts


def crops_main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Cut ball crops out of a generated set, for the reader.")
    ap.add_argument("data", type=Path, help="a generated set (data/synth-pool2)")
    ap.add_argument("--out", type=Path, help="default: <data>-crops")
    ap.add_argument("--per-ball", type=int, default=1, help="crops per labelled ball (shifted a little)")
    ap.add_argument("--keep-cloth", action="store_true", help="keep the cloth round each ball (the old way)")
    a = ap.parse_args(argv)
    out = a.out or a.data.with_name(a.data.name + "-crops")
    if out.exists():
        shutil.rmtree(out)  # always rebuilt from the set
    counts = make_crops(a.data, out, per_ball=a.per_ball, blank=not a.keep_cloth)
    train = sum(v for k, v in counts.items() if k.startswith("train/"))
    val = sum(v for k, v in counts.items() if k.startswith("val/"))
    print(f"{train} training crops, {val} check crops in {out}")
    missing = sorted({k.split("/")[1] for k in counts} - {k.split("/")[1] for k in counts if k.startswith("val/")})
    if not val:
        raise SystemExit("no check crops at all (make a bigger set)")
    if missing:
        print(f"note: no check crops for {', '.join(missing)} (fine for a small test set)")


def reader_settings() -> dict:
    """Training settings that never change a ball's identity (see train.augment_settings)."""
    return dict(fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.2, hsv_v=0.4, erasing=0.1)


def train_main(argv: list[str] | None = None) -> None:
    from ultralytics import YOLO

    from .train import pick_device

    ap = argparse.ArgumentParser(description="Train the close-up ball reader.")
    ap.add_argument("--data", type=Path, required=True, help="crops folder from mr147-crops")
    ap.add_argument("--game", default="pool")
    ap.add_argument("--start-from", default="yolo11n-cls.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    device = pick_device() if a.device == "auto" else a.device
    model = YOLO(a.start_from)
    model.train(data=str(a.data.resolve()), epochs=a.epochs, imgsz=CROP_SIZE, batch=a.batch, device=device,
                project=str(Path("runs").resolve()), name=f"{a.game}-reader", exist_ok=True,
                deterministic=False, **reader_settings())
    best = Path(model.trainer.best)
    out = Path("models") / f"{a.game}-reader.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copy(best, out)
    # The reader must be used on the same kind of crop it learnt from.
    crops = a.data / "crops.json"
    blank = crops.exists() and bool(json.loads(crops.read_text()).get("blank_surroundings"))
    _settings_file(out).write_text(json.dumps({"blank_surroundings": blank, "crops": str(a.data)}))
    print(f"\nbest reader: {out}   (scores in {best.parent.parent})")
