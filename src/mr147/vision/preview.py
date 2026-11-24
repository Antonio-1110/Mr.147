"""Draw the automatic labels onto images so a person can check them.

    uv run mr147-preview data/synth-pool --n 12
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


DRAW = {"pocket": (0, 140, 255), "red": (255, 255, 0), "yellow": (0, 230, 255), "green": (60, 255, 60), "brown": (40, 90, 160),
        "blue": (255, 120, 40), "pink": (200, 150, 255), "black": (255, 255, 255), "white": (255, 0, 255)}


def class_names(data: Path) -> list[str]:
    """Read the label names from data.yaml (lines like '  3: brown')."""
    names = []
    for line in (data / "data.yaml").read_text().splitlines():
        if line.startswith("  ") and ":" in line:
            names.append(line.split(":", 1)[1].strip())
    return names


def draw(img_path: Path, label_path: Path, names: list[str]) -> np.ndarray:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    for line in label_path.read_text().split("\n"):
        if not line.strip():
            continue
        c, cx, cy, bw, bh = line.split()
        name = names[int(c)]
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        p0, p1 = (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2))
        cv2.rectangle(img, p0, p1, DRAW.get(name, (255, 255, 0)), 1)
        if bw > 12:
            cv2.putText(img, name, (p0[0], p0[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, DRAW.get(name, (255, 255, 0)), 1, cv2.LINE_AA)
    return img


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--split", default="train")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    imgs = sorted((a.data / "images" / a.split).glob("*.jpg"))[: a.n]
    names = class_names(a.data)
    tiles = [draw(p, a.data / "labels" / a.split / (p.stem + ".txt"), names) for p in imgs]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % a.cols)
    grid = np.vstack([np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)])
    out = a.out or a.data / "preview.jpg"
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(out)


if __name__ == "__main__":
    main()
