"""Check a training set before spending hours training on it.

    uv run mr147-check-data data/synth-pool

Looks for the mistakes that don't crash anything but quietly teach the model the
wrong thing:

  * every picture has a label file and every label file has a picture
  * the class names in data.yaml are exactly the game's classes, in order
  * every label line is "class x y w h" with a known class and a box inside the picture
  * the cloth isn't burnt out (flat cyan / lime) in many pictures
  * both train and val exist, and no layout is in both

mr147-train runs these checks first and stops if any fail.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .table import SPECS

IMAGE_TYPES = {".jpg", ".jpeg", ".png"}


def read_names(folder: Path) -> list[str]:
    names = {}
    lines = (folder / "data.yaml").read_text().splitlines()
    in_names = False
    for line in lines:
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names and line.startswith("  ") and ":" in line:
            k, v = line.strip().split(":", 1)
            names[int(k)] = v.strip()
        elif in_names:
            in_names = False
    return [names[i] for i in sorted(names)]


def blown_fraction(img_rgb: np.ndarray) -> float:
    """Share of the picture that is strongly coloured and burnt out (a cloth of any colour)."""
    top = img_rgb.max(axis=2).astype(int)
    low = img_rgb.min(axis=2).astype(int)
    return float(((top >= 250) & (low < 0.75 * top)).mean())


def check_dataset(folder: Path, game: str, look_at_pictures: int = 60,
                  require_all_classes: bool = True) -> tuple[list[str], dict]:
    """Returns (problems, stats). No problems = safe to train."""
    folder = Path(folder)
    spec = SPECS[game]
    problems: list[str] = []
    stats: dict = {}
    if not (folder / "data.yaml").exists():
        return [f"{folder}/data.yaml is missing"], stats

    names = read_names(folder)
    if names != list(spec.classes):
        problems.append(f"data.yaml classes {names} are not the {game} classes {list(spec.classes)}")

    class_counts: Counter = Counter()
    layouts: dict[str, set] = {}
    for split in ("train", "val"):
        img_dir, lab_dir = folder / "images" / split, folder / "labels" / split
        images = {p.stem: p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_TYPES} if img_dir.exists() else {}
        labels = {p.stem: p for p in lab_dir.glob("*.txt")} if lab_dir.exists() else {}
        stats[f"{split}_images"] = len(images)
        if not images:
            problems.append(f"no {split} pictures")
            continue
        if missing := sorted(set(images) - set(labels)):
            problems.append(f"{len(missing)} {split} pictures have no label file (e.g. {missing[0]})")
        if orphans := sorted(set(labels) - set(images)):
            problems.append(f"{len(orphans)} {split} label files have no picture (e.g. {orphans[0]})")
        bad_lines = 0
        example = ""
        for stem, lab in labels.items():
            for line in lab.read_text().splitlines():
                parts = line.split()
                ok = len(parts) == 5
                if ok:
                    try:
                        c, x, y, w, h = int(parts[0]), *map(float, parts[1:])
                    except ValueError:
                        ok = False
                if ok:
                    ok = (0 <= c < len(spec.classes) and 0 < w <= 1 and 0 < h <= 1
                          and 0 <= x - w / 2 + 1e-6 and x + w / 2 - 1e-6 <= 1
                          and 0 <= y - h / 2 + 1e-6 and y + h / 2 - 1e-6 <= 1)
                if ok:
                    class_counts[spec.classes[c]] += 1
                else:
                    bad_lines += 1
                    example = example or f"{lab.name}: '{line}'"
        if bad_lines:
            problems.append(f"{bad_lines} bad label lines in {split} (e.g. {example})")
        meta_dir = folder / "meta"
        layouts[split] = set()
        for stem in images:
            m = meta_dir / f"{stem}.json"
            if m.exists():
                meta = json.loads(m.read_text())
                layouts[split].add((stem[:2], meta.get("layout")))
    if layouts.get("train") and layouts.get("val") and (both := layouts["train"] & layouts["val"]):
        problems.append(f"{len(both)} layouts are in both train and val (the score would be too kind)")

    missing_classes = [c for c in spec.classes if class_counts[c] == 0]
    if missing_classes and require_all_classes:
        problems.append(f"no examples at all of: {', '.join(missing_classes)}")
    stats["boxes_per_class"] = dict(class_counts)

    # Look at some pictures: burnt-out cloth teaches the wrong colours.
    pics = sorted((folder / "images" / "train").glob("*.jpg"))[:look_at_pictures]
    blown = [p.name for p in pics if blown_fraction(cv2.imread(str(p))[..., ::-1]) > 0.10]
    stats["blown_out_pictures"] = f"{len(blown)} of {len(pics)}"
    if pics and len(blown) > 0.1 * len(pics):
        problems.append(f"{len(blown)} of {len(pics)} pictures have burnt-out cloth (e.g. {blown[0]})")
    return problems, stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--game", choices=sorted(SPECS), help="default: read from the folder name")
    a = ap.parse_args(argv)
    game = a.game or ("pool" if "pool" in a.folder.name else "snooker")
    problems, stats = check_dataset(a.folder, game)
    for k, v in stats.items():
        print(f"{k}: {v}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nlooks fine")


if __name__ == "__main__":
    main()
