"""Score a detector (and reader) against an answer key of real pictures.

    uv run --extra train python -m mr147.vision.score_key --model models/pool.pt \\
        --reader models/pool-reader.pt --ball-set modern

The key (data/real-pool/answer-key.json) lists, for each picture, every ball's box
and its true name. Names come from the one-of-each step, as the robot would use them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def score(detector, key: dict, folder: Path, game: str = "pool") -> dict:
    from .detector import iou
    from .identify import name_once_in_picture

    right = total = missed = extra = 0
    wrong: list[str] = []
    for picture, balls in key.items():
        bgr = cv2.imread(str(folder / picture))
        if bgr is None:
            raise SystemExit(f"missing picture: {folder / picture}")
        found = [d for d in name_once_in_picture(detector.detect(bgr[..., ::-1].copy()), game) if d.name != "pocket"]
        used = set()
        for ball in balls:
            total += 1
            best = max(range(len(found)), key=lambda i: iou(found[i].box, ball["box"]), default=None)
            if best is None or iou(found[best].box, ball["box"]) < 0.3:
                missed += 1
                continue
            used.add(best)
            if found[best].name == ball["name"]:
                right += 1
            else:
                wrong.append(f"{ball['name']}->{found[best].name}")
        extra += len(found) - len(used)
    return dict(right=right, total=total, missed=missed, extra=extra, wrong=wrong)


def main(argv: list[str] | None = None) -> None:
    from .detector import YoloDetector

    ap = argparse.ArgumentParser(description="Score models against the real-picture answer key.")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--reader", type=Path)
    ap.add_argument("--ball-set", choices=["traditional", "modern"], default="modern")
    ap.add_argument("--key", type=Path, default=Path("data/real-pool/answer-key.json"))
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    key = json.loads(a.key.read_text())
    detector = YoloDetector(a.model, reader=a.reader, ball_set=a.ball_set)
    # Photos and video frames (named <video>-f<frame>.jpg) are scored separately:
    # video frames are smaller and blurrier, and are what the robot will see.
    groups = {"photos": {k: v for k, v in key.items() if "-f" not in k},
              "video frames": {k: v for k, v in key.items() if "-f" in k}}
    for group, part in groups.items():
        if not part:
            continue
        s = score(detector, part, a.key.parent)
        wrong = " ".join(s["wrong"][:20]) + (" ..." if len(s["wrong"]) > 20 else "")
        print(f"{a.label or a.model.name} [{group}]: right {s['right']} of {s['total']} balls, missed {s['missed']}, "
              f"non-balls called balls {s['extra']}" + (f"   wrong: {wrong}" if wrong else ""))


if __name__ == "__main__":
    main()
