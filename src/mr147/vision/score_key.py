"""Score a detector (and reader) against an answer key of real pictures.

    uv run --extra train python -m mr147.vision.score_key --model models/pool.pt \\
        --reader models/pool-reader.pt --ball-set modern

The key (data/real-pool/answer-key.json) lists, for each picture, every ball's box
and its true name. Names come from the one-of-each step, as the robot would use them.

Each ball counts as right, wrong, or left unnamed. Unnamed is not a mistake: the
robot looks again. Wrong is, so the aim is wrong = 0 with as few unnamed as possible.
  --bars 0.9,0.95   leave a ball unnamed when its name scored below the bar (vote.py)
  --vote 30         for video frames, also follow each ball through the frames up to
                    30 frames (half a second at 60 fps) either side and average the
                    reader's answers (needs the videos next to the key: <clip>.MOV)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2

FRAME = re.compile(r"frames/(?P<clip>.+)-f(?P<n>\d+)\.jpg$")


def video_frames(video: Path, centre: int, reach: int, step: int) -> tuple[list, int]:
    """RGB frames centre-reach .. centre+reach (every `step`), and the centre's index in that list."""
    wanted = set(range(max(centre - reach, 0), centre + reach + 1, step)) | {centre}
    cap = cv2.VideoCapture(str(video))
    out, k, mid = [], 0, 0
    while k <= max(wanted):
        ok, bgr = cap.read()  # read in order: seeking in phone videos isn't frame-exact
        if not ok:
            break
        if k in wanted:
            if k == centre:
                mid = len(out)
            out.append(bgr[..., ::-1].copy())
        k += 1
    cap.release()
    return out, mid


def score(detector, key: dict, folder: Path, game: str = "pool", bars=(0.0,), vote: int = 0,
          step: int = 6) -> dict:
    """Per bar: right / wrong / unnamed / missed counts, and the wrong ones ("8->10", "8->?")."""
    from .detector import iou
    from .vote import UNNAMED, name_balls

    results = {bar: dict(right=0, wrong=0, unnamed=0, total=0, missed=0, extra=0, mistakes=[]) for bar in bars}
    for picture, balls in key.items():
        m = FRAME.match(picture)
        video = folder / f"{m['clip']}.MOV" if m else None
        if vote and video is not None and video.exists():
            frames, mid = video_frames(video, int(m["n"]), vote, step)
        else:
            bgr = cv2.imread(str(folder / picture))
            if bgr is None:
                raise SystemExit(f"missing picture: {folder / picture}")
            frames, mid = [bgr[..., ::-1].copy()], 0
        found = [detector.detect(f) for f in frames]
        for bar in bars:
            r = results[bar]
            named = [d for d in name_balls(found, mid, game, bar) if d.name != "pocket"]
            used = set()
            for ball in balls:
                r["total"] += 1
                best = max(range(len(named)), key=lambda i: iou(named[i].box, ball["box"]), default=None)
                if best is None or iou(named[best].box, ball["box"]) < 0.3:
                    r["missed"] += 1
                    continue
                used.add(best)
                if named[best].name == UNNAMED:
                    r["unnamed"] += 1
                elif named[best].name == ball["name"]:
                    r["right"] += 1
                else:
                    r["wrong"] += 1
                    r["mistakes"].append(f"{ball['name']}->{named[best].name}")
            r["extra"] += len(named) - len(used)
    return results


def main(argv: list[str] | None = None) -> None:
    from .detector import YoloDetector

    ap = argparse.ArgumentParser(description="Score models against the real-picture answer key.")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--reader", type=Path)
    ap.add_argument("--ball-set", choices=["traditional", "modern"], default="modern")
    ap.add_argument("--key", type=Path, default=Path("data/real-pool/answer-key.json"))
    ap.add_argument("--label", default="")
    ap.add_argument("--bars", default="0", help="comma-separated: leave balls unnamed below this score")
    ap.add_argument("--vote", type=int, default=0, help="video frames either side to follow each ball through")
    ap.add_argument("--step", type=int, default=6, help="use every n-th frame when voting")
    a = ap.parse_args(argv)
    key = json.loads(a.key.read_text())
    bars = tuple(float(b) for b in a.bars.split(","))
    detector = YoloDetector(a.model, reader=a.reader, ball_set=a.ball_set)
    # Photos and video frames (named <video>-f<frame>.jpg) are scored separately:
    # video frames are smaller and blurrier, and are what the robot will see.
    groups = {"photos": {k: v for k, v in key.items() if "-f" not in k},
              "video frames": {k: v for k, v in key.items() if "-f" in k}}
    for group, part in groups.items():
        if not part:
            continue
        for bar, s in score(detector, part, a.key.parent, bars=bars, vote=a.vote if group == "video frames" else 0,
                            step=a.step).items():
            how = (f"bar {bar:.2f}" if bar else "no bar") + (f", vote ±{a.vote}" if a.vote and group != "photos" else "")
            wrong = " ".join(s["mistakes"][:20]) + (" ..." if len(s["mistakes"]) > 20 else "")
            print(f"{a.label or a.model.name} [{group}, {how}]: right {s['right']}, wrong {s['wrong']}, "
                  f"unnamed {s['unnamed']} of {s['total']} balls, missed {s['missed']}, "
                  f"non-balls called balls {s['extra']}" + (f"   wrong: {wrong}" if wrong else ""), flush=True)


if __name__ == "__main__":
    main()
