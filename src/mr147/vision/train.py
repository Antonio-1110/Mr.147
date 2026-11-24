"""Train the ball detector on the simulator's pictures.

    uv sync --extra train                      # once: installs ultralytics + PyTorch
    uv run mr147-gen --game snooker --layouts 1500 --views 4
    uv run mr147-train --game snooker

The finished model is copied to models/<game>.pt. Training starts from a small
ready-made YOLO model and teaches it our labels only (balls and pockets).

To fine-tune on real photos later, point --data at a folder that has both real
and simulated pictures, and start from the model you already have:

    uv run mr147-train --game snooker --data data/real-snooker --start-from models/snooker.pt --epochs 20
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon graphics chip
    return "cpu"


def augment_settings(game: str) -> dict:
    """How the trainer may alter pictures on the fly (checked by tests/test_data_pipeline.py)."""
    return dict(
        # The simulator already varies colour and lighting. Turn off the trainer's
        # colour shifts: they would turn a pink ball red or a yellow ball brown.
        hsv_h=0.0,
        hsv_s=0.2,
        fliplr=0.5 if game == "snooker" else 0.0,  # mirrored numbers would confuse pool
        flipud=0.0,  # upside-down numbers too
        mosaic=1.0,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", choices=["snooker", "pool"], default="snooker")
    ap.add_argument("--data", type=Path, help="dataset folder (default data/synth-<game>)")
    ap.add_argument("--start-from", default="yolo11n.pt",
                    help="ready-made model to start from, or your own .pt to keep training it")
    ap.add_argument("--epochs", type=int, default=60, help="passes over all pictures")
    ap.add_argument("--image-size", type=int, default=1280, help="far balls are tiny, so keep this big")
    ap.add_argument("--batch", type=int, default=8, help="lower it if you run out of memory")
    ap.add_argument("--device", default="auto", help="auto, mps, cpu, or a GPU number")
    ap.add_argument("--name", help="run name under runs/ (default: the game)")
    ap.add_argument("--skip-check", action="store_true", help="don't check the pictures and labels first")
    a = ap.parse_args(argv)

    data = a.data or Path(f"data/synth-{a.game}")
    if not a.skip_check:
        from .check_data import check_dataset

        problems, _ = check_dataset(data, a.game)
        if problems:
            raise SystemExit("the training set has problems, not training:\n  - " + "\n  - ".join(problems)
                             + f"\n(see: uv run mr147-check-data {data})")

    from ultralytics import YOLO

    device = pick_device() if a.device == "auto" else a.device
    print(f"training on {data} with {device}")
    model = YOLO(a.start_from)
    model.train(
        data=str(data / "data.yaml"),
        epochs=a.epochs,
        imgsz=a.image_size,
        batch=a.batch,
        device=device,
        project=str(Path("runs").resolve()),  # a relative path would end up under runs/detect/
        name=a.name or a.game,
        exist_ok=True,
        deterministic=False,  # Apple's GPU has no exact-repeat versions of some steps; it only warned
        **augment_settings(a.game),
    )
    best = Path(model.trainer.best)
    Path("models").mkdir(exist_ok=True)
    out = Path("models") / f"{a.game}.pt"
    shutil.copy(best, out)
    print(f"\nbest model: {out}   (scores and charts in {best.parent.parent})")


if __name__ == "__main__":
    main()
