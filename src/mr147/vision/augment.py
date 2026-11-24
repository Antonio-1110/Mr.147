"""Camera-style image effects: exposure, colour cast, blur, noise, JPEG artefacts.

None of these move a ball, so the labels stay valid.
"""

import cv2
import numpy as np


def camera_effects(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    # Exposure. A real camera brightens a dark scene but doesn't burn out the cloth,
    # so the brightening is capped where the cloth's brightest parts would clip.
    top = np.percentile(x.max(axis=2), 97)  # any cloth colour
    x = x * min(rng.uniform(0.7, 1.35), max(1.0, 0.97 / max(top, 1e-3)))
    x = x * rng.uniform(0.92, 1.08, 3)[None, None, :]  # white balance
    x = np.clip(x, 0, 1) ** rng.uniform(0.8, 1.25)  # contrast / gamma
    if rng.random() < 0.15:  # vignette
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
        x *= (1 - rng.uniform(0.1, 0.35) * np.clip(r - 0.4, 0, None))[..., None]
    x = (np.clip(x, 0, 1) * 255).astype(np.uint8)

    r = rng.random()
    if r < 0.35:  # out of focus
        s = rng.uniform(0.4, 1.6)
        x = cv2.GaussianBlur(x, (0, 0), s)
    elif r < 0.5:  # motion blur from the robot's head moving
        k = int(rng.integers(3, 10))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
        kernel = cv2.warpAffine(kernel, rot, (k, k))
        x = cv2.filter2D(x, -1, kernel / max(kernel.sum(), 1e-6))

    sigma = rng.uniform(0, 7)  # sensor noise
    if sigma > 0.5:
        x = np.clip(x.astype(np.float32) + rng.normal(0, sigma, x.shape), 0, 255).astype(np.uint8)
    return x


def jpeg_quality(rng: np.random.Generator) -> int:
    return int(rng.integers(60, 96))
