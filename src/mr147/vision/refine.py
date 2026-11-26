"""Tighten a ball's box to its real outline, to a fraction of a pixel.

The ball finder's boxes are good to a pixel or two, which at a few metres is
several millimetres on the table. The ball's edge in the picture is much sharper
than that, so we find it directly:

  1. From the box centre, look outwards along 72 spokes for the place where the
     colour changes most (the ball's edge), to a tenth of a pixel.
  2. Fit an ellipse through those edge points (a ball seen off-centre in a wide
     lens looks slightly oval). Spokes that hit something else, such as a ball in
     front, its shadow, or a shine on the cloth, don't fit and are dropped.
  3. If too few spokes agree, or the answer is far from the finder's box, keep the
     finder's box: refining must never make things worse.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

SPOKES = 72
STEP = 0.2  # pixels between samples along a spoke


def _sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear colour samples (... x 3) at float pixel positions."""
    return cv2.remap(img, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def edge_points(img: np.ndarray, box, inner: float = 0.55, outer: float = 1.35) -> tuple[np.ndarray, np.ndarray]:
    """Where each spoke leaves the ball: (points N x 2, edge strength N).

    The edge is the outermost strong colour change along the spoke (at least half
    as strong as the strongest), so a shading line inside the ball isn't taken.
    """
    x0, y0, x1, y1 = box
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    th = np.linspace(0, 2 * math.pi, SPOKES, endpoint=False)
    r_px = max(rx, ry)
    s = np.arange(inner * r_px, outer * r_px, STEP)  # distance from the centre, in pixels
    ex, ey = np.cos(th) * (rx / r_px), np.sin(th) * (ry / r_px)
    # Boxes count pixel i as running from i to i + 1 (the renderer's and the finder's
    # convention); sampling counts pixel i's centre as i. Hence the - 0.5.
    col = _sample(img, cx - 0.5 + ex[:, None] * s[None], cy - 0.5 + ey[:, None] * s[None]).astype(np.float32)
    k = max(1, int(round(0.6 / STEP)))  # compare colours ~1.2 px apart
    g = np.zeros(col.shape[:2], np.float32)
    g[:, k:-k] = np.linalg.norm(col[:, 2 * k:] - col[:, :-2 * k], axis=2)
    pts, strength = np.zeros((SPOKES, 2)), np.zeros(SPOKES)
    for j in range(SPOKES):
        gj = g[j]
        peaks = np.nonzero((gj[1:-1] >= gj[:-2]) & (gj[1:-1] > gj[2:]) & (gj[1:-1] >= 0.5 * gj.max()))[0] + 1
        if len(peaks) == 0:
            continue
        i = peaks[-1]
        a, b, c = gj[i - 1], gj[i], gj[i + 1]
        den = a - 2 * b + c
        f = float(np.clip(0.5 * (a - c) / den, -1, 1)) if abs(den) > 1e-6 else 0.0
        at = s[0] + (i + f) * STEP
        pts[j] = cx + ex[j] * at, cy + ey[j] * at
        strength[j] = b
    # The bottom of the ball sits in its own shadow on the cloth, so that part of
    # the outline is unreliable: leave it out (the rest pins the ellipse down).
    strength[np.sin(th) > BOTTOM_SKIP] = 0.0
    return pts, strength


BOTTOM_SKIP = 0.7  # spokes pointing further down than ~45 degrees are ignored


def _ellipse_distance(pts: np.ndarray, ell) -> np.ndarray:
    """Roughly how far each point is from the ellipse outline, in pixels."""
    (cx, cy), (w, h), ang = ell
    a = math.radians(ang)
    d = pts - [cx, cy]
    u = d[:, 0] * math.cos(a) + d[:, 1] * math.sin(a)
    v = -d[:, 0] * math.sin(a) + d[:, 1] * math.cos(a)
    rad = np.hypot(u, v)
    t = np.arctan2(v * w, u * h)  # direction on the ellipse
    on = np.hypot(w / 2 * np.cos(t), h / 2 * np.sin(t))
    return np.abs(rad - on)


def _ellipse_box(ell) -> np.ndarray:
    (cx, cy), (w, h), ang = ell
    a = math.radians(ang)
    hx = math.hypot(w / 2 * math.cos(a), h / 2 * math.sin(a))
    hy = math.hypot(w / 2 * math.sin(a), h / 2 * math.cos(a))
    return np.array([cx - hx, cy - hy, cx + hx, cy + hy])


def refine_ball_box(image: np.ndarray, box, min_agree: float = 0.45) -> tuple[np.ndarray, bool]:
    """(tighter box, True) or (the finder's box, False) when the edge can't be trusted."""
    box = np.asarray(box, float)
    h, w = image.shape[:2]
    size = min(box[2] - box[0], box[3] - box[1])
    if size < 8:  # too small to see an edge reliably
        return box, False
    if box[0] <= 1 or box[1] <= 1 or box[2] >= w - 1 or box[3] >= h - 1:
        return box, False  # cut off by the picture's edge: the box centre isn't the ball's
    pts, strength = edge_points(image, box)
    inside = (pts[:, 0] > 1) & (pts[:, 0] < w - 2) & (pts[:, 1] > 1) & (pts[:, 1] < h - 2)
    ok = inside & (strength > 0.25 * np.median(strength[inside])) if inside.any() else inside
    tol = max(0.6, 0.025 * size)
    ell = None
    for _ in range(4):  # fit, drop spokes that don't agree, fit again
        if ok.sum() < max(12, min_agree * SPOKES):
            return box, False
        ell = cv2.fitEllipse(pts[ok].astype(np.float32))
        dist = _ellipse_distance(pts, ell)
        new_ok = ok & (dist < max(tol, 2.5 * np.median(dist[ok])))
        if new_ok.sum() == ok.sum():
            break
        ok = new_ok
    if ell is None or ok.sum() < max(12, min_agree * SPOKES):
        return box, False
    (_, _), (ew, eh), _ = ell
    if max(ew, eh) > 1.3 * min(ew, eh):  # a ball is never that oval: something else was fitted
        return box, False
    new = _ellipse_box(ell)
    centre_shift = np.hypot(*(new[:2] + new[2:] - box[:2] - box[2:]) / 2)
    ratio = (new[2] - new[0]) / max(box[2] - box[0], 1e-6)
    if centre_shift > 0.25 * size or not 0.8 < ratio < 1.25:
        return box, False
    # The finder's box is clipped at the picture's edge; keep that clipping.
    return np.array([max(new[0], 0), max(new[1], 0), min(new[2], w), min(new[3], h)]), True
