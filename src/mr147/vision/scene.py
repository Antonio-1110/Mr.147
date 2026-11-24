"""A 3D snooker table rendered with Panda3D, plus automatic ball labels.

Each call to `Renderer.render()` produces:
  * the photo-like image the robot's camera would see, and
  * an "ID image" where every ball is painted a unique flat colour and the rest
    is black. Counting those pixels tells us how much of each ball is visible,
    so balls hidden behind other balls get dropped from the labels.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
from panda3d.core import (
    AmbientLight,
    ColorAttrib,
    FrameBufferProperties,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    GraphicsOutput,
    GraphicsPipe,
    LineSegs,
    Material,
    NodePath,
    PerspectiveLens,
    RenderState,
    Shader,
    ShaderAttrib,
    LightAttrib,
    TextureAttrib,
    Spotlight,
    Texture,
    WindowProperties,
    loadPrcFileData,
)

from . import table as T
from .shaders import FRAG, MAX_LIGHTS, VERT
from .pool_balls import CLASSIC, PoolBallStyle, pool_ball_image
from .table import POOL_HEAD_X, POOL_FOOT_X, POOL_RGB, SNOOKER, Ball, TableSpec

W_DEFAULT, H_DEFAULT = 1280, 720
CLOTH_ID = 150  # ID-image value for the cloth and cushions (balls and pockets use 0, 1, 2, ...)


# --------------------------------------------------------------------------- geometry


def _sphere(radius: float, rings: int = 24, segments: int = 36) -> GeomNode:
    vdata = GeomVertexData("sphere", GeomVertexFormat.getV3n3t2(), Geom.UHStatic)
    vw, nw, tw = (GeomVertexWriter(vdata, n) for n in ("vertex", "normal", "texcoord"))
    for i in range(rings + 1):
        phi = math.pi * i / rings
        for j in range(segments + 1):
            th = 2 * math.pi * j / segments
            n = (math.sin(phi) * math.cos(th), math.sin(phi) * math.sin(th), math.cos(phi))
            vw.addData3(*(radius * c for c in n))
            nw.addData3(*n)
            tw.addData2(j / segments, i / rings)
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(rings):
        for j in range(segments):
            a = i * (segments + 1) + j
            b = a + segments + 1
            tris.addVertices(a, b, a + 1)
            tris.addVertices(a + 1, b, b + 1)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("sphere")
    node.addGeom(geom)
    return node


def _box(sx: float, sy: float, sz: float, uv_scale: float = 1.0) -> GeomNode:
    """Axis-aligned box centred on the origin in x/y, from z=0 to z=sz."""
    vdata = GeomVertexData("box", GeomVertexFormat.getV3n3t2(), Geom.UHStatic)
    vw, nw, tw = (GeomVertexWriter(vdata, n) for n in ("vertex", "normal", "texcoord"))
    hx, hy = sx / 2, sy / 2
    faces = [  # normal, 4 corners
        ((0, 0, 1), [(-hx, -hy, sz), (hx, -hy, sz), (hx, hy, sz), (-hx, hy, sz)]),
        ((0, 0, -1), [(-hx, hy, 0), (hx, hy, 0), (hx, -hy, 0), (-hx, -hy, 0)]),
        ((1, 0, 0), [(hx, -hy, 0), (hx, hy, 0), (hx, hy, sz), (hx, -hy, sz)]),
        ((-1, 0, 0), [(-hx, hy, 0), (-hx, -hy, 0), (-hx, -hy, sz), (-hx, hy, sz)]),
        ((0, 1, 0), [(hx, hy, 0), (-hx, hy, 0), (-hx, hy, sz), (hx, hy, sz)]),
        ((0, -1, 0), [(-hx, -hy, 0), (hx, -hy, 0), (hx, -hy, sz), (-hx, -hy, sz)]),
    ]
    tris = GeomTriangles(Geom.UHStatic)
    for k, (n, corners) in enumerate(faces):
        for c in corners:
            vw.addData3(*c)
            nw.addData3(*n)
            tw.addData2(c[0] * uv_scale, c[1] * uv_scale)
        tris.addVertices(4 * k, 4 * k + 1, 4 * k + 2)
        tris.addVertices(4 * k, 4 * k + 2, 4 * k + 3)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("box")
    node.addGeom(geom)
    return node


def _off_cloth(centre, half_l: float, half_w: float):
    """Moves points (relative to `centre`) that lie over the playing area onto its edge.

    The pocket's rail-level parts (the cut-out, liner and cap) must stop at the cushion
    line; drawn as full circles they would hover over the cloth like a black lid.
    Corner pockets sit on the playing area's corner, so their points go to the corner.
    """
    px, py = centre
    corner = abs(abs(px) - half_l) < 1e-6 and abs(abs(py) - half_w) < 1e-6

    def snap(x: float, y: float) -> tuple[float, float]:
        wx, wy = px + x, py + y
        if abs(wx) >= half_l or abs(wy) >= half_w:
            return x, y
        if corner:
            return 0.0, 0.0
        if half_l - abs(wx) < half_w - abs(wy):
            return math.copysign(half_l, wx) - px, y
        return x, math.copysign(half_w, wy) - py

    return snap


def _disc(radius: float, segments: int = 40, snap=None) -> GeomNode:
    vdata = GeomVertexData("disc", GeomVertexFormat.getV3n3(), Geom.UHStatic)
    vw, nw = GeomVertexWriter(vdata, "vertex"), GeomVertexWriter(vdata, "normal")
    snap = snap or (lambda x, y: (x, y))
    vw.addData3(*snap(0.0, 0.0), 0)
    nw.addData3(0, 0, 1)
    for j in range(segments + 1):
        th = 2 * math.pi * j / segments
        vw.addData3(*snap(radius * math.cos(th), radius * math.sin(th)), 0)
        nw.addData3(0, 0, 1)
    tris = GeomTriangles(Geom.UHStatic)
    for j in range(segments):
        tris.addVertices(0, j + 1, j + 2)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("disc")
    node.addGeom(geom)
    return node


def _ring(r_in: float, r_out: float, a0: float = 0.0, a1: float = 2 * math.pi,
          segments: int = 40, snap=None) -> GeomNode:
    """Flat ring (or part of one, from angle a0 to a1) facing up: pocket caps and liners."""
    snap = snap or (lambda x, y: (x, y))
    vdata = GeomVertexData("ring", GeomVertexFormat.getV3n3t2(), Geom.UHStatic)
    vw, nw, tw = (GeomVertexWriter(vdata, n) for n in ("vertex", "normal", "texcoord"))
    for j in range(segments + 1):
        th = a0 + (a1 - a0) * j / segments
        for r in (r_in, r_out):
            vw.addData3(*snap(r * math.cos(th), r * math.sin(th)), 0)
            nw.addData3(0, 0, 1)
            tw.addData2(j / segments, r / r_out)
    tris = GeomTriangles(Geom.UHStatic)
    for j in range(segments):
        a = 2 * j
        tris.addVertices(a, a + 1, a + 3)
        tris.addVertices(a, a + 3, a + 2)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("ring")
    node.addGeom(geom)
    return node


def _noise_texture(rng: np.random.Generator, size: int = 256, strength: float = 18.0) -> Texture:
    """Soft grey noise used to give the cloth and wood a little texture (strength: grey levels)."""
    n = rng.normal(0, 1, (size // 8, size // 8)).astype(np.float32)
    n = cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC)  # soft blotches
    n = n + 0.35 * rng.normal(0, 1, (size, size)).astype(np.float32)  # fine grain
    n = 205 + strength * (n - n.mean()) / (n.std() + 1e-6)
    img = np.clip(n, 0, 255).astype(np.uint8)
    tex = Texture("noise")
    tex.setup2dTexture(size, size, Texture.TUnsignedByte, Texture.FLuminance)
    tex.setRamImage(img.tobytes())
    tex.setWrapU(Texture.WMRepeat)
    tex.setWrapV(Texture.WMRepeat)
    tex.setMinfilter(Texture.FTLinearMipmapLinear)
    tex.setMagfilter(Texture.FTLinear)
    return tex


def _material(rgb, shininess: float, spec: float) -> Material:
    m = Material()
    m.setDiffuse((*rgb, 1))
    m.setAmbient((*rgb, 1))
    m.setSpecular((spec, spec, spec, 1))
    m.setShininess(shininess)
    return m


def pool_ball_texture(num: int, style: PoolBallStyle = CLASSIC, w: int = 512, h: int = 256) -> Texture:
    """Wrap-around picture of a pool ball in a given set's style (see pool_balls.py)."""
    img = pool_ball_image(num, style, w, h)
    # Panda reads textures bottom row first, and our sphere's u runs the other way round.
    img = img[::-1, ::-1]
    tex = Texture(f"pool{num}-{style.name}")
    tex.setup2dTexture(w, h, Texture.TUnsignedByte, Texture.FRgb)
    tex.setRamImageAs(np.ascontiguousarray(img).tobytes(), "RGB")
    tex.setMinfilter(Texture.FTLinearMipmapLinear)
    tex.setAnisotropicDegree(4)
    return tex


# --------------------------------------------------------------------------- camera pose


@dataclass
class CameraPose:
    pos: tuple[float, float, float]  # metres, table frame (cloth at z=0)
    look_at: tuple[float, float, float]
    fov_h: float  # horizontal field of view, degrees
    roll: float = 0.0  # degrees


def random_camera(rng: np.random.Generator, spec: TableSpec = SNOOKER) -> CameraPose:
    """Robot standing somewhere around the table, eyes at head height, looking at the cloth."""
    half_l = spec.length / 2 + spec.cushion_w + spec.rail_w
    half_w = spec.width / 2 + spec.cushion_w + spec.rail_w
    gap = rng.uniform(0.15, 1.2)  # how far the robot stands from the table edge
    side = rng.integers(4)
    if side < 2:  # long sides
        x = rng.uniform(-half_l - 0.3, half_l + 0.3)
        y = (half_w + gap) * (1 if side == 0 else -1)
    else:  # short ends
        x = (half_l + gap) * (1 if side == 2 else -1)
        y = rng.uniform(-half_w - 0.3, half_w + 0.3)
    # Eye height above the floor: mostly standing (1.4-1.85 m), sometimes bent over.
    eye = rng.uniform(1.4, 1.85) if rng.random() < 0.75 else rng.uniform(1.0, 1.4)
    z = eye - spec.cloth_height
    target = (
        rng.uniform(-spec.length / 2 * 0.9, spec.length / 2 * 0.9),
        rng.uniform(-spec.width / 2 * 0.9, spec.width / 2 * 0.9),
        0.0,
    )
    return CameraPose((float(x), float(y), float(z)), target, float(rng.uniform(55, 80)),
                      float(rng.normal(0, 2.0)))


# --------------------------------------------------------------------------- renderer


@dataclass
class Look:
    """Random appearance settings for one layout."""

    cloth_rgb: tuple
    wood_rgb: tuple
    floor_rgb: tuple
    sky_rgb: tuple
    lights: list = field(default_factory=list)  # (x, y, z, intensity, rgb)
    ambient: float = 0.25
    ball_jitter: float = 0.04
    pool_balls: PoolBallStyle = CLASSIC  # which kind of ball set (pool only)
    cloth_name: str = "green"
    pockets: dict = field(default_factory=dict)  # how the pockets look (see random_pockets)


# Pool cloth colours seen in halls, with rough shares. Snooker is always green.
POOL_CLOTHS = {  # name: (share, rgb low, rgb high)
    "electric blue": (0.32, (0.10, 0.35, 0.70), (0.25, 0.55, 0.90)),
    "navy": (0.08, (0.02, 0.10, 0.30), (0.06, 0.35, 0.70)),
    "green": (0.22, (0.02, 0.28, 0.06), (0.12, 0.55, 0.20)),
    "teal": (0.06, (0.00, 0.35, 0.35), (0.10, 0.55, 0.55)),
    "grey": (0.10, (0.30, 0.30, 0.31), (0.60, 0.60, 0.62)),
    "burgundy": (0.07, (0.40, 0.03, 0.06), (0.65, 0.10, 0.12)),
    "tan": (0.05, (0.55, 0.42, 0.25), (0.75, 0.60, 0.40)),
    "purple": (0.04, (0.25, 0.08, 0.40), (0.45, 0.20, 0.60)),
    "black": (0.03, (0.04, 0.04, 0.05), (0.10, 0.10, 0.11)),
    "red": (0.03, (0.65, 0.05, 0.05), (0.85, 0.15, 0.12)),
}
CAPS = {  # pocket corner caps on the rail top: name -> (rgb, shininess, specular)
    "none": None,
    "chrome": ((0.55, 0.55, 0.57), 90.0, 0.7),
    "black plastic": ((0.04, 0.04, 0.04), 40.0, 0.35),
    "brass": ((0.50, 0.37, 0.14), 60.0, 0.45),
    "dark grey": ((0.22, 0.22, 0.23), 30.0, 0.25),
}


def random_pockets(rng: np.random.Generator, game: str) -> dict:
    """How the pockets look: hole darkness, liner (leather / rubber) round the opening, caps."""
    liner = (tuple(np.full(3, rng.uniform(0.02, 0.08))) if rng.random() < 0.6  # black rubber / leather
             else tuple(np.array([0.30, 0.17, 0.08]) * rng.uniform(0.4, 1.2)))  # brown leather
    caps = ["none", "chrome", "black plastic", "brass", "dark grey"]
    p = [0.35, 0.25, 0.2, 0.08, 0.12] if game == "pool" else [0.55, 0.1, 0.1, 0.2, 0.05]
    return {
        "hole": float(rng.uniform(0.0, 0.05)),  # how dark the inside looks
        "liner_rgb": liner,
        "liner_width": float(rng.uniform(0.08, 0.3)),  # share of the opening's radius
        "cap": str(rng.choice(caps, p=p)),
        "cap_width": float(rng.uniform(0.015, 0.045)),  # metres
        "cap_angle": float(rng.uniform(1.6, 2.6)),  # half the arc the cap covers (radians)
    }


def random_look(rng: np.random.Generator, spec: TableSpec = SNOOKER) -> Look:
    if spec.game == "snooker":  # snooker is always green cloth; only the shade varies
        h = rng.uniform(0.0, 1.0)
        cloth = (0.03 + 0.06 * h, 0.28 + 0.25 * rng.random(), 0.06 + 0.08 * rng.random())
        cloth_name = "green"
    else:  # pool: any hall colour, so one detector works on every table
        names = list(POOL_CLOTHS)
        shares = np.array([POOL_CLOTHS[n][0] for n in names])
        cloth_name = str(rng.choice(names, p=shares / shares.sum()))
        _, lo, hi = POOL_CLOTHS[cloth_name]
        # One shade between the low and high colour, plus a little per-channel wobble.
        # (Drawing each channel on its own turned some "grey" cloths beige or lilac.)
        lo, hi = np.array(lo), np.array(hi)
        t = rng.uniform(0, 1) + rng.uniform(-0.06, 0.06, 3)
        cloth = tuple(float(v) for v in np.clip(lo + np.clip(t, 0, 1) * (hi - lo), 0, 1))
    wood = tuple(np.array([0.30, 0.14, 0.06]) * rng.uniform(0.4, 1.4))
    if spec.game == "pool" and rng.random() < 0.7:  # black or dark grey rails (most pool halls)
        wood = tuple(np.full(3, rng.uniform(0.02, 0.10)))
    floor = tuple(rng.uniform(0.05, 0.35, 3))
    sky = tuple(rng.uniform(0.0, 0.12) * rng.uniform(0.6, 1.0, 3))
    if spec.game == "pool" and rng.random() < 0.5:  # a bright, lit room behind the table
        sky = tuple(rng.uniform(0.35, 0.85) * rng.uniform(0.85, 1.0, 3))
    # Snooker canopy: a row of lamps about 1 m above the table, plus random extra lamps.
    lights = []
    temp = rng.uniform(-0.08, 0.08)  # warm / cool tint
    tint = (1 + temp, 1.0, 1 - temp)
    n = int(rng.choice([2, 3, 4]))
    h = rng.uniform(0.8, 1.4)
    base = rng.uniform(0.8, 1.5)
    for i in range(n):
        x = (i - (n - 1) / 2) * spec.length / n + rng.normal(0, 0.1)
        lights.append((x, rng.normal(0, 0.1), h, base * rng.uniform(0.85, 1.15), tint))
    if rng.random() < 0.35:  # an extra room light off to one side
        lights.append((rng.uniform(-3, 3), rng.choice([-1, 1]) * rng.uniform(1.5, 3),
                       rng.uniform(1.2, 2.2), rng.uniform(0.2, 0.6), tint))
    look = Look(cloth, wood, floor, sky, lights, float(rng.uniform(0.12, 0.4)),
                float(rng.uniform(0.0, 0.08)))
    look.cloth_name = cloth_name
    look.pockets = random_pockets(rng, spec.game)
    if spec.game == "pool":  # a different maker's ball set on each table
        look.pool_balls = PoolBallStyle.random(rng)
    return look


def _setup_panda(width: int, height: int) -> None:
    display = "pandaegl" if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") else "pandagl"
    loadPrcFileData(
        "",
        f"""
        window-type offscreen
        win-size {width} {height}
        load-display {display}
        audio-library-name null
        framebuffer-multisample 1
        multisamples 4
        sync-video 0
        show-frame-rate-meter 0
        notify-level warning
        default-directnotify-level warning
        """,
    )


def _cloth_level(rgb: np.ndarray, cloth: np.ndarray | None = None) -> float | None:
    """Median brightness (strongest channel) of the cloth.

    cloth: which pixels are cloth (the renderer knows). Without it, guess: clearly
    blue or green pixels (only right for blue and green cloth).
    """
    top = rgb.max(axis=2).astype(int)
    if cloth is None:
        r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
        cloth = (np.maximum(g, b) > 25) & (r < 0.6 * np.maximum(g, b))
    if cloth.mean() < 0.05:
        return None
    return float(np.median(top[cloth]))


def _blown_cloth(rgb: np.ndarray, cloth: np.ndarray | None = None) -> float:
    """Fraction of the picture that is cloth and burnt out (a channel at 250+)."""
    if cloth is None:
        r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
        return float(((np.maximum(g, b) >= 250) & (r < 0.75 * np.maximum(g, b))).mean())
    return float((cloth & (rgb.max(axis=2) >= 250)).mean())


class Renderer:
    def __init__(self, width: int = W_DEFAULT, height: int = H_DEFAULT, seed: int = 0):
        _setup_panda(width, height)
        from direct.showbase.ShowBase import ShowBase  # import after config is set

        self.w, self.h = width, height
        self.base = ShowBase()
        self.base.disableMouse()
        self.base.setFrameRateMeter(False)  # other Panda3D users (pooltool) may switch it on
        self.rng = np.random.default_rng(seed)
        self.render = self.base.render

        # Main colour buffer: copy pixels into RAM after each frame.
        self.color_tex = Texture()
        self.base.win.addRenderTexture(self.color_tex, GraphicsOutput.RTMCopyRam)
        self.lens = PerspectiveLens()
        self.lens.setNearFar(0.05, 30)
        self.base.cam.node().setLens(self.lens)

        # ID buffer: same camera, no lighting, no antialiasing, flat colours.
        fbp = FrameBufferProperties()
        fbp.setRgbColor(True)
        fbp.setRgbaBits(8, 8, 8, 0)
        fbp.setDepthBits(24)
        self.id_buf = self.base.graphicsEngine.makeOutput(
            self.base.pipe, "id", -2, fbp, WindowProperties.size(width, height),
            GraphicsPipe.BFRefuseWindow, self.base.win.getGsg(), self.base.win,
        )
        self.id_tex = Texture()
        self.id_buf.addRenderTexture(self.id_tex, GraphicsOutput.RTMCopyRam)
        self.id_buf.setClearColor((0, 0, 0, 1))
        id_cam = self.base.makeCamera(self.id_buf, lens=self.lens)
        id_cam.reparentTo(self.base.camera)
        cn = id_cam.node()
        cn.setTagStateKey("vid")
        black = RenderState.make(ColorAttrib.makeFlat((0, 0, 0, 1)), LightAttrib.makeAllOff(),
                                 TextureAttrib.makeAllOff(), ShaderAttrib.makeOff(), 1000)
        cn.setInitialState(black)
        self._id_cam = cn

        self.scene = self.render.attachNewNode("scene")
        self.ball_np: list[NodePath] = []
        self._spheres: dict[float, GeomNode] = {}
        self._pool_tex: dict[int, Texture] = {}
        self._pool_style: PoolBallStyle | None = None
        self.spec = SNOOKER
        self._shader = Shader.make(Shader.SL_GLSL, VERT, FRAG)

    # ----------------------------------------------------------------- build

    def build(self, balls: list[Ball], look: Look, spec: TableSpec = SNOOKER) -> None:
        rng = self.rng
        self.spec = spec
        # Panda3D keeps a lamp's shadow buffer alive after the lamp is removed, so
        # each rebuild would leave buffers that still render every frame (big slowdown).
        for lamp in getattr(self, "_lamps", []):
            lamp.setShadowCaster(False)
        self._lamps = []
        self.scene.removeNode()
        self.render.clearLight()
        self.scene = self.render.attachNewNode("scene")
        self.base.win.setClearColor((*look.sky_rgb, 1))
        noise = _noise_texture(rng)
        # Real cloth is nearly even; strong blotches made it look like water.
        cloth_noise = _noise_texture(rng, strength=float(rng.uniform(3.0, 9.0)))

        def add(node, pos, rgb, shin=10.0, spec=0.05, tex=True):
            np_ = self.scene.attachNewNode(node)
            np_.setPos(*pos)
            np_.setMaterial(_material(rgb, shin, spec), 1)
            np_.setColor((*rgb, 1))
            if tex:
                np_.setTexture(tex if isinstance(tex, Texture) else noise)
            return np_

        L, Wd = spec.length, spec.width
        cw, rw, rh, ch = spec.cushion_w, spec.rail_w, spec.rail_h, spec.cloth_height
        # Cloth bed (extends under the cushions) and the table body.
        self._id_cam.setTagState(str(CLOTH_ID), RenderState.make(
            ColorAttrib.makeFlat(self._id_colour(CLOTH_ID)), LightAttrib.makeAllOff(),
            TextureAttrib.makeAllOff(), ShaderAttrib.makeOff(), 1001))
        add(_box(L + 2 * cw, Wd + 2 * cw, 0.02, uv_scale=3), (0, 0, -0.02), look.cloth_rgb,
            tex=cloth_noise).setTag("vid", str(CLOTH_ID))
        add(_box(L + 2 * (cw + rw) - 0.04, Wd + 2 * (cw + rw) - 0.04, ch - 0.1),
            (0, 0, -ch), look.wood_rgb, shin=30, spec=0.2)
        # Floor.
        add(_box(30, 30, 0.01, uv_scale=0.5), (0, 0, -ch - 0.01), look.floor_rgb)

        # Cushions: six segments with gaps at the pockets.
        cp, mp = spec.corner_pocket_r * 1.15, spec.middle_pocket_r * 1.25
        seg_len = L / 2 - cp - mp
        cushion_rgb = tuple(c * 0.9 for c in look.cloth_rgb)
        for sy in (-1, 1):
            for sx in (-1, 1):
                cx = sx * (mp + seg_len / 2)
                add(_box(seg_len, cw, spec.cushion_h, 3), (cx, sy * (Wd / 2 + cw / 2), 0),
                    cushion_rgb, tex=cloth_noise).setTag("vid", str(CLOTH_ID))
        for sx in (-1, 1):
            add(_box(cw, Wd - 2 * cp, spec.cushion_h, 3), (sx * (L / 2 + cw / 2), 0, 0),
                cushion_rgb, tex=cloth_noise).setTag("vid", str(CLOTH_ID))
        # Wooden rails all round.
        add(_box(L + 2 * (cw + rw), rw, rh, 2), (0, Wd / 2 + cw + rw / 2, 0), look.wood_rgb, 40, 0.3)
        add(_box(L + 2 * (cw + rw), rw, rh, 2), (0, -Wd / 2 - cw - rw / 2, 0), look.wood_rgb, 40, 0.3)
        add(_box(rw, Wd + 2 * cw, rh, 2), (L / 2 + cw + rw / 2, 0, 0), look.wood_rgb, 40, 0.3)
        add(_box(rw, Wd + 2 * cw, rh, 2), (-L / 2 - cw - rw / 2, 0, 0), look.wood_rgb, 40, 0.3)
        # Pockets: dark holes on the cloth and cut-outs on the rail.
        # Both discs of a pocket share one ID so the ID image shows the whole dark hole.
        # Round the opening: a leather or rubber liner, and on many tables a metal or
        # plastic cap over the rail (these are not part of the pocket's label).
        self.n_balls = len(balls)
        pk = look.pockets or random_pockets(rng, spec.game)
        hole = (pk["hole"],) * 3
        for k, (px, py, pr) in enumerate(spec.pockets):
            vid = str(len(balls) + k)
            self._id_cam.setTagState(vid, RenderState.make(
                ColorAttrib.makeFlat(self._id_colour(len(balls) + k)), LightAttrib.makeAllOff(),
                TextureAttrib.makeAllOff(), ShaderAttrib.makeOff(), 1001))
            snap = _off_cloth((px, py), L / 2, Wd / 2)  # rail-level parts stop at the cushion line
            for z in (0.0006, rh + 0.0006):
                shade = hole if z > 0.01 else tuple(c * 0.6 for c in hole)  # deeper = darker
                d = add(_disc(pr * (1.0 if z < 0.01 else 1.1), snap=None if z < 0.01 else snap),
                        (px, py, z), shade, tex=False)
                d.setShaderOff(1)
                d.setLightOff(1)
                d.setColor((*shade, 1), 1)
                d.setTag("vid", vid)
            ro = pr * 1.1
            liner = add(_ring(ro * (1 - pk["liner_width"]), ro, snap=snap), (px, py, rh + 0.0009), pk["liner_rgb"],
                        shin=20, spec=0.15)
            liner.setTag("vid", vid)  # the liner is inside the opening: part of the pocket
            cap = CAPS[pk["cap"]]
            if cap is not None:
                rgb, shin, sp = cap
                out = math.atan2(py, px)  # the cap sits on the outer side of the opening
                add(_ring(ro, ro + pk["cap_width"], out - pk["cap_angle"], out + pk["cap_angle"], snap=snap),
                    (px, py, rh + 0.0012), rgb, shin=shin, spec=sp, tex=False)
        self._markings(spec, add)

        # Balls.
        self.ball_np = []
        sphere = self._spheres.setdefault(spec.ball_r, _sphere(spec.ball_r))
        for i, b in enumerate(balls):
            n = self.scene.attachNewNode(sphere.makeCopy())
            n.setPos(b.x, b.y, spec.ball_r)
            if spec.game == "pool":  # painted texture; the colour lives in the texture
                num = 0 if b.name == "cue" else int(b.name)
                if look.pool_balls is not self._pool_style:  # new ball set: repaint
                    self._pool_tex, self._pool_style = {}, look.pool_balls
                if num not in self._pool_tex:
                    self._pool_tex[num] = pool_ball_texture(num, look.pool_balls)
                n.setTexture(self._pool_tex[num], 1)
                rgb = np.clip(np.ones(3) * (1 + rng.normal(0, look.ball_jitter, 3)), 0, 1)
                n.setHpr(*rng.uniform(0, 360, 3))  # balls come to rest at any angle
            else:
                rgb = np.clip(np.array(T.BALL_RGB[b.name]) * (1 + rng.normal(0, look.ball_jitter, 3)), 0, 1)
            gloss = look.pool_balls.shine if spec.game == "pool" else 1.0  # worn sets are duller
            n.setMaterial(_material(tuple(rgb), rng.uniform(60, 120), min(1.0, rng.uniform(0.5, 0.9) * gloss)), 1)
            n.setColor((*rgb, 1))
            n.setTag("vid", str(i))
            self._id_cam.setTagState(str(i), RenderState.make(
                ColorAttrib.makeFlat(self._id_colour(i)), LightAttrib.makeAllOff(),
                TextureAttrib.makeAllOff(), ShaderAttrib.makeOff(), 1001))
            self.ball_np.append(n)

        # Lights.
        amb = AmbientLight("amb")
        amb.setColor((look.ambient, look.ambient, look.ambient, 1))
        self._amb = amb
        self._light_colours = [(amb, (look.ambient, look.ambient, look.ambient, 1))]
        self.exposure = 1.0
        # How bright this camera likes the cloth; a dark (navy, black) cloth stays darker.
        self.cloth_target = float(rng.uniform(175, 235)) * float(np.clip(max(look.cloth_rgb) / 0.75, 0.45, 1.0))
        self.render.setLight(self.scene.attachNewNode(amb))
        for k, (x, y, z, inten, tint) in enumerate(look.lights[:MAX_LIGHTS]):
            sl = Spotlight(f"lamp{k}")
            sl.setColor((inten * tint[0], inten * tint[1], inten * tint[2], 1))
            self._light_colours.append((sl, (inten * tint[0], inten * tint[1], inten * tint[2], 1)))
            lens = PerspectiveLens()
            lens.setFov(120)
            lens.setNearFar(0.1, 6)
            sl.setLens(lens)
            sl.setExponent(0.6)
            sl.setAttenuation((1, 0, 0.15))
            sl.setShadowCaster(True, 2048, 2048)
            self._lamps.append(sl)
            ln = self.scene.attachNewNode(sl)
            ln.setPos(x, y, z)
            ln.lookAt(x, y, 0)
            self.render.setLight(ln)
        self.scene.setShader(self._shader)

    def _markings(self, spec: TableSpec, add) -> None:
        L, Wd = spec.length, spec.width

        def unlit(np_):
            np_.setShaderOff(1)
            np_.setLightOff(1)
            return np_

        if spec.game == "snooker":  # baulk line, the D, and the spots
            ls = LineSegs()
            ls.setThickness(1.5)
            ls.setColor(0.85, 0.85, 0.85, 1)
            ls.moveTo(T.BAULK_X, -Wd / 2, 0.0005)
            ls.drawTo(T.BAULK_X, Wd / 2, 0.0005)
            for k in range(41):
                a = math.pi / 2 + math.pi * k / 40
                p = (T.BAULK_X + T.D_RADIUS * math.cos(a), T.D_RADIUS * math.sin(a), 0.0005)
                ls.drawTo(*p) if k else ls.moveTo(*p)
            unlit(self.scene.attachNewNode(ls.create()))
            for sx, sy in T.SPOTS.values():
                unlit(add(_disc(0.004, 12), (sx, sy, 0.0004), (0.8, 0.8, 0.8), tex=False))
        else:  # pool: head and foot spots, diamonds on the rails
            for sx in (POOL_HEAD_X, POOL_FOOT_X):
                unlit(add(_disc(0.006, 12), (sx, 0.0, 0.0004), (0.85, 0.85, 0.85), tex=False))
            y_rail = Wd / 2 + spec.cushion_w + spec.rail_w / 2
            x_rail = L / 2 + spec.cushion_w + spec.rail_w / 2
            for k in (1, 2, 3, 5, 6, 7):
                for sy in (-1, 1):
                    unlit(add(_disc(0.007, 10), (-L / 2 + k * L / 8, sy * y_rail, spec.rail_h + 0.0005),
                              (0.9, 0.88, 0.8), tex=False))
            for k in (1, 2, 3):
                for sx in (-1, 1):
                    unlit(add(_disc(0.007, 10), (sx * x_rail, -Wd / 2 + k * Wd / 4, spec.rail_h + 0.0005),
                              (0.9, 0.88, 0.8), tex=False))

    @staticmethod
    def _id_colour(i: int) -> tuple:
        v = i + 1  # 0 is background
        return ((v % 25) * 10 / 255, (v // 25) * 40 / 255, 1.0, 1)

    def _decode_ids(self, rgb: np.ndarray) -> np.ndarray:
        """Turn the ID image into a per-pixel ball index (-1 = no ball)."""
        r, g, b = (rgb[..., k].astype(np.int32) for k in range(3))
        v = np.rint(r / 10).astype(np.int32) + np.rint(g / 40).astype(np.int32) * 25
        return np.where(b > 128, v - 1, -1)

    # ----------------------------------------------------------------- camera + render

    def set_camera(self, pose: CameraPose) -> None:
        self.lens.setFov(pose.fov_h)
        self.lens.setAspectRatio(self.w / self.h)
        cam = self.base.camera
        cam.setPos(*pose.pos)
        cam.lookAt(*pose.look_at)
        cam.setR(pose.roll)

    def _grab(self, tex: Texture) -> np.ndarray:
        data = tex.getRamImageAs("RGB")
        img = np.frombuffer(bytes(data), np.uint8).reshape(tex.getYSize(), tex.getXSize(), 3)
        return img[::-1].copy()  # Panda's rows start at the bottom

    def render_frame(self, auto_exposure: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Returns (rgb image, per-pixel ball index).

        Like a real camera's auto-exposure, the lights are scaled so the cloth comes
        out well exposed (its brightest channel around `self.cloth_target`, 0..255)
        and not burnt out: a blown cloth turns flat cyan (pool) or lime (snooker),
        which real phone and robot cameras avoid.
        """
        too_bright = math.inf  # lowest exposure seen that burnt the cloth out
        cloth = None
        for attempt in range(8):
            self.base.graphicsEngine.renderFrame()
            self.base.graphicsEngine.renderFrame()  # second frame so shadows are up to date
            rgb = self._grab(self.color_tex)
            if not auto_exposure:
                break
            if cloth is None:  # which pixels are cloth: the ID image knows (any cloth colour)
                cloth = self._decode_ids(self._grab(self.id_tex)) == CLOTH_ID
            level = _cloth_level(rgb, cloth)
            if _blown_cloth(rgb, cloth) > 0.02:
                too_bright = min(too_bright, self.exposure)
                self._set_exposure(self.exposure * 0.72)
            elif attempt < 6 and level is not None and level > 1.15 * self.cloth_target:
                # Too bright (but not burnt out): darker, like a camera does for a dark cloth.
                too_bright = min(too_bright, self.exposure)
                self._set_exposure(self.exposure * max(0.6, self.cloth_target / level))
            elif (attempt < 6 and level is not None and level < 0.85 * self.cloth_target
                  and self.exposure < 4 and self.exposure < 0.8 * too_bright):
                # Brighter, but never back up to a level that burnt out before.
                self._set_exposure(min(self.exposure * min(1.6, self.cloth_target / max(level, 1)),
                                       0.85 * too_bright))
            else:
                break
        ids = self._decode_ids(self._grab(self.id_tex))
        self.cloth_mask = ids == CLOTH_ID
        return rgb, np.where(self.cloth_mask, -1, ids)  # callers only see balls and pockets

    def _set_exposure(self, exposure: float) -> None:
        self.exposure = exposure
        for light, (r, g, b, a) in getattr(self, "_light_colours", []):
            light.setColor((r * exposure, g * exposure, b * exposure, a))

    # ----------------------------------------------------------------- projection

    def world_to_pixel(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project Nx3 world points. Returns (Nx2 pixels, in-front-of-camera mask)."""
        m = np.array(self.render.getMat(self.base.cam), dtype=np.float64).reshape(4, 4)
        p = np.array(self.lens.getProjectionMat(), dtype=np.float64).reshape(4, 4)
        hom = np.hstack([pts, np.ones((len(pts), 1))]) @ m @ p  # Panda uses row vectors
        front = hom[:, 3] > 1e-6
        ndc = hom[:, :2] / np.where(front, hom[:, 3], 1)[:, None]
        px = np.stack([(ndc[:, 0] + 1) / 2 * self.w, (1 - ndc[:, 1]) / 2 * self.h], axis=1)
        return px, front

    def ball_outline(self, b: Ball, n: int = 64) -> np.ndarray | None:
        """Exact outline of the ball's silhouette in pixels (None if behind the camera)."""
        R = self.spec.ball_r
        c = np.array([b.x, b.y, R])
        e = np.array(self.base.cam.getPos(self.render))
        d = c - e
        dist = np.linalg.norm(d)
        if dist <= R * 1.01:
            return None
        u = d / dist
        # The silhouette is a circle on the sphere, in the plane normal to the view ray.
        centre = c - (R**2 / dist) * u
        rad = R * math.sqrt(1 - (R / dist) ** 2)
        a = np.cross(u, [0, 0, 1.0])
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(u, [1.0, 0, 0])
        a /= np.linalg.norm(a)
        bvec = np.cross(u, a)
        th = np.linspace(0, 2 * math.pi, n, endpoint=False)
        pts = centre + rad * (np.cos(th)[:, None] * a + np.sin(th)[:, None] * bvec)
        px, front = self.world_to_pixel(pts)
        return px if front.all() else None

    def pocket_outline(self, k: int, n: int = 48) -> np.ndarray | None:
        """Outline (pixels) of pocket k: the hole in the cloth plus its cut-out in the rail."""
        px, py, pr = self.spec.pockets[k]
        th = np.linspace(0, 2 * math.pi, n, endpoint=False)
        pts = []
        snap = _off_cloth((px, py), self.spec.length / 2, self.spec.width / 2)
        for z, r in ((0.0, pr), (self.spec.rail_h, pr * 1.1)):
            xy = [snap(r * c, r * s) if z > 0 else (r * c, r * s) for c, s in zip(np.cos(th), np.sin(th))]
            pts.append(np.array([[px + x, py + y, z] for x, y in xy]))
        px_, front = self.world_to_pixel(np.vstack(pts))
        if not front.all():
            return None
        return cv2.convexHull(px_.astype(np.float32)).reshape(-1, 2).astype(np.float64)

    # ----------------------------------------------------------------- drawing on the table

    def add_path(self, points, rgb=(1, 1, 1), thickness: float = 3.0, z: float = 0.002,
                 dashed: bool = False) -> None:
        """Draw a line through table points (x, y), just above the cloth."""
        ls = LineSegs()
        ls.setThickness(thickness)
        ls.setColor(*rgb, 1)
        pts = [np.array([p[0], p[1], z]) for p in points]
        for a, b in zip(pts, pts[1:]):
            if not dashed:
                ls.moveTo(*a)
                ls.drawTo(*b)
                continue
            n = max(1, int(np.linalg.norm(b - a) / 0.04))
            for k in range(0, n, 2):
                ls.moveTo(*(a + (b - a) * k / n))
                ls.drawTo(*(a + (b - a) * min(k + 1, n) / n))
        np_ = self.scene.attachNewNode(ls.create())
        np_.setShaderOff(1)
        np_.setLightOff(1)

    def add_ring(self, x: float, y: float, radius: float, rgb=(1, 1, 1), thickness: float = 3.0,
                 z: float = 0.002) -> None:
        th = np.linspace(0, 2 * math.pi, 41)
        self.add_path([(x + radius * math.cos(t), y + radius * math.sin(t)) for t in th], rgb,
                      thickness, z)

    def camera_info(self) -> dict:
        cam = self.base.cam
        fx = (self.w / 2) / math.tan(math.radians(self.lens.getHfov()) / 2)
        fy = (self.h / 2) / math.tan(math.radians(self.lens.getVfov()) / 2)
        return {
            "position": list(cam.getPos(self.render)),
            "hpr_deg": list(cam.getHpr(self.render)),
            "world_to_camera_rowmajor": [list(r) for r in np.array(self.render.getMat(cam)).reshape(4, 4)],
            "intrinsics": {"fx": fx, "fy": fy, "cx": self.w / 2, "cy": self.h / 2,
                           "width": self.w, "height": self.h},
            "eye_height_above_floor": cam.getZ(self.render) + self.spec.cloth_height,
        }
