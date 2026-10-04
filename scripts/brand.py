"""Render the Ask Physics logo, README banner, and Fermi model artwork.

    make brand                              # everything, from scratch
    python scripts/brand.py luna banner     # some pieces: logo, banner, luna, tellus, solem, celeste
    python scripts/brand.py --reuse banner  # reuse body renders cached in build/brand

The bodies are rendered with numpy, not drawn: the Moon is a cratered sphere
lit with the Lommel-Seeliger law; the Earth wraps NASA's Blue Marble and NOAA
relief around a sphere with procedural clouds, ocean glint, city lights, and
an atmosphere; the Sun has limb darkening, granulation, sunspots, spicules,
and loop prominences; and the black hole is ray-traced through Schwarzschild
geodesics with a Doppler-beamed accretion disk and relativistic jets.
Headless Chromium then typesets the equations and labels on top.

Needs numpy, Pillow, and Node with Playwright. The first run downloads Google
Fonts and the Earth maps (basemap-data from PyPI) into build/brand.
"""

from __future__ import annotations

import base64
import json
import os
import re
import struct
import subprocess
import sys
import urllib.request
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
BUILD = ROOT / "build" / "brand"
ART = 1200  # model artwork is ART x ART pixels

Array = NDArray[np.float64]

# ---------------------------------------------------------------- image helpers


def write_png(path: Path, rgb: Array) -> None:
    """Write an HxWx3 float image in [0, 1] as an 8-bit PNG (no Pillow needed)."""
    data = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    height, width, _ = data.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in data)

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def blur(img: Array, sigma: float) -> Array:
    """Gaussian blur through the FFT, zero-padded so nothing wraps around."""
    pad = int(3 * sigma) + 1
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)))
    h, w, _ = padded.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    kernel = np.exp(-2 * np.pi**2 * sigma**2 * (fx**2 + fy**2))
    out = np.fft.irfft2(
        np.fft.rfft2(padded, axes=(0, 1)) * kernel[..., None], s=(h, w), axes=(0, 1)
    )
    return np.asarray(out[pad:-pad, pad:-pad])


def bloom(img: Array, threshold: float, strength: float, sigmas: tuple[float, ...]) -> Array:
    bright = np.clip(img - threshold, 0, None)
    return img + strength * sum((blur(bright, s) for s in sigmas), np.zeros_like(img))


def tonemap(img: Array, exposure: float = 1.0) -> Array:
    """ACES filmic curve, then sRGB gamma."""
    x = img * exposure
    mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return np.asarray(np.clip(mapped, 0, 1) ** (1 / 2.2))


def downsample(img: Array, factor: int) -> Array:
    h, w, c = img.shape
    return np.asarray(img.reshape(h // factor, factor, w // factor, factor, c).mean(axis=(1, 3)))


def smoothstep(e0: float, e1: float, x: Array) -> Array:
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return np.asarray(t * t * (3 - 2 * t))


def normalize(v: Array) -> Array:
    return np.asarray(v / np.linalg.norm(v, axis=-1, keepdims=True))


def palette(t: Array, stops: list[tuple[float, tuple[float, float, float]]]) -> Array:
    xs = [s[0] for s in stops]
    return np.stack([np.interp(t, xs, [s[1][i] for s in stops]) for i in range(3)], axis=-1)


# ---------------------------------------------------------------------- noise

_GRADIENTS = np.array(
    [
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1],
    ],
    dtype=np.float64,
)  # fmt: skip


def _hash(
    ix: NDArray[np.int64], iy: NDArray[np.int64], iz: NDArray[np.int64], seed: int
) -> NDArray[np.uint32]:
    h = (
        ix.astype(np.uint32) * np.uint32(0x8DA6B343)
        ^ iy.astype(np.uint32) * np.uint32(0xD8163841)
        ^ iz.astype(np.uint32) * np.uint32(0xCB1AB31F)
        ^ np.uint32(seed * 0x27D4EB2F & 0xFFFFFFFF)
    )
    h ^= h >> np.uint32(13)
    h *= np.uint32(0x5BD1E995)
    h ^= h >> np.uint32(15)
    return np.asarray(h, dtype=np.uint32)


def noise(p: Array, seed: int = 0) -> Array:
    """3D gradient (Perlin-style) noise in roughly [-1, 1]; ``p`` has shape (..., 3)."""
    cell = np.floor(p)
    f = p - cell
    i = cell.astype(np.int64)
    u = f * f * f * (f * (f * 6 - 15) + 10)
    total = np.zeros(p.shape[:-1])
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                g = _GRADIENTS[_hash(i[..., 0] + dx, i[..., 1] + dy, i[..., 2] + dz, seed) & 15]
                dot = (
                    g[..., 0] * (f[..., 0] - dx)
                    + g[..., 1] * (f[..., 1] - dy)
                    + g[..., 2] * (f[..., 2] - dz)
                )
                wx = u[..., 0] if dx else 1 - u[..., 0]
                wy = u[..., 1] if dy else 1 - u[..., 1]
                wz = u[..., 2] if dz else 1 - u[..., 2]
                total += dot * wx * wy * wz
    return total


def fbm(p: Array, octaves: int, seed: int = 0, gain: float = 0.5) -> Array:
    total = np.zeros(p.shape[:-1])
    amp = 1.0
    for octave in range(octaves):
        total += amp * noise(p * 2.0**octave, seed + 101 * octave)
        amp *= gain
    return total


def hash01(n: NDArray[np.int64], seed: int) -> Array:
    return np.asarray(_hash(n, n * 7 + 3, n * 13 + 5, seed) / 2.0**32)


# --------------------------------------------------------------------- scenes


def sphere(n: int, cx: float, cy: float, radius: float) -> tuple[Array, Array, Array, Array]:
    """Unit-sphere normals for a disk of ``radius`` pixels centered at (cx, cy)."""
    j, i = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    x = (j - cx) / radius
    y = (cy - i) / radius
    r = np.sqrt(x * x + y * y)
    z = np.sqrt(np.clip(1 - r * r, 0, None))
    return x, y, z, r


def rotation(yaw: float, pitch: float) -> Array:
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return np.asarray(rx @ ry)


def rotate(v: Array, yaw: float, pitch: float) -> Array:
    return np.asarray(v @ rotation(yaw, pitch).T)


def starfield(h: int, w: int, count: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3))
    ys, xs = rng.integers(0, h, count), rng.integers(0, w, count)
    mag = rng.pareto(2.2, count) * 0.25 + 0.05
    tint = palette(
        rng.random(count), [(0, (1.0, 0.8, 0.6)), (0.5, (1, 1, 1)), (1, (0.7, 0.8, 1.0))]
    )
    np.add.at(img, (ys, xs), tint * np.minimum(mag, 6)[:, None])
    return blur(img, 0.7 * w / 1200) * 3.5


def render_luna(n: int = ART) -> Array:
    ss = 2
    size = n * ss
    radius = 0.36 * size
    x, y, z, r = sphere(size, 0.56 * size, 0.5 * size, radius)
    inside = r < 1
    normal = np.stack([x, y, z], axis=-1)
    view = rotation(0.9, 0.25)
    pin = normal[inside] @ view.T  # body coordinates of the visible pixels

    # Maria (dark basalt plains), highlands, and fine regolith texture.
    maria = smoothstep(-0.1, 0.15, fbm(pin * 1.3, 6, seed=1))
    albedo = 0.62 - 0.33 * maria + 0.1 * fbm(pin * 4, 5, seed=2) + 0.05 * fbm(pin * 22, 3, seed=3)

    # Craters: a power-law size distribution of bowls with raised rims.
    rng = np.random.default_rng(7)
    height = np.zeros(pin.shape[0])
    for _ in range(1800):
        c = normalize(rng.normal(size=3))
        if (c @ view)[2] < -0.1:  # on the far side
            continue
        rad = min(0.006 * (1 - rng.random()) ** (-1 / 1.4), 0.14)
        cosd = pin @ c
        near = np.nonzero(cosd > np.cos(2.5 * rad))[0]
        if near.size == 0:
            continue
        t = np.sqrt(np.clip(2 * (1 - cosd[near]), 0, None)) / rad
        depth = 0.2 * rad
        bowl = np.where(t < 1, depth * (t * t - 1), 0.0)
        rim = 0.25 * depth * np.exp(-(((t - 1) / 0.22) ** 2))
        ejecta = np.where(t > 1, 0.1 * depth * np.exp(-(t - 1) * 2.5), 0.0)
        height[near] += bowl + rim + ejecta
        if rng.random() < 0.1:  # young craters are brighter, with a halo
            albedo[near] += 0.12 * np.exp(-((t / 1.8) ** 2))

    height += 0.004 * fbm(pin * 40, 4, seed=4)
    grid = np.zeros(z.shape)
    grid[inside] = height
    gy, gx = np.gradient(grid)
    bump = 0.7 * radius
    lit_normal = normalize(np.stack([x - bump * gx, y + bump * gy, z], axis=-1))[inside]

    # A crescent: the Sun is behind the Moon and off to the right.
    light = normalize(np.array([0.9, 0.24, -0.62]))
    mu0 = np.clip(lit_normal @ light, 0, None)
    mu = np.clip(z[inside], 1e-3, None)
    shade = 0.8 * 2 * mu0 / (mu0 + mu) + 0.2 * mu0
    terminator = smoothstep(-0.03, 0.06, normal[inside] @ light)
    lit = (albedo * shade * terminator)[:, None] * np.array([1.0, 0.96, 0.9]) * 0.95
    earthshine = (0.012 * albedo * (0.4 + 0.6 * mu))[:, None] * np.array([0.6, 0.75, 1.0])

    img = starfield(size, size, 900, seed=11) * (~inside)[..., None]
    img[inside] = lit + earthshine
    img = downsample(img, ss)
    img = bloom(img, 0.8, 0.25, (6, 24))
    return tonemap(img, 1.1)


# NASA's Blue Marble Next Generation and NOAA's ETOPO1 relief (both public
# domain), as packaged in basemap-data on PyPI.
EARTH_WHEEL = (
    "https://files.pythonhosted.org/packages/b6/2a/"
    "b99e7e00092e3f2d659e6e7203985cdd2f78c9c68786a41316a1e81bdb05/"
    "basemap_data-2.0.0-py3-none-any.whl"
)
EARTH_WHEEL_SHA256 = "56103a0dfa411c77bec1888b232bb70e7ceb089f83679531fbb4bb6efafa5035"


def earth_maps() -> tuple[Array, Array]:
    """(albedo RGB, elevation) equirectangular maps in [0, 1], downloaded once."""
    import hashlib
    import io
    import zipfile

    from PIL import Image  # only the Earth needs Pillow: pip install pillow

    wheel = BUILD / "basemap_data.whl"
    if not wheel.exists():
        data = _fetch(EARTH_WHEEL)
        if hashlib.sha256(data).hexdigest() != EARTH_WHEEL_SHA256:
            raise SystemExit("basemap-data wheel failed its checksum")
        wheel.parent.mkdir(parents=True, exist_ok=True)
        wheel.write_bytes(data)
    with zipfile.ZipFile(wheel) as archive:

        def load(name: str) -> Array:
            raw = archive.read(f"mpl_toolkits/basemap_data/{name}.jpg")
            return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float64) / 255

        return load("bmng"), load("etopo1").mean(axis=2)


def sample(texture: Array, lat: Array, lon: Array) -> Array:
    """Bilinear lookup in an equirectangular map (rows run north to south)."""
    h, w = texture.shape[:2]
    u = (lon / (2 * np.pi) + 0.5) * w - 0.5
    v = (0.5 - lat / np.pi) * h - 0.5
    u0, v0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fu, fv = u - u0, v - v0
    if texture.ndim == 3:
        fu, fv = fu[:, None], fv[:, None]
    x0, x1 = u0 % w, (u0 + 1) % w
    y0, y1 = np.clip(v0, 0, h - 1), np.clip(v0 + 1, 0, h - 1)
    top = texture[y0, x0] * (1 - fu) + texture[y0, x1] * fu
    bottom = texture[y1, x0] * (1 - fu) + texture[y1, x1] * fu
    return np.asarray(top * (1 - fv) + bottom * fv)


def render_tellus(n: int = ART) -> Array:
    ss = 2
    size = n * ss
    radius = 0.6 * size
    x, y, z, r = sphere(size, 0.5 * size, 0.83 * size, radius)
    inside = r < 1
    normal = np.stack([x, y, z], axis=-1)
    nin = normal[inside]

    # Face Africa and Europe, north up with a little roll.
    lat0, lon0, roll = np.radians(8), np.radians(14), np.radians(-8)
    center = np.array([np.cos(lat0) * np.sin(lon0), np.sin(lat0), np.cos(lat0) * np.cos(lon0)])
    east = np.array([np.cos(lon0), 0.0, -np.sin(lon0)])
    north = np.cross(center, east)
    east, north = (
        np.cos(roll) * east + np.sin(roll) * north,
        -np.sin(roll) * east + np.cos(roll) * north,
    )
    p = nin[:, :1] * east + nin[:, 1:2] * north + nin[:, 2:3] * center
    lat = np.arcsin(np.clip(p[:, 1], -1, 1))
    lon = np.arctan2(p[:, 0], p[:, 2])

    albedo_map, relief_map = earth_maps()
    surface = sample(albedo_map, lat, lon) ** 2.2  # sRGB to linear
    relief = sample(relief_map, lat, lon)
    blue = surface[:, 2] - 0.5 * (surface[:, 0] + surface[:, 1])
    ocean = smoothstep(0.004, 0.02, blue) * smoothstep(0.12, 0.04, surface.mean(axis=1))
    surface *= (1 - 0.3 * ocean)[:, None]  # Blue Marble's oceans are a touch too bright
    land = 1 - ocean
    ice = smoothstep(0.35, 0.6, surface.mean(axis=1))
    green = smoothstep(0.0, 0.03, surface[:, 1] - surface[:, 0]) * land

    # Relief: perturb the normals with the ETOPO elevation (land only).
    grid = np.zeros(z.shape)
    grid[inside] = relief * land
    gy, gx = np.gradient(grid)
    bump = 9.0
    lit_normal = normalize(np.stack([x - bump * gx, y + bump * gy, z], axis=-1))[inside]

    # Clouds: fractal, lightly swirled, denser away from the subtropical deserts.
    cwarp = np.stack([fbm(p * 2.5 + s, 4, seed=20 + s) for s in (0, 1, 2)], axis=-1)
    cq = p + 0.25 * cwarp
    cloud = fbm(cq * np.array([4.0, 6.0, 4.0]), 8, seed=23, gain=0.55)
    belt = 0.12 * np.cos(lat * 6) ** 2 - 0.1 * np.exp(-(((np.abs(lat) - 0.42) / 0.12) ** 2))
    cover = smoothstep(0.18, 0.6, cloud + belt) * 0.8

    light = normalize(np.array([-0.85, 0.42, 0.32]))
    ndl = nin @ light
    day = smoothstep(-0.05, 0.3, ndl)[:, None]
    relief_light = np.clip(lit_normal @ light, 0, None) / np.clip(ndl, 0.05, None)
    relief_light = (1 + (np.clip(relief_light, 0, 2) - 1) * land)[:, None]
    half = normalize(light + np.array([0.0, 0.0, 1.0]))
    spec = np.clip(nin @ half, 0, None) ** 300 * 0.8 + np.clip(nin @ half, 0, None) ** 30 * 0.04
    glint = (spec * ocean * (1 - cover))[:, None] * np.array([1, 0.9, 0.72])

    lit = (surface * relief_light + glint) * day
    clouds = cover[:, None] * np.array([0.9, 0.92, 0.96]) * day
    shaded = lit * (1 - cover[:, None]) + clouds

    # Night side: city lights scattered over inhabited (green) land.
    towns = smoothstep(0.45, 0.8, fbm(p * 60, 3, seed=30)) * smoothstep(
        -0.15, 0.25, fbm(p * 6, 3, seed=31)
    )
    night = smoothstep(0.0, -0.15, ndl)
    glow = (towns * (0.25 + 0.75 * green) * land * (1 - ice) * night * (1 - 0.8 * cover))[:, None]
    shaded += glow * np.array([1.0, 0.62, 0.25]) * 0.5

    # Atmosphere: blue limb haze, a thin sunset band, and an outer halo.
    mu = np.clip(nin[:, 2], 0, 1)
    haze = ((1 - mu) ** 3 * smoothstep(-0.25, 0.4, ndl))[:, None] * np.array([0.2, 0.42, 1.0])
    sunset = (np.exp(-(((ndl - 0.02) / 0.05) ** 2)) * (1 - mu) ** 3)[:, None] * np.array(
        [1.0, 0.38, 0.12]
    )

    radial = np.stack([x, y, np.zeros_like(x)], axis=-1) / np.maximum(r, 1e-6)[..., None]
    facing = smoothstep(-0.35, 0.55, radial @ light)
    t = np.clip(r - 1, 0, None)
    halo = np.exp(-t / 0.01) * 0.8 + np.exp(-t / 0.045) * 0.2
    img = ((~inside) * halo * facing)[..., None] * np.array([0.3, 0.55, 1.0])
    img += starfield(size, size, 700, seed=12) * (~inside)[..., None]
    img[inside] = shaded + haze * 0.9 + sunset * 0.25
    img = downsample(img, ss)
    img = bloom(img, 0.7, 0.25, (5, 20))
    return tonemap(img, 1.15)


def render_solem(n: int = ART) -> Array:
    ss = 2
    size = n * ss
    radius = 0.27 * size
    x, y, z, r = sphere(size, 0.5 * size, 0.5 * size, radius)
    inside = r < 1
    normal = np.stack([x, y, z], axis=-1)
    p = rotate(normal, 0.4, 0.1)
    mu = np.clip(z, 0, 1)

    limb = 1 - 0.62 * (1 - mu) - 0.2 * (1 - mu) ** 2
    cells = np.abs(noise(p * 110, seed=40))
    granules = smoothstep(0.0, 0.3, cells) * 0.26 + 0.74
    supergran = (1 + 0.08 * fbm(p * 9, 4, seed=41)) * (1 + 0.14 * fbm(p * 26, 4, seed=46))

    # Sunspots: dark umbrae with streaky penumbrae in the active latitudes.
    rng = np.random.default_rng(5)
    spots = np.ones(z.shape)
    for _ in range(9):
        lat = rng.choice([-1, 1]) * rng.uniform(0.12, 0.38)
        lon = rng.uniform(-1.1, 1.1)
        c = np.array([np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat)])
        size_ = rng.uniform(0.012, 0.035)
        d = np.sqrt(np.clip(2 * (1 - p @ c), 0, None)) / size_
        streak = 0.75 + 0.25 * noise(p * 300, seed=42)
        spots *= (
            1
            - 0.8 * smoothstep(0.55, 0.35, d)
            - 0.35 * streak * smoothstep(1.2, 0.8, d) * (d > 0.45)
        )
    faculae = 1 + 0.35 * smoothstep(0.35, 0.8, fbm(p * 20, 3, seed=43)) * (1 - mu) ** 2

    intensity = np.clip(limb * granules * supergran * faculae * spots, 0, None)
    disk = palette(
        intensity,
        [(0.0, (0.2, 0.02, 0.0)), (0.4, (0.75, 0.16, 0.01)), (0.7, (1.0, 0.42, 0.04)),
         (0.95, (1.0, 0.68, 0.22)), (1.15, (1.0, 0.88, 0.55))],
    ) * (1.7 * intensity)[..., None]  # fmt: skip
    disk *= inside[..., None]

    # Corona with streamers, and prominences licking off the limb.
    dirs = np.stack([x, y], axis=-1) / np.maximum(r, 1e-6)[..., None]
    angular = np.stack([dirs[..., 0] * 3, dirs[..., 1] * 3, np.log(np.maximum(r, 1)) * 0.6], -1)
    streamers = 0.55 + 0.9 * np.clip(fbm(angular, 5, seed=44), -0.5, 1)
    t = np.clip(r - 1, 0, None)
    corona = np.exp(-t / 0.03) * 1.1 + np.exp(-t / 0.11) * 0.4 * streamers + 0.04 / (1 + t) ** 3
    corona_color = np.array([1.0, 0.55, 0.18])
    # Spicules: a fringe of thin radial jets right at the limb.
    spicule_p = np.stack([dirs[..., 0] * 22, dirs[..., 1] * 22, t * 3], axis=-1)
    flames = smoothstep(0.05, 0.45, fbm(spicule_p, 4, seed=45)) * np.exp(-t / 0.016)
    # Prominences: glowing magnetic loops anchored at two footpoints on the limb.
    prominences = np.zeros(z.shape)
    for angle, width, height in ((2.3, 0.09, 0.13), (-0.5, 0.06, 0.09), (4.0, 0.12, 0.07)):
        foot = np.array([np.cos(angle), np.sin(angle)])
        tangent = np.array([-foot[1], foot[0]])
        a = (x - foot[0]) * tangent[0] + (y - foot[1]) * tangent[1]
        b = (x - foot[0]) * foot[0] + (y - foot[1]) * foot[1]
        ring = np.sqrt((a / width) ** 2 + (b / height) ** 2)
        strands = np.stack([a * 60, b * 25, ring * 4], -1)
        strand = 0.2 + 1.2 * np.clip(fbm(strands, 4, seed=47) + 0.3, 0, 1)
        loop = np.exp(-(((ring - 1) * min(width, height) / 0.018) ** 2)) * strand
        loop += 0.15 * smoothstep(1.0, 0.5, ring) * strand  # plasma inside the arch
        prominences += 0.4 * loop * (b > 0) * (r > 1)
    outside = (~inside)[..., None]
    halo = (
        corona[..., None] * corona_color
        + flames[..., None] * np.array([1.0, 0.3, 0.08]) * 4.0
        + prominences[..., None] * np.array([1.0, 0.16, 0.05]) * 4.0
    )
    img = disk + halo * outside

    stars = starfield(size, size, 300, seed=13) * outside * 0.6
    img = downsample(img + stars, ss)
    img = bloom(img, 1.0, 0.3, (8, 40, 120))
    return tonemap(img, 0.85)


def render_celeste(n: int = ART) -> Array:
    """Ray-trace a Schwarzschild black hole (r_s = 1) with a thin disk and jets."""
    # Camera a little above the disk plane, with a slight roll for drama.
    dist, elev, fov, roll = 40.0, np.radians(8.5), np.radians(40), np.radians(-12)
    cam = np.array([0.0, dist * np.sin(elev), -dist * np.cos(elev)])
    fwd = normalize(-cam)
    right = normalize(np.cross(np.array([0.0, 1.0, 0.0]), fwd))
    up = np.cross(fwd, right)
    j, i = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    u = (2 * j / n - 1) * np.tan(fov / 2)
    v = (1 - 2 * i / n) * np.tan(fov / 2)
    u, v = u * np.cos(roll) - v * np.sin(roll), u * np.sin(roll) + v * np.cos(roll)
    dirs = normalize(fwd + u[..., None] * right + v[..., None] * up).reshape(-1, 3)

    count = dirs.shape[0]
    color = np.zeros((count, 3))
    jet_glow = np.zeros(count)
    pos = np.tile(cam, (count, 1))
    vel = dirs.copy()
    h2 = np.sum(np.cross(pos, vel) ** 2, axis=1)
    alive = np.arange(count)
    r_in, r_out = 3.0, 12.0

    def accel(x: Array, hh: Array) -> Array:
        rr = np.linalg.norm(x, axis=1)
        return np.asarray(-1.5 * hh[:, None] * x / rr[:, None] ** 5)

    for _ in range(1400):
        if alive.size == 0:
            break
        x, vv, hh = pos[alive], vel[alive], h2[alive]
        rr = np.linalg.norm(x, axis=1)
        dt = np.clip(0.06 * rr, 0.015, 0.8)[:, None]
        a = accel(x, hh)
        v_half = vv + 0.5 * a * dt
        x_new = x + v_half * dt
        v_new = v_half + 0.5 * accel(x_new, hh) * dt
        r_new = np.linalg.norm(x_new, axis=1)

        # Relativistic jets: glowing cones along the spin axis.
        cyl = np.hypot(x_new[:, 0], x_new[:, 2])
        ay = np.abs(x_new[:, 1])
        width = 0.12 + 0.045 * ay
        jet = (
            np.exp(-((cyl / width) ** 2))
            * smoothstep(1.2, 3.0, ay)
            * np.exp(-ay / 22)
            * (0.6 + 0.4 * np.cos(ay * 1.3) ** 2)
        )
        jet_glow[alive] += jet * dt[:, 0] * 0.4

        done = np.zeros(alive.size, dtype=bool)
        crossed = x[:, 1] * x_new[:, 1] < 0
        if crossed.any():
            k = np.nonzero(crossed)[0]
            t = x[k, 1] / (x[k, 1] - x_new[k, 1])
            hit = x[k] + (x_new[k] - x[k]) * t[:, None]
            rc = np.hypot(hit[:, 0], hit[:, 2])
            on_disk = (rc > r_in) & (rc < r_out)
            if on_disk.any():
                kk, hp, rd = k[on_disk], hit[on_disk], rc[on_disk]
                temp = rd**-0.75 * (1 - np.sqrt(r_in / rd)) ** 0.25
                temp /= 0.214  # the profile peaks at r = 49/36 r_in; scale that to 1
                tangent = np.stack([-hp[:, 2], np.zeros_like(rd), hp[:, 0]], axis=1) / rd[:, None]
                beta = np.sqrt(0.5 / (rd - 1)) * 0.75
                gamma = 1 / np.sqrt(1 - beta**2)
                toward = -normalize(v_new[kk])
                doppler = 1 / (gamma * (1 - beta * np.sum(tangent * toward, axis=1)))
                g = doppler * np.sqrt(1 - 1 / rd)
                ang = np.arctan2(hp[:, 2], hp[:, 0])
                spiral = ang + 2.2 * np.log(rd)
                tex_p = np.stack([np.cos(spiral) * 2.5, np.sin(spiral) * 2.5, rd * 0.9], axis=1)
                tex = 0.55 + 0.7 * np.clip(fbm(tex_p, 5, seed=50) + 0.5, 0, 1.4)
                fade = smoothstep(r_out, r_out * 0.55, rd) * smoothstep(r_in, r_in * 1.08, rd)
                shifted = np.clip(temp * g, 0, 2)
                emit = (temp**2.2 * g**3.2 * tex * fade)[:, None] * 1.3
                tint = palette(
                    shifted,
                    [(0.0, (0.3, 0.04, 0.0)), (0.35, (0.85, 0.25, 0.04)), (0.65, (1.0, 0.6, 0.25)),
                     (0.95, (1.0, 0.88, 0.7)), (1.3, (0.85, 0.9, 1.0)), (2.0, (0.6, 0.75, 1.0))],
                )  # fmt: skip
                color[alive[kk]] = emit * tint
                done[kk] = True

        captured = r_new < 1.02
        escaped = (r_new > 60) & (np.sum(x_new * v_new, axis=1) > 0)
        if escaped.any():
            e = np.nonzero(escaped & ~done)[0]
            d = normalize(v_new[e])
            color[alive[e]] = sky(d)
        done |= captured | escaped
        pos[alive], vel[alive] = x_new, v_new
        alive = alive[~done]

    jets = jet_glow[:, None] * np.array([0.5, 0.65, 1.0])
    img = (color + jets).reshape(n, n, 3)
    img = bloom(img, 0.9, 0.22, (3, 14, 50))
    return tonemap(img, 0.95)


def sky(d: Array) -> Array:
    """Background stars and a faint galactic band, looked up by direction."""
    k = 240
    cell = np.floor(d * k).astype(np.int64)
    key = (cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663) ^ (cell[:, 2] * 83492791)
    is_star = hash01(key, 60) < 0.05
    jitter = np.stack([hash01(key, 61 + a) for a in range(3)], axis=1)
    center = normalize((cell + jitter) / k)
    spot = np.exp(-np.sum((d - center) ** 2, axis=1) * k * k / 0.04)
    mag = hash01(key, 64) ** 8 * 5 + 0.15
    stars = (is_star * spot * mag)[:, None] * palette(
        hash01(key, 65), [(0, (1.0, 0.8, 0.65)), (0.5, (1, 1, 1)), (1, (0.7, 0.8, 1.0))]
    )
    band_axis = normalize(np.array([0.3, 0.9, 0.3]))
    band = np.exp(-((d @ band_axis) ** 2) / 0.05) * (0.5 + 0.5 * fbm(d * 3, 4, seed=66))
    nebula = band[:, None] * np.array([0.05, 0.035, 0.08])
    return np.asarray(stars * 1.3 + nebula)


BODIES: dict[str, Callable[[], Array]] = {
    "luna": render_luna,
    "tellus": render_tellus,
    "solem": render_solem,
    "celeste": render_celeste,
}


def body_image(name: str, force: bool = False) -> Path:
    """Render a body to build/brand/<name>.png, reusing a cached render."""
    path = BUILD / f"{name}.png"
    if force or not path.exists():
        print(f"rendering {name}...", flush=True)
        write_png(path, BODIES[name]())
    return path


# ----------------------------------------------------------------- typesetting

FONT_FAMILIES = (
    "family=Space+Grotesk:wght@500;700"
    "&family=JetBrains+Mono:wght@400;600"
    "&family=STIX+Two+Text:ital,wght@0,400;1,400"
)
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124"


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return bytes(response.read())


def font_css() -> str:
    """Google Fonts CSS with every font file inlined, cached after the first fetch."""
    cache = BUILD / "fonts.css"
    if not cache.exists():
        css = _fetch(f"https://fonts.googleapis.com/css2?{FONT_FAMILIES}&display=block").decode()

        def inline(match: re.Match[str]) -> str:
            data = base64.b64encode(_fetch(match.group(1))).decode()
            return f"url(data:font/woff2;base64,{data})"

        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(re.sub(r"url\((https://[^)]+)\)", inline, css))
    return cache.read_text()


def data_url(path: Path) -> str:
    kind = "svg+xml" if path.suffix == ".svg" else "png"
    return f"data:image/{kind};base64,{base64.b64encode(path.read_bytes()).decode()}"


BASE_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #000; overflow: hidden; }
.eq { position: absolute; font-family: 'STIX Two Text', 'DejaVu Serif', serif; font-style: italic;
      color: #e4ebff; white-space: nowrap; letter-spacing: 0.01em;
      text-shadow: 0 0 14px rgba(139, 233, 253, 0.35), 0 0 2px rgba(0, 0, 0, 0.9); }
.eq .up { font-style: normal; }
.eq sub, .eq sup { font-size: 0.62em; line-height: 0; }
.name { font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #fff;
        letter-spacing: -0.01em; }
.mono { font-family: 'JetBrains Mono', monospace; color: #a6accd; }
"""


# Each equation: (HTML, left px, top px, size px, rotation deg, opacity).
Equation = tuple[str, int, int, int, float, float]

EQUATIONS: dict[str, list[Equation]] = {
    "luna": [
        ("F = <span class=up>G</span> m<sub>1</sub>m<sub>2</sub> / r<sup>2</sup>", 54, 60, 46, -2, 0.85),
        ("T<sup>2</sup> = 4π<sup>2</sup>a<sup>3</sup> / <span class=up>G</span>M", 640, 54, 34, 2, 0.6),
        ("v = √(<span class=up>G</span>M / r)", 46, 300, 38, -4, 0.7),
        ("ω = 2π / T", 150, 470, 32, 0, 0.55),
        ("L = m v r", 80, 650, 30, 3, 0.5),
        ("E = ½mv<sup>2</sup> − <span class=up>G</span>Mm / r", 330, 520, 34, -6, 0.38),
        ("F<sub>tide</sub> ≈ 2<span class=up>G</span>MmR / d<sup>3</sup>", 360, 720, 30, -6, 0.3),
        ("x = x<sub>0</sub> + v<sub>0</sub>t + ½at<sup>2</sup>", 40, 830, 32, 2, 0.55),
        ("g<sub>☾</sub> = 1.62 m/s<sup>2</sup>", 860, 1100, 30, 0, 0.6),
        ("a = 384 400 km", 950, 140, 26, 0, 0.45),
    ],
    "tellus": [
        ("F = ma", 760, 46, 52, 0, 0.9),
        ("∇ · <b>E</b> = ρ / ε<sub>0</sub>", 60, 168, 38, -3, 0.7),
        ("∇ × <b>B</b> = μ<sub>0</sub><b>J</b> + μ<sub>0</sub>ε<sub>0</sub> ∂<b>E</b>/∂t", 60, 236, 28, -4, 0.5),
        ("g = <span class=up>G</span>M / R<sup>2</sup> ≈ 9.81 m/s<sup>2</sup>", 470, 176, 32, 0, 0.65),
        ("v<sub>esc</sub> = √(2<span class=up>G</span>M / R)", 880, 186, 32, 4, 0.65),
        ("PV = nRT", 1010, 110, 28, 0, 0.5),
        ("<b>F</b><sub>c</sub> = −2m <b>Ω</b> × <b>v</b>", 56, 300, 24, -2, 0.45),
        ("λ = v / f", 590, 110, 26, 0, 0.45),
        ("τ = <b>r</b> × <b>F</b>", 1000, 290, 28, 6, 0.5),
    ],
    "solem": [
        ("E = mc<sup>2</sup>", 60, 56, 56, -2, 0.9),
        ("L = 4πR<sup>2</sup>σT<sup>4</sup>", 790, 64, 38, 2, 0.7),
        ("4 <sup>1</sup>H → <sup>4</sup>He + 2e<sup>+</sup> + 2ν<sub>e</sub>", 330, 150, 30, 0, 0.55),
        ("λ<sub>max</sub> T = b", 960, 430, 32, 6, 0.6),
        ("E = hν", 60, 420, 34, -5, 0.6),
        ("dP/dr = −<span class=up>G</span>m ρ / r<sup>2</sup>", 40, 780, 30, 3, 0.5),
        ("t<sub>KH</sub> ≈ <span class=up>G</span>M<sup>2</sup> / RL", 930, 760, 30, -4, 0.5),
        ("σ = 5.67 × 10<sup>−8</sup> W m<sup>−2</sup> K<sup>−4</sup>", 700, 1110, 26, 0, 0.5),
    ],
    "celeste": [
        ("r<sub>s</sub> = 2<span class=up>G</span>M / c<sup>2</sup>", 60, 60, 48, -2, 0.9),
        ("T<sub>H</sub> = ħc<sup>3</sup> / 8π<span class=up>G</span>Mk<sub>B</sub>", 800, 70, 32, 2, 0.65),
        ("S = k<sub>B</sub>c<sup>3</sup>A / 4<span class=up>G</span>ħ", 70, 230, 30, -3, 0.55),
        ("G<sub>μν</sub> = 8π<span class=up>G</span> T<sub>μν</sub> / c<sup>4</sup>", 830, 230, 30, 3, 0.55),
        ("L<sub>Edd</sub> = 4π<span class=up>G</span>Mm<sub>p</sub>c / σ<sub>T</sub>", 740, 930, 30, -2, 0.55),
        ("ds<sup>2</sup> = −(1 − r<sub>s</sub>/r) c<sup>2</sup>dt<sup>2</sup> + dr<sup>2</sup>/(1 − r<sub>s</sub>/r) + r<sup>2</sup>dΩ<sup>2</sup>", 560, 1046, 22, 0, 0.5),
        ("r<sub>isco</sub> = 3 r<sub>s</sub>", 930, 840, 28, 4, 0.5),
    ],
}  # fmt: skip

ROLES = {
    "luna": "test model",
    "tellus": "classifier",
    "solem": "planner and explainer",
    "celeste": "the last resort",
}

LABEL_AT_TOP = {"tellus"}


def art_page(name: str) -> str:
    from askphysics.lm.config import get_config

    model = f"fermi-{name}-1"
    params = get_config(model).num_parameters() / 1e6
    count = f"{params:.2f}M" if params < 1 else f"{params:.1f}M"
    equations = "\n".join(
        f'<div class="eq" style="left:{x}px;top:{y}px;font-size:{size}px;opacity:{alpha};'
        f'transform:rotate({angle}deg)">{html}</div>'
        for html, x, y, size, angle, alpha in EQUATIONS[name]
    )
    place = "top:44px" if name in LABEL_AT_TOP else "bottom:46px"
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{font_css()}
{BASE_CSS}
.art {{ position: absolute; inset: 0; width: {ART}px; height: {ART}px; }}
.vignette {{ position: absolute; inset: 0;
  background: radial-gradient(circle at 50% 50%, transparent 60%, rgba(0,0,0,0.45) 100%); }}
.label {{ position: absolute; left: 52px; {place}; }}
.label .name {{ font-size: 56px; line-height: 1; }}
.label .mono {{ font-size: 20px; margin-top: 12px; }}
</style></head><body style="width:{ART}px;height:{ART}px">
<img class="art" src="{data_url(body_image(name))}">
<div class="vignette"></div>
{equations}
<div class="label"><div class="name">{model}</div>
<div class="mono">~{count} parameters · {ROLES[name]}</div></div>
</body></html>"""


def banner_page() -> str:
    stars = BUILD / "banner-stars.png"
    if not stars.exists():
        write_png(stars, tonemap(starfield(800, 2560, 1400, seed=21)))
    bodies = [
        # (name, diameter, crop position, zoom, mask): sizes grow with the models.
        ("luna", 52, "73% 50%", "135%", "none"),
        ("tellus", 72, "50% 22%", "150%", "radial-gradient(circle, #000 58%, transparent 71%)"),
        ("solem", 94, "50% 50%", "175%", "radial-gradient(circle, #000 58%, transparent 71%)"),
        ("celeste", 128, "50% 50%", "150%", "radial-gradient(circle, #000 58%, transparent 71%)"),
    ]
    row = "\n".join(
        f'<figure><div class="body" style="width:{d}px;height:{d}px;'
        f"background-image:url({data_url(body_image(n))});background-position:{pos};"
        f'background-size:{zoom};-webkit-mask-image:{mask}"></div>'
        f"<figcaption>{n}</figcaption></figure>"
        for n, d, pos, zoom, mask in bodies
    )
    watermark = " · ".join(
        [
            "F = ma",
            "E = mc²",
            "v² = v₀² + 2ad",
            "∇·E = ρ/ε₀",
            "pV = nRT",
            "λ = h/p",
            "F = Gm₁m₂/r²",
            "τ = r × F",
            "ΔS ≥ 0",
            "E = hν",
        ]
        * 3
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{font_css()}
{BASE_CSS}
body {{ width: 1280px; height: 400px; position: relative;
  background: radial-gradient(ellipse at 20% 30%, #1d2042 0%, #0b0c1a 55%, #05060c 100%); }}
.stars {{ position: absolute; inset: 0; width: 1280px; height: 400px; mix-blend-mode: screen; }}
.wm {{ position: absolute; left: -40px; font-family: 'STIX Two Text', serif; font-style: italic;
  color: #8be9fd; opacity: 0.07; white-space: nowrap; font-size: 22px; }}
.logo {{ position: absolute; left: 64px; top: 86px; width: 228px; height: 228px;
  filter: drop-shadow(0 10px 30px rgba(139, 233, 253, 0.25)); }}
.title {{ position: absolute; left: 326px; top: 92px; }}
.title .name {{ font-size: 88px; line-height: 1; }}
.title .name span {{ background: linear-gradient(90deg, #8be9fd, #bd93f9);
  -webkit-background-clip: text; background-clip: text; color: transparent; }}
.tag {{ font-family: 'Space Grotesk', sans-serif; font-weight: 500; font-size: 27px;
  color: #d6dbf0; margin-top: 18px; }}
.title .mono {{ font-size: 16px; margin-top: 16px; letter-spacing: 0.02em; }}
.row {{ position: absolute; right: 44px; top: 120px; display: flex; align-items: flex-end;
  gap: 14px; }}
figure {{ display: flex; flex-direction: column; align-items: center; gap: 10px; }}
.body {{ border-radius: 50%; mix-blend-mode: screen; }}
figcaption {{ font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #8b93a7; }}
</style></head><body>
<img class="stars" src="{data_url(stars)}">
<div class="wm" style="top:18px">{watermark}</div>
<div class="wm" style="top:352px;left:-260px">{watermark}</div>
<img class="logo" src="{data_url(OUT / "logo.svg")}">
<div class="title">
  <div class="name">Ask <span>Physics</span></div>
  <div class="tag">Physics answers you can check.</div>
  <div class="mono">our own models · SymPy + Pint · no APIs</div>
</div>
<div class="row">{row}</div>
</body></html>"""


def logo_page() -> str:
    return f"""<!doctype html><html><head><style>body{{margin:0;background:transparent}}
img{{display:block;width:512px;height:512px}}</style></head><body>
<img src="{data_url(OUT / "logo.svg")}"></body></html>"""


def shoot(jobs: list[dict[str, object]]) -> None:
    env = dict(os.environ)
    env.setdefault(
        "NODE_PATH",
        subprocess.run(
            ["npm", "root", "-g"], capture_output=True, text=True, check=True
        ).stdout.strip(),
    )
    spec = BUILD / "jobs.json"
    spec.write_text(json.dumps(jobs))
    subprocess.run(["node", str(ROOT / "scripts" / "shoot.mjs"), str(spec)], check=True, env=env)


def page(name: str, html: str) -> str:
    path = BUILD / f"{name}.html"
    path.write_text(html, encoding="utf-8")
    return str(path)


def main(args: list[str]) -> None:
    reuse = "--reuse" in args
    targets = [a for a in args if a != "--reuse"] or ["logo", "banner", *BODIES]
    for target in targets:
        if target in BODIES:
            body_image(target, force=not reuse)
    jobs: list[dict[str, object]] = []
    for target in targets:
        if target == "logo":
            jobs.append(
                {"html": page("logo", logo_page()), "out": str(OUT / "logo.png"),
                 "width": 512, "height": 512, "transparent": True}
            )  # fmt: skip
        elif target == "banner":
            jobs.append(
                {"html": page("banner", banner_page()), "out": str(OUT / "banner.png"),
                 "width": 1280, "height": 400, "scale": 2}
            )  # fmt: skip
        elif target in BODIES:
            jobs.append(
                {"html": page(target, art_page(target)), "out": str(OUT / f"fermi-{target}-1.jpg"),
                 "width": ART, "height": ART, "type": "jpeg", "quality": 88}
            )  # fmt: skip
        else:
            raise SystemExit(f"unknown target {target!r}")
    shoot(jobs)


if __name__ == "__main__":
    main(sys.argv[1:])
