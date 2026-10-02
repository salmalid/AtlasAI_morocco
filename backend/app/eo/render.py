"""Raster -> PNG rendering for map overlays.

Colormaps are defined as explicit hex stops and interpolated with NumPy. That
keeps the render path dependency-light and, more importantly, makes the palette
a reviewable artefact: a drought map's colours encode a claim about severity,
so the breakpoints belong in source, not in a library default.

NaN pixels render fully transparent, so a cloud gap reads as "no observation"
on the map rather than as a value.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

RGBA = np.ndarray


def _hex(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


@dataclass(frozen=True)
class Colormap:
    name: str
    stops: tuple[str, ...]

    def lut(self, size: int = 256) -> np.ndarray:
        """``(size, 3)`` uint8 lookup table interpolated across the stops."""
        anchors = np.array([_hex(s) for s in self.stops], dtype=np.float32)
        positions = np.linspace(0.0, 1.0, len(anchors))
        target = np.linspace(0.0, 1.0, size)
        channels = [np.interp(target, positions, anchors[:, i]) for i in range(3)]
        return np.stack(channels, axis=1).round().astype(np.uint8)


COLORMAPS: dict[str, Colormap] = {
    # Sparse-to-dense vegetation: desert sand through scrub to closed canopy.
    "vegetation": Colormap("vegetation", ("#a6611a", "#dfc27d", "#f5f5dc", "#a6d96a", "#1a9641", "#00441b")),
    # Water: dry basin (tan) to deep water (navy).
    "water": Colormap("water", ("#f0e3c8", "#d9f0f7", "#74c2e1", "#2b7bba", "#08306b")),
    "moisture": Colormap("moisture", ("#8c510a", "#dfc27d", "#f6e8c3", "#c7eae5", "#35978f", "#01665e")),
    # Drought severity runs the other way: green is safe, dark red is extreme.
    "drought": Colormap("drought", ("#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c", "#7f0000")),
    "urban": Colormap("urban", ("#f7f7f7", "#cccccc", "#969696", "#636363", "#d94801", "#7f2704")),
    "soil": Colormap("soil", ("#004529", "#78c679", "#ffffcc", "#fed976", "#e6550d", "#7f2704")),
    "burn": Colormap("burn", ("#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921")),
    "turbidity": Colormap("turbidity", ("#08306b", "#4292c6", "#9ecae1", "#fdd0a2", "#e6550d", "#7f2704")),
    "algae": Colormap("algae", ("#08306b", "#2b7bba", "#78c679", "#addd8e", "#238443", "#004529")),
    "elevation": Colormap("elevation", ("#2b7bba", "#c7e9b4", "#78c679", "#fed976", "#d95f0e", "#8c2d04", "#ffffff")),
    "nightlights": Colormap("nightlights", ("#000004", "#1b0c41", "#4f0d6c", "#a52c60", "#ed6925", "#fcffa4")),
    "risk": Colormap("risk", ("#1a9850", "#a6d96a", "#ffffbf", "#fdae61", "#d73027", "#67001f")),
    # Diverging: change detection, where sign matters more than magnitude.
    "change": Colormap("change", ("#67001f", "#d6604d", "#f7f7f7", "#4393c3", "#053061")),
    "gray": Colormap("gray", ("#000000", "#ffffff")),
}


def colorize(
    array: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    colormap: str = "vegetation",
    opacity: float = 1.0,
) -> RGBA:
    """Map a float raster to RGBA, NaN -> transparent."""
    cmap = COLORMAPS.get(colormap, COLORMAPS["gray"])
    lut = cmap.lut()

    valid = np.isfinite(array)
    span = vmax - vmin
    if abs(span) < 1e-12:
        span = 1.0
    normalised = np.clip((np.nan_to_num(array, nan=vmin) - vmin) / span, 0.0, 1.0)
    idx = (normalised * (len(lut) - 1)).round().astype(np.int32)

    rgba = np.zeros((*array.shape, 4), dtype=np.uint8)
    rgba[..., :3] = lut[idx]
    rgba[..., 3] = (valid * np.clip(opacity, 0.0, 1.0) * 255).astype(np.uint8)
    return rgba


def colorize_classes(array: np.ndarray, palette: dict[int, str], *, opacity: float = 1.0) -> RGBA:
    """Render a categorical raster from an explicit ``{class_id: hex}`` map."""
    rgba = np.zeros((*array.shape, 4), dtype=np.uint8)
    alpha = int(np.clip(opacity, 0.0, 1.0) * 255)
    codes = np.nan_to_num(array, nan=-1).astype(np.int32)
    for class_id, colour in palette.items():
        hit = codes == class_id
        if not hit.any():
            continue
        rgba[hit, :3] = _hex(colour)
        rgba[hit, 3] = alpha
    return rgba


def colorize_mask(mask: np.ndarray, colour: str = "#2b7bba", *, opacity: float = 0.75) -> RGBA:
    """Render a boolean detection mask; everything outside it is transparent."""
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    hit = np.nan_to_num(mask, nan=0).astype(bool)
    rgba[hit, :3] = _hex(colour)
    rgba[hit, 3] = int(np.clip(opacity, 0.0, 1.0) * 255)
    return rgba


def to_png(rgba: RGBA) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def rgb_composite(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    *,
    percentile: float = 2.0,
) -> RGBA:
    """True/false-colour composite with a percentile stretch per channel.

    A linear 0-1 stretch renders Moroccan scenes almost black because bare
    rock and desert occupy a narrow bright band; clipping the tails restores
    usable contrast.
    """
    channels = []
    valid = np.isfinite(red) & np.isfinite(green) & np.isfinite(blue)
    for band in (red, green, blue):
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            channels.append(np.zeros(band.shape, dtype=np.uint8))
            continue
        lo, hi = np.percentile(finite, [percentile, 100 - percentile])
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        stretched = np.clip((np.nan_to_num(band, nan=lo) - lo) / (hi - lo), 0, 1)
        channels.append((stretched * 255).astype(np.uint8))

    rgba = np.zeros((*red.shape, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = channels
    rgba[..., 3] = (valid * 255).astype(np.uint8)
    return rgba


def legend(colormap: str, vmin: float, vmax: float, steps: int = 7) -> list[dict]:
    """Legend swatches for the frontend, sampled evenly across the range."""
    cmap = COLORMAPS.get(colormap, COLORMAPS["gray"])
    lut = cmap.lut()
    out = []
    for i in range(steps):
        fraction = i / max(steps - 1, 1)
        r, g, b = lut[int(fraction * (len(lut) - 1))]
        out.append(
            {
                "value": round(vmin + fraction * (vmax - vmin), 3),
                "color": f"#{r:02x}{g:02x}{b:02x}",
            }
        )
    return out
