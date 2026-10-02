"""Spectral indices.

Each index is a published, peer-reviewed band ratio — the physical layer the
AI modules reason on top of. Keeping them in one registry means a module never
hard-codes band maths, and the API can expose any index as a map layer for
free.

All functions take a ``{band_name: array}`` dict in surface-reflectance units
and return float32 arrays with NaN preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

Bands = dict[str, np.ndarray]


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised difference ``(a - b) / (a + b)`` with a safe denominator."""
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = a + b
        out = np.where(np.abs(denom) < 1e-9, np.nan, (a - b) / denom)
    return out.astype(np.float32)


@dataclass(frozen=True)
class Index:
    key: str
    name: str
    description: str
    requires: tuple[str, ...]
    compute: Callable[[Bands], np.ndarray]
    #: Physically meaningful display range.
    vmin: float
    vmax: float
    colormap: str
    reference: str = ""

    def __call__(self, bands: Bands) -> np.ndarray:
        missing = [b for b in self.requires if b not in bands]
        if missing:
            raise ValueError(f"{self.key} needs band(s) {', '.join(missing)}")
        return self.compute(bands)


def _ndvi(b: Bands) -> np.ndarray:
    return _ratio(b["nir"], b["red"])


def _ndwi(b: Bands) -> np.ndarray:
    return _ratio(b["green"], b["nir"])


def _mndwi(b: Bands) -> np.ndarray:
    return _ratio(b["green"], b["swir16"])


def _ndbi(b: Bands) -> np.ndarray:
    return _ratio(b["swir16"], b["nir"])


def _ndmi(b: Bands) -> np.ndarray:
    return _ratio(b["nir"], b["swir16"])


def _nbr(b: Bands) -> np.ndarray:
    return _ratio(b["nir"], b["swir22"])


def _bsi(b: Bands) -> np.ndarray:
    """Bare Soil Index — separates bare ground from both vegetation and built-up."""
    with np.errstate(divide="ignore", invalid="ignore"):
        num = (b["swir16"] + b["red"]) - (b["nir"] + b["blue"])
        den = (b["swir16"] + b["red"]) + (b["nir"] + b["blue"])
        out = np.where(np.abs(den) < 1e-9, np.nan, num / den)
    return out.astype(np.float32)


def _savi(b: Bands, soil_factor: float = 0.5) -> np.ndarray:
    """Soil-Adjusted Vegetation Index (Huete 1988).

    NDVI saturates and misreads where vegetation is sparse and bright soil
    dominates the pixel — which describes most of Morocco outside the north.
    SAVI's L term suppresses that soil background.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        nir, red = b["nir"], b["red"]
        den = nir + red + soil_factor
        out = np.where(np.abs(den) < 1e-9, np.nan, ((nir - red) / den) * (1 + soil_factor))
    return out.astype(np.float32)


def _evi(b: Bands) -> np.ndarray:
    """Enhanced Vegetation Index — de-saturates over dense canopy (Huete 2002)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        nir, red, blue = b["nir"], b["red"], b["blue"]
        den = nir + 6.0 * red - 7.5 * blue + 1.0
        out = np.where(np.abs(den) < 1e-9, np.nan, 2.5 * (nir - red) / den)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _awei(b: Bands) -> np.ndarray:
    """Automated Water Extraction Index, shadow-suppressing form (Feyisa 2014).

    Outperforms NDWI where terrain shadow in the Atlas would otherwise be
    misread as water.
    """
    out = 4.0 * (b["green"] - b["swir16"]) - (0.25 * b["nir"] + 2.75 * b["swir22"])
    return out.astype(np.float32)


def _ndti(b: Bands) -> np.ndarray:
    """Normalised Difference Turbidity Index (Lacaux 2007) — suspended sediment."""
    return _ratio(b["red"], b["green"])


def _chl_ratio(b: Bands) -> np.ndarray:
    """Blue/green ratio, a first-order chlorophyll-a proxy in inland water."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(b["green"] < 1e-6, np.nan, b["blue"] / b["green"])
    return out.astype(np.float32)


REGISTRY: dict[str, Index] = {
    idx.key: idx
    for idx in (
        Index("ndvi", "NDVI", "Vegetation greenness and photosynthetic vigour",
              ("nir", "red"), _ndvi, -0.2, 0.9, "vegetation", "Rouse et al. 1974"),
        Index("savi", "SAVI", "Vegetation index corrected for bright soil background",
              ("nir", "red"), _savi, -0.2, 0.9, "vegetation", "Huete 1988"),
        Index("evi", "EVI", "Vegetation index resistant to canopy saturation",
              ("nir", "red", "blue"), _evi, -0.2, 0.9, "vegetation", "Huete et al. 2002"),
        Index("ndwi", "NDWI", "Open-water delineation from green/NIR contrast",
              ("green", "nir"), _ndwi, -0.6, 0.8, "water", "McFeeters 1996"),
        Index("mndwi", "MNDWI", "Water index using SWIR — better over built-up areas",
              ("green", "swir16"), _mndwi, -0.6, 0.8, "water", "Xu 2006"),
        Index("awei", "AWEI", "Shadow-suppressing water extraction index",
              ("green", "nir", "swir16", "swir22"), _awei, -1.0, 1.0, "water", "Feyisa et al. 2014"),
        Index("ndmi", "NDMI", "Canopy moisture content — early drought stress signal",
              ("nir", "swir16"), _ndmi, -0.6, 0.6, "moisture", "Gao 1996"),
        Index("ndbi", "NDBI", "Built-up surface index",
              ("swir16", "nir"), _ndbi, -0.5, 0.5, "urban", "Zha et al. 2003"),
        Index("bsi", "BSI", "Bare soil / desertification index",
              ("blue", "red", "nir", "swir16"), _bsi, -0.5, 0.6, "soil", "Rikimaru et al. 2002"),
        Index("nbr", "NBR", "Normalised Burn Ratio — fire severity",
              ("nir", "swir22"), _nbr, -0.6, 0.9, "burn", "Key & Benson 2006"),
        Index("ndti", "NDTI", "Water turbidity / suspended sediment proxy",
              ("red", "green"), _ndti, -0.3, 0.4, "turbidity", "Lacaux et al. 2007"),
        Index("chl", "Chlorophyll ratio", "Blue/green algal-pigment proxy",
              ("blue", "green"), _chl_ratio, 0.5, 2.0, "algae", "O'Reilly et al. 1998"),
    )
}


def compute(key: str, bands: Bands) -> np.ndarray:
    if key not in REGISTRY:
        raise KeyError(f"unknown index '{key}'; known: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[key](bands)


def required_bands(keys: Sequence[str]) -> list[str]:
    """Union of bands needed for a set of indices — one fetch, many indices."""
    needed: list[str] = []
    for key in keys:
        for band in REGISTRY[key].requires:
            if band not in needed:
                needed.append(band)
    return needed


def stats(array: np.ndarray) -> dict[str, float | None]:
    """Robust summary of an index raster, NaN-safe and JSON-safe."""
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "count": 0, "valid_fraction": 0.0,
            "mean": None, "median": None, "std": None,
            "min": None, "max": None, "p10": None, "p90": None,
        }
    q10, q50, q90 = (float(v) for v in np.percentile(finite, [10, 50, 90]))
    return {
        "count": int(finite.size),
        "valid_fraction": round(float(finite.size / array.size), 4),
        "mean": round(float(np.mean(finite)), 4),
        "median": round(q50, 4),
        "std": round(float(np.std(finite)), 4),
        "min": round(float(np.min(finite)), 4),
        "max": round(float(np.max(finite)), 4),
        "p10": round(q10, 4),
        "p90": round(q90, 4),
    }
