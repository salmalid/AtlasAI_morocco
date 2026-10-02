"""The analysis-module contract.

Every AI module in the platform implements :class:`Module`. A module receives
an area, a time window and a pixel budget; it returns numbers, map layers,
provenance and — importantly — its own caveats.

Two design rules run through all of them:

1. **Physics first, learning second.** Each module ships a deterministic
   implementation grounded in published spectral science, so the platform
   produces real, defensible output on day one with no trained weights. Where a
   neural backend would do better, the module declares it via
   ``neural_backend`` and uses it when weights are present. Nothing here
   fabricates a model prediction it cannot actually make.

2. **Every number carries its uncertainty.** Results report the fraction of
   pixels actually observed and the caveats that apply, because a drought index
   computed from 30% cloud-free pixels is a different claim from one computed
   from 98%.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np

from app.core.geo import BBox
from app.eo.catalog import SensorId
from app.eo.raster import Grid

logger = logging.getLogger(__name__)

LayerKind = Literal["continuous", "categorical", "mask", "rgb"]


@dataclass
class AnalysisRequest:
    """What to analyse, where, and how hard to work at it."""

    bbox: BBox
    start: dt.date
    end: dt.date
    #: Optional earlier window. Change-detection modules require it; others
    #: use it to report a trend.
    baseline_start: dt.date | None = None
    baseline_end: dt.date | None = None
    resolution_m: float | None = None
    max_cloud: float = 30.0
    max_items: int = 12

    def window_days(self) -> int:
        return (self.end - self.start).days

    def with_window(self, start: dt.date, end: dt.date) -> "AnalysisRequest":
        return AnalysisRequest(
            bbox=self.bbox,
            start=start,
            end=end,
            resolution_m=self.resolution_m,
            max_cloud=self.max_cloud,
            max_items=self.max_items,
        )

    def has_baseline(self) -> bool:
        return self.baseline_start is not None and self.baseline_end is not None


@dataclass
class Layer:
    """A renderable raster produced by a module."""

    key: str
    name: str
    kind: LayerKind
    array: np.ndarray
    colormap: str = "gray"
    vmin: float = 0.0
    vmax: float = 1.0
    unit: str = ""
    description: str = ""
    #: For categorical layers: ``{class_id: (label, hex_colour)}``.
    palette: dict[int, tuple[str, str]] = field(default_factory=dict)
    opacity: float = 0.85
    #: Extra channels for ``kind="rgb"`` layers.
    rgb: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def descriptor(self) -> dict:
        """JSON view — the pixels stay server-side and stream as PNG."""
        return {
            "key": self.key,
            "name": self.name,
            "kind": self.kind,
            "colormap": self.colormap,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "unit": self.unit,
            "description": self.description,
            "opacity": self.opacity,
            "classes": [
                {"id": cid, "label": label, "color": colour}
                for cid, (label, colour) in sorted(self.palette.items())
            ],
        }


@dataclass
class ClassArea:
    """Area accounting for one class of a classified raster."""

    id: int
    label: str
    color: str
    pixels: int
    area_km2: float
    share: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "color": self.color,
            "area_km2": round(self.area_km2, 3),
            "share": round(self.share, 4),
            "pixels": self.pixels,
        }


@dataclass
class AnalysisResult:
    module: str
    title: str
    summary: str
    grid: Grid
    metrics: dict[str, object] = field(default_factory=dict)
    breakdown: list[ClassArea] = field(default_factory=list)
    layers: list[Layer] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    #: 0-1 self-assessed reliability, driven mostly by valid-pixel fraction.
    confidence: float = 1.0
    observed_dates: list[str] = field(default_factory=list)

    def layer(self, key: str) -> Layer | None:
        return next((layer for layer in self.layers if layer.key == key), None)

    def to_dict(self, *, layer_url: str | None = None) -> dict:
        payload = {
            "module": self.module,
            "title": self.title,
            "summary": self.summary,
            "metrics": self.metrics,
            "breakdown": [c.to_dict() for c in self.breakdown],
            "layers": [layer.descriptor() for layer in self.layers],
            "sources": self.sources,
            "caveats": self.caveats,
            "confidence": round(self.confidence, 3),
            "observed_dates": self.observed_dates,
            "grid": {
                "bbox": list(self.grid.bbox.as_tuple()),
                "corners": self.grid.lonlat_corners(),
                "width": self.grid.width,
                "height": self.grid.height,
                "resolution_m": round(self.grid.resolution_m, 2),
            },
        }
        if layer_url:
            for descriptor in payload["layers"]:
                descriptor["png_url"] = layer_url.format(layer=descriptor["key"])
        return payload


@runtime_checkable
class Module(Protocol):
    """Structural contract for an analysis module."""

    id: str
    title: str
    description: str
    sensors: tuple[SensorId, ...]
    #: Question this module answers, shown in the UI and used by the assistant.
    question: str
    #: True when the module cannot run without a baseline window.
    needs_baseline: bool
    #: Name of the neural architecture that would supersede the analytic
    #: implementation, if trained weights are supplied.
    neural_backend: str | None

    def run(self, request: AnalysisRequest) -> AnalysisResult: ...


# --- shared helpers ---------------------------------------------------------


def classify(array: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """Bin a continuous raster into ordinal classes 0..len(thresholds).

    NaN in, ``-1`` out — an unobserved pixel must never silently become
    class 0.
    """
    out = np.full(array.shape, -1, dtype=np.int16)
    valid = np.isfinite(array)
    codes = np.zeros(array.shape, dtype=np.int16)
    for threshold in thresholds:
        codes += (array >= threshold).astype(np.int16)
    out[valid] = codes[valid]
    return out


def class_areas(
    classes: np.ndarray,
    palette: dict[int, tuple[str, str]],
    grid: Grid,
) -> list[ClassArea]:
    """Per-class pixel counts converted to km², excluding unobserved pixels."""
    observed = int(np.count_nonzero(classes >= 0))
    out: list[ClassArea] = []
    for class_id, (label, colour) in sorted(palette.items()):
        pixels = int(np.count_nonzero(classes == class_id))
        out.append(
            ClassArea(
                id=class_id,
                label=label,
                color=colour,
                pixels=pixels,
                area_km2=pixels * grid.pixel_area_km2,
                share=pixels / observed if observed else 0.0,
            )
        )
    return out


def confidence_from_coverage(*arrays: np.ndarray) -> float:
    """Reliability score driven by how much of the AOI was actually seen.

    Below ~40% valid pixels the result is indicative at best, so the curve is
    deliberately punishing at the low end.
    """
    fractions = []
    for array in arrays:
        if array.size:
            fractions.append(float(np.count_nonzero(np.isfinite(array)) / array.size))
    if not fractions:
        return 0.0
    coverage = min(fractions)
    return round(float(np.clip(coverage**1.5, 0.0, 1.0)), 3)


def coverage_caveats(coverage: float, request: AnalysisRequest) -> list[str]:
    notes: list[str] = []
    if coverage < 0.4:
        notes.append(
            f"Only {coverage:.0%} of the area was cloud-free in this window — treat the "
            "numbers as indicative and widen the date range for a firmer result."
        )
    elif coverage < 0.75:
        notes.append(f"{coverage:.0%} of the area was observed; the rest was cloud or scene edge.")
    if request.window_days() < 10:
        notes.append(
            "Short time window: Sentinel-2 revisits every ~5 days, so few scenes contribute "
            "to this composite."
        )
    return notes


def percent(value: float) -> float:
    return round(float(value) * 100, 2)


def safe_mean(array: np.ndarray) -> float | None:
    finite = array[np.isfinite(array)]
    return round(float(finite.mean()), 4) if finite.size else None


def trend_label(delta: float, *, small: float = 0.02) -> str:
    if abs(delta) < small:
        return "stable"
    return "increasing" if delta > 0 else "decreasing"
