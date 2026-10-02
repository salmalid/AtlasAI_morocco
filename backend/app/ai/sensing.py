"""Scene loading shared by every analysis module.

A :class:`Scene` is "the pixels for one area, one sensor, one time window,
already cloud-masked, composited and in physical units". Modules ask for one
and then do science; none of them touch STAC or GDAL directly.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from pystac import Item

from app.ai.base import AnalysisRequest
from app.core.config import settings
from app.eo import catalog, indices, raster
from app.eo.catalog import COLLECTIONS, Collection, NoDataError, SensorId
from app.eo.raster import CompositeMethod, Grid

logger = logging.getLogger(__name__)


@dataclass
class Scene:
    """Analysis-ready pixels for one sensor and one time window."""

    sensor: SensorId
    collection: Collection
    grid: Grid
    bands: dict[str, np.ndarray]
    items: list[Item]
    start: dt.date
    end: dt.date
    _index_cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def index(self, key: str) -> np.ndarray:
        """Compute (and memoise) a spectral index from the loaded bands."""
        if key not in self._index_cache:
            self._index_cache[key] = indices.compute(key, self.bands)
        return self._index_cache[key]

    def band(self, key: str) -> np.ndarray:
        return self.bands[key]

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def dates(self) -> list[str]:
        return sorted({catalog.item_date(item).isoformat() for item in self.items})

    def sources(self) -> list[dict]:
        return catalog.describe(self.items)

    def coverage(self) -> float:
        if not self.bands:
            return 0.0
        return min(raster.valid_fraction(band) for band in self.bands.values())

    def observed_mask(self) -> np.ndarray:
        """True where every loaded band has data."""
        masks = [np.isfinite(band) for band in self.bands.values()]
        out = masks[0]
        for mask in masks[1:]:
            out = out & mask
        return out


def resolve_resolution(request: AnalysisRequest, collection: Collection) -> float:
    """Requested resolution, floored at the sensor's native GSD.

    Asking for 10 m from a 500 m MODIS product would fabricate detail that the
    instrument never captured, so the native resolution always wins.
    """
    requested = request.resolution_m or settings.default_resolution
    return max(float(requested), collection.resolution_m)


def load_scene(
    request: AnalysisRequest,
    bands: Sequence[str],
    *,
    sensor: SensorId = "sentinel-2",
    start: dt.date | None = None,
    end: dt.date | None = None,
    grid: Grid | None = None,
    method: CompositeMethod = "median",
    mask_clouds: bool = True,
    max_items: int | None = None,
) -> Scene:
    """Search, read, mask and composite one window of imagery.

    Pass an existing ``grid`` when two epochs must be pixel-aligned — every
    change-detection module depends on that alignment.
    """
    collection = COLLECTIONS[sensor]
    start = start or request.start
    end = end or request.end

    if grid is None:
        grid = Grid.from_bbox(request.bbox, resolve_resolution(request, collection))

    items = catalog.search(
        sensor,
        request.bbox,
        start,
        end,
        max_cloud=request.max_cloud if collection.has_cloud_cover else None,
        limit=max_items or request.max_items,
    )
    # Best-first ordering means truncating keeps the clearest scenes.
    items = items[: (max_items or request.max_items)]

    loaded = raster.load_bands(
        items, collection, list(bands), grid, method=method, mask_clouds=mask_clouds
    )
    scene = Scene(
        sensor=sensor,
        collection=collection,
        grid=grid,
        bands=loaded,
        items=items,
        start=start,
        end=end,
    )
    logger.info(
        "loaded %s: %d scenes, %dx%d @ %.0fm, coverage %.0f%%",
        sensor, len(items), grid.width, grid.height, grid.resolution_m, scene.coverage() * 100,
    )
    return scene


def load_with_fallback(
    request: AnalysisRequest,
    bands: Sequence[str],
    *,
    sensors: Sequence[SensorId] = ("sentinel-2", "landsat"),
    **kwargs,
) -> Scene:
    """Try each sensor in turn; the first that yields usable pixels wins.

    Sentinel-2 only reaches back to 2015 (and 2017 for reliable L2A), so any
    query about the 2013-2016 era has to fall through to Landsat.
    """
    errors: list[str] = []
    for sensor in sensors:
        try:
            scene = load_scene(request, bands, sensor=sensor, **kwargs)
            if scene.coverage() > 0.05:
                return scene
            errors.append(f"{sensor}: scenes found but {scene.coverage():.0%} valid pixels")
        except (NoDataError, ValueError) as exc:
            errors.append(f"{sensor}: {exc}")
    raise NoDataError(
        "No usable imagery from " + ", ".join(sensors) + ".",
        hint=" | ".join(errors),
    )


def seasonal_baseline_windows(
    end: dt.date,
    *,
    years: int = 5,
    pad_days: int = 20,
) -> list[tuple[dt.date, dt.date]]:
    """Same calendar window in each of the preceding ``years`` years.

    Drought and vegetation anomalies are only meaningful against the *same
    season* in prior years: an August NDVI of 0.25 is catastrophic in the Rif
    and unremarkable in the Drâa valley. Comparing like-with-like is what makes
    the anomaly a signal rather than an artefact of phenology.
    """
    windows = []
    for offset in range(1, years + 1):
        try:
            anchor = end.replace(year=end.year - offset)
        except ValueError:  # 29 February
            anchor = end.replace(year=end.year - offset, day=28)
        windows.append((anchor - dt.timedelta(days=pad_days), anchor + dt.timedelta(days=pad_days)))
    return windows
