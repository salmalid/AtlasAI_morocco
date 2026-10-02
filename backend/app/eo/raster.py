"""Windowed reads of remote Cloud-Optimized GeoTIFFs.

The platform never downloads whole scenes. It defines an analysis grid, then
uses GDAL's range-request reader to pull only the pixels that fall inside it,
warping each source scene from its native UTM projection into that grid on the
fly. A national-scale query and a neighbourhood-scale query therefore cost
roughly the same number of bytes.

Grids are EPSG:3857 so rendered overlays drop straight onto a web map. Because
Mercator inflates distance by 1/cos(latitude), the pixel size is deflated by
the same factor at grid construction — so one pixel really does span
``resolution_m`` on the ground, and pixel-count statistics convert to km²
without a further correction.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import planetary_computer
import rasterio
from pystac import Item
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_bounds

from app.core.config import settings
from app.core.geo import BBox
from app.eo.catalog import Collection, NoDataError

logger = logging.getLogger(__name__)

WEB_MERCATOR = CRS.from_epsg(3857)
WGS84 = CRS.from_epsg(4326)

#: GDAL tuning for reading COGs over HTTPS. Disabling directory listing is the
#: single biggest win — without it GDAL issues a request per sibling file.
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff,.jp2",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_HTTP_TIMEOUT": "60",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "33554432",
}

CompositeMethod = Literal["median", "first", "max", "min", "mean"]


@dataclass(frozen=True)
class Grid:
    """The raster lattice every band in one analysis is resampled onto."""

    bbox: BBox
    transform: Affine
    width: int
    height: int
    resolution_m: float
    crs: CRS = WEB_MERCATOR

    @classmethod
    def from_bbox(cls, bbox: BBox, resolution_m: float, *, max_pixels: int | None = None) -> "Grid":
        max_pixels = max_pixels or settings.max_pixels
        resolution_m = bbox.fit_resolution(resolution_m, max_pixels)

        left, bottom, right, top = transform_bounds(WGS84, WEB_MERCATOR, *bbox.as_tuple(), densify_pts=21)
        # One Mercator metre is 1/cos(lat) ground metres; shrink the pixel so
        # its *ground* footprint matches the requested resolution.
        _, centre_lat = bbox.centroid
        merc_res = resolution_m / max(math.cos(math.radians(centre_lat)), 1e-6)

        width = max(int(round((right - left) / merc_res)), 1)
        height = max(int(round((top - bottom) / merc_res)), 1)
        transform = Affine.translation(left, top) * Affine.scale(merc_res, -merc_res)
        return cls(bbox, transform, width, height, resolution_m)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def pixel_area_km2(self) -> float:
        return (self.resolution_m**2) / 1e6

    @property
    def mercator_bounds(self) -> tuple[float, float, float, float]:
        left, top = self.transform * (0, 0)
        right, bottom = self.transform * (self.width, self.height)
        return (left, bottom, right, top)

    def lonlat_corners(self) -> list[list[float]]:
        """Corner coordinates in the order MapLibre's image source expects."""
        left, bottom, right, top = transform_bounds(
            self.crs, WGS84, *self.mercator_bounds, densify_pts=21
        )
        return [[left, top], [right, top], [right, bottom], [left, bottom]]

    def index_of(self, lon: float, lat: float) -> tuple[int, int] | None:
        """Row/col of a lon/lat point, or ``None`` if outside the grid."""
        xs, ys = warp_transform(WGS84, self.crs, [lon], [lat])
        col, row = ~self.transform * (xs[0], ys[0])
        row, col = int(row), int(col)
        if 0 <= row < self.height and 0 <= col < self.width:
            return row, col
        return None


def sign_href(href: str) -> str:
    """Attach a short-lived SAS token to a Planetary Computer blob URL.

    Tokens are cached per storage container inside ``planetary_computer``, so
    this is one network call per container per hour no matter how many assets
    are read. Anonymous access is rate-limited, hence the backoff; a free
    subscription key raises the ceiling substantially.
    """
    if "blob.core.windows.net" not in href:
        return href  # not a PC-hosted asset; nothing to sign

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            return planetary_computer.sign(href)
        except Exception as exc:  # noqa: BLE001 - the SDK raises bare requests errors
            last_error = exc
            if attempt < 3:
                delay = 2**attempt
                logger.warning("SAS signing failed (attempt %d), retrying in %ds", attempt + 1, delay)
                time.sleep(delay)

    raise NoDataError(
        f"Could not obtain a read token for {href.split('?')[0]}: {last_error}",
        hint=(
            "The Planetary Computer anonymous rate limit is easy to hit when several "
            "analyses run in quick succession. Wait a moment and retry, or set "
            "ATLASAI_PC_SUBSCRIPTION_KEY to raise the limit — it is free and requires "
            "no payment details: https://planetarycomputer.developer.azure-api.net/"
        ),
    )


def read_band(item: Item, asset_key: str, grid: Grid, *, resampling: Resampling | None = None) -> np.ndarray:
    """Read one asset of one item onto ``grid`` as float32 with NaN nodata."""
    asset = item.assets.get(asset_key)
    if asset is None:
        raise NoDataError(f"Item {item.id} has no asset '{asset_key}'")
    href = sign_href(asset.href)

    # Nearest for categorical/quality layers, bilinear for continuous ones.
    if resampling is None:
        resampling = Resampling.nearest if _is_categorical(asset_key) else Resampling.bilinear

    with rasterio.Env(**GDAL_ENV):
        try:
            level = _overview_level(href, grid)
            open_kwargs = {"OVERVIEW_LEVEL": level} if level is not None else {}
            with rasterio.open(href, **open_kwargs) as src:
                nodata = _nodata_for(src)
                with WarpedVRT(
                    src,
                    crs=grid.crs,
                    transform=grid.transform,
                    width=grid.width,
                    height=grid.height,
                    resampling=resampling,
                    src_nodata=nodata,
                    nodata=nodata,
                    warp_extras={"NUM_THREADS": "ALL_CPUS"},
                ) as vrt:
                    # masked=True routes the nodata value into the mask, which we
                    # then flatten to NaN for uniform downstream maths.
                    data = vrt.read(1, masked=True)
        except RasterioIOError as exc:
            raise NoDataError(
                f"Could not read {asset_key} from {item.id}: {exc}",
                hint="The scene may have been withdrawn, or the signed URL expired — retry.",
            ) from exc

    # Cast before filling: QA and class bands come back as integer masked
    # arrays, and NaN has no integer representation.
    return data.astype(np.float32).filled(np.nan)


def _nodata_for(src) -> float | int:
    """The nodata value to warp with, invented if the source declares none.

    This guards against silent data fabrication. When an AOI spans a tile
    boundary — routine for the 1x1-degree Copernicus DEM — the warper has no
    source pixels for the uncovered part of the grid and fills it with **zero**.
    If the source declares no nodata value, that zero is reported as valid data:
    ``read_masks`` shows nothing masked while half the array is fabricated sea
    level. Averaged into a statistic, it silently drags the answer toward zero.

    Supplying an explicit nodata makes the warper mark those pixels instead, so
    they become NaN and are excluded. The sentinel is chosen outside each
    dtype's meaningful range: NaN for floats, and the type maximum for integers
    (255 for the 0-100 land-cover codes, 65535 for 0-10000 reflectance).
    """
    if src.nodata is not None:
        return src.nodata
    dtype = np.dtype(src.dtypes[0])
    if np.issubdtype(dtype, np.floating):
        return float("nan")
    return int(np.iinfo(dtype).max)


def _overview_level(href: str, grid: Grid) -> int | None:
    """Pick the coarsest overview still finer than the target grid.

    This is the difference between a usable platform and an unusable one. A
    Sentinel-2 band is 10 m; rendering a 30 m grid over a city means GDAL would
    otherwise fetch ~9x more pixels than it can use, turning a two-second read
    into a two-minute one. COGs ship a pyramid of pre-decimated overviews for
    exactly this, but the warper does not select one on its own.

    Returns a 0-based overview index, or ``None`` to read full resolution.
    """
    key = href.split("?")[0]  # the SAS token changes hourly; the file does not
    if key in _OVERVIEW_CACHE:
        entry = _OVERVIEW_CACHE[key]
    else:
        try:
            with rasterio.open(href) as src:
                entry = (src.overviews(1), abs(src.transform.a))
        except RasterioIOError:
            entry = None
        if len(_OVERVIEW_CACHE) > 4096:
            _OVERVIEW_CACHE.clear()
        _OVERVIEW_CACHE[key] = entry

    if entry is None:
        return None
    factors, native_res = entry
    if not factors or native_res <= 0:
        return None

    # Grid resolution is in ground metres; source resolution is in its own CRS
    # units, which for the UTM zones these products use is also metres.
    wanted = grid.resolution_m / native_res
    if wanted < 2:
        return None

    # Stay strictly finer than the target so resampling still has detail to
    # work with — never upsample an over-decimated overview back up.
    usable = [i for i, factor in enumerate(factors) if factor <= wanted]
    return max(usable) if usable else None


#: ``{unsigned_href: (overview_factors, native_resolution)}``, or ``None`` when
#: the header could not be read. Bounded and cleared wholesale — it is a
#: latency optimisation, not a correctness requirement.
_OVERVIEW_CACHE: dict[str, tuple[list[int], float] | None] = {}


def read_stack(
    items: Sequence[Item],
    asset_key: str,
    grid: Grid,
    *,
    max_workers: int = 8,
) -> np.ndarray:
    """Read the same asset from several items into a ``(n, h, w)`` cube.

    Reads run concurrently: they are network-bound, and a composite over ten
    scenes is otherwise ten sequential round trips.
    """
    if not items:
        raise NoDataError("No items to read")

    def _safe_read(item: Item) -> np.ndarray | None:
        try:
            return read_band(item, asset_key, grid)
        except NoDataError as exc:
            logger.warning("skipping %s: %s", item.id, exc)
            return None

    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        layers = [layer for layer in pool.map(_safe_read, items) if layer is not None]

    if not layers:
        raise NoDataError(f"Every scene failed to provide asset '{asset_key}'")
    return np.stack(layers)


def composite(cube: np.ndarray, method: CompositeMethod = "median") -> np.ndarray:
    """Collapse a ``(n, h, w)`` cube to ``(h, w)``, ignoring NaNs.

    Median over several dates is the standard cheap cloud-free composite: a
    cloud is bright and transient, so it loses to the clear-sky majority.
    """
    if cube.ndim == 2:
        return cube
    with np.errstate(all="ignore"):
        if method == "median":
            return np.nanmedian(cube, axis=0).astype(np.float32)
        if method == "mean":
            return np.nanmean(cube, axis=0).astype(np.float32)
        if method == "max":
            return np.nanmax(cube, axis=0).astype(np.float32)
        if method == "min":
            return np.nanmin(cube, axis=0).astype(np.float32)
        if method == "first":
            out = np.full(cube.shape[1:], np.nan, dtype=np.float32)
            for layer in cube:
                fill = np.isnan(out) & ~np.isnan(layer)
                out[fill] = layer[fill]
            return out
    raise ValueError(f"unknown composite method: {method}")


def load_bands(
    items: Sequence[Item],
    collection: Collection,
    bands: Sequence[str],
    grid: Grid,
    *,
    method: CompositeMethod = "median",
    mask_clouds: bool = True,
) -> dict[str, np.ndarray]:
    """Load named logical bands as a cloud-masked, scaled, composited dict.

    This is the workhorse every AI module calls. It returns physical units
    (surface reflectance 0-1, backscatter in linear power, metres of
    elevation), so downstream index maths needs no per-sensor special cases.
    """
    missing = [b for b in bands if b not in collection.bands]
    if missing:
        raise ValueError(f"{collection.stac_id} has no band(s): {', '.join(missing)}")

    clear: np.ndarray | None = None
    if mask_clouds and collection.qa_asset:
        clear = _clear_sky_cube(items, collection, grid)

    out: dict[str, np.ndarray] = {}
    for band in bands:
        cube = read_stack(items, collection.bands[band], grid)
        if collection.scale != 1.0 or collection.offset != 0.0:
            cube = cube * collection.scale + collection.offset
        if clear is not None:
            cube = np.where(clear, cube, np.nan)
        out[band] = composite(cube, method)
    return out


def _clear_sky_cube(items: Sequence[Item], collection: Collection, grid: Grid) -> np.ndarray | None:
    """Boolean ``(n, h, w)`` cube: True where a pixel is usable."""
    try:
        qa = read_stack(items, collection.qa_asset, grid)  # type: ignore[arg-type]
    except NoDataError:
        logger.warning("no QA band for %s — proceeding unmasked", collection.stac_id)
        return None

    if collection.stac_id.startswith("sentinel-2"):
        return _mask_from_scl(qa)
    if collection.stac_id.startswith("landsat"):
        return _mask_from_qa_pixel(qa)
    return None


def _mask_from_scl(scl: np.ndarray) -> np.ndarray:
    """Sentinel-2 scene classification layer -> clear-sky boolean.

    Rejected: 0 no-data, 1 saturated, 3 cloud shadow, 8/9 cloud medium+high
    probability, 10 thin cirrus. Class 11 (snow) is kept — in the Atlas it is
    real signal, not an artefact.
    """
    bad = np.isin(scl, (0, 1, 3, 8, 9, 10))
    return ~bad & ~np.isnan(scl)


def _mask_from_qa_pixel(qa: np.ndarray) -> np.ndarray:
    """Landsat Collection-2 QA_PIXEL bitmask -> clear-sky boolean.

    Bit 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow.
    """
    qa_int = np.nan_to_num(qa, nan=1.0).astype(np.uint16)
    bad_bits = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
    return (qa_int & bad_bits) == 0


def _is_categorical(asset_key: str) -> bool:
    key = asset_key.lower()
    return any(token in key for token in ("scl", "qa", "mask", "class", "data", "map", "burn"))


def valid_fraction(array: np.ndarray) -> float:
    """Share of pixels carrying data — the honesty metric for every result."""
    if array.size == 0:
        return 0.0
    return float(np.count_nonzero(~np.isnan(array)) / array.size)
