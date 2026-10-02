"""STAC catalog access.

Wraps the Microsoft Planetary Computer STAC API, which serves every collection
this platform uses under an open licence and without credentials. A free
subscription key only raises the anonymous rate limit.

Collection ids and asset keys below were verified against the live API — do not
guess them, they differ per collection (Sentinel-2 uses ``B04``, Landsat uses
``red``).
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Literal

import planetary_computer
import pystac_client
from pystac import Item

from app.core.config import settings
from app.core.geo import BBox

logger = logging.getLogger(__name__)

SensorId = Literal[
    "sentinel-2",
    "landsat",
    "sentinel-1",
    "modis-vi",
    "modis-burn",
    "modis-fire",
    "dem",
    "landcover",
    "worldcover",
]


@dataclass(frozen=True)
class Collection:
    """A remote-sensing product and the vocabulary needed to use it."""

    id: SensorId
    stac_id: str
    title: str
    #: Logical band name -> STAC asset key.
    bands: dict[str, str]
    #: Native ground sample distance in metres.
    resolution_m: float
    #: Multiply raw DN by this, then add offset, to reach physical units.
    scale: float = 1.0
    offset: float = 0.0
    #: Asset carrying the per-pixel quality mask, if any.
    qa_asset: str | None = None
    #: Nominal revisit in days — used to explain data gaps to the user.
    revisit_days: float | None = None
    #: Extra STAC query applied to every search.
    query: dict = field(default_factory=dict)
    #: Whether items carry an ``eo:cloud_cover`` property we can sort on.
    has_cloud_cover: bool = False
    licence: str = "open"


COLLECTIONS: dict[SensorId, Collection] = {
    "sentinel-2": Collection(
        id="sentinel-2",
        stac_id="sentinel-2-l2a",
        title="Sentinel-2 L2A surface reflectance",
        bands={
            "coastal": "B01",
            "blue": "B02",
            "green": "B03",
            "red": "B04",
            "rededge1": "B05",
            "rededge2": "B06",
            "rededge3": "B07",
            "nir": "B08",
            "nir08": "B8A",
            "swir16": "B11",
            "swir22": "B12",
            "visual": "visual",
        },
        resolution_m=10,
        # L2A products are stored as uint16 scaled by 10000.
        scale=1e-4,
        qa_asset="SCL",
        revisit_days=5,
        has_cloud_cover=True,
    ),
    "landsat": Collection(
        id="landsat",
        stac_id="landsat-c2-l2",
        title="Landsat Collection 2 Level-2 surface reflectance",
        bands={
            "coastal": "coastal",
            "blue": "blue",
            "green": "green",
            "red": "red",
            "nir": "nir08",
            "swir16": "swir16",
            "swir22": "swir22",
            "thermal": "lwir11",
        },
        resolution_m=30,
        # USGS Collection-2 surface reflectance scaling.
        scale=2.75e-5,
        offset=-0.2,
        qa_asset="qa_pixel",
        revisit_days=8,  # Landsat 8 + 9 combined
        has_cloud_cover=True,
        query={"platform": {"in": ["landsat-8", "landsat-9"]}},
    ),
    "sentinel-1": Collection(
        id="sentinel-1",
        stac_id="sentinel-1-rtc",
        title="Sentinel-1 radiometrically terrain-corrected backscatter",
        bands={"vv": "vv", "vh": "vh"},
        resolution_m=10,
        revisit_days=6,
    ),
    "modis-vi": Collection(
        id="modis-vi",
        stac_id="modis-13A1-061",
        title="MODIS 16-day vegetation indices (500 m)",
        bands={
            "ndvi": "500m_16_days_NDVI",
            "evi": "500m_16_days_EVI",
            "red": "500m_16_days_red_reflectance",
            "nir": "500m_16_days_NIR_reflectance",
        },
        resolution_m=500,
        scale=1e-4,
        revisit_days=16,
    ),
    "modis-burn": Collection(
        id="modis-burn",
        stac_id="modis-64A1-061",
        title="MODIS monthly burned area (500 m)",
        bands={"burn_date": "Burn_Date", "uncertainty": "Burn_Date_Uncertainty"},
        resolution_m=500,
        revisit_days=30,
    ),
    "modis-fire": Collection(
        id="modis-fire",
        stac_id="modis-14A1-061",
        title="MODIS daily thermal anomalies / active fire (1 km)",
        bands={"fire_mask": "FireMask", "frp": "MaxFRP"},
        resolution_m=1000,
        revisit_days=1,
    ),
    "dem": Collection(
        id="dem",
        stac_id="cop-dem-glo-30",
        title="Copernicus DEM GLO-30 (30 m)",
        bands={"elevation": "data"},
        resolution_m=30,
    ),
    "landcover": Collection(
        id="landcover",
        stac_id="io-lulc-annual-v02",
        title="Esri 10 m annual land cover",
        bands={"class": "data"},
        resolution_m=10,
    ),
    "worldcover": Collection(
        id="worldcover",
        stac_id="esa-worldcover",
        title="ESA WorldCover 10 m land cover",
        bands={"class": "map"},
        resolution_m=10,
    ),
}


class NoDataError(RuntimeError):
    """No imagery satisfies the request.

    Carries a human-readable hint because "no data" is nearly always a
    fixable query problem (too narrow a window, too strict a cloud filter).
    """

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


@lru_cache(maxsize=1)
def _client() -> pystac_client.Client:
    """STAC client that deliberately does **not** sign at search time.

    The obvious setup — ``modifier=planetary_computer.sign_inplace`` — requests
    a SAS token for every asset of every item returned, which trips the
    anonymous rate limit on any realistic query and returns HTTP 429. It is
    also wrong: SAS tokens expire in about an hour, so a search result cached
    for later use would carry dead URLs.

    Assets are instead signed just before they are read (see
    ``app.eo.raster.sign_href``), which needs one cached token per storage
    container regardless of how many scenes are involved.
    """
    if settings.pc_subscription_key:
        planetary_computer.set_subscription_key(settings.pc_subscription_key)
    return pystac_client.Client.open(settings.stac_url)


def search(
    sensor: SensorId,
    bbox: BBox,
    start: dt.date,
    end: dt.date,
    *,
    max_cloud: float | None = None,
    limit: int = 40,
) -> list[Item]:
    """Find items covering ``bbox`` between ``start`` and ``end``.

    Returns items sorted best-first: least cloudy for optical sensors, most
    recent otherwise. Signing happens automatically, so asset hrefs come back
    ready to read.
    """
    collection = COLLECTIONS[sensor]
    query = dict(collection.query)
    if max_cloud is not None and collection.has_cloud_cover:
        query["eo:cloud_cover"] = {"lt": max_cloud}

    search_result = _client().search(
        collections=[collection.stac_id],
        bbox=bbox.as_tuple(),
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        query=query or None,
        limit=min(limit, 100),
        max_items=limit,
    )
    items = list(search_result.items())

    if not items:
        raise NoDataError(
            f"No {collection.title} scenes cover this area between "
            f"{start.isoformat()} and {end.isoformat()}.",
            hint=_gap_hint(collection, max_cloud),
        )

    if collection.has_cloud_cover:
        items.sort(key=lambda it: (it.properties.get("eo:cloud_cover", 100.0), _dt_key(it)))
    else:
        items.sort(key=_dt_key, reverse=True)
    return items


def _gap_hint(collection: Collection, max_cloud: float | None) -> str:
    parts = []
    if collection.revisit_days:
        parts.append(
            f"{collection.title.split()[0]} revisits every ~{collection.revisit_days:g} days, "
            "so a window shorter than that can legitimately be empty"
        )
    if max_cloud is not None and collection.has_cloud_cover:
        parts.append(f"the cloud filter (<{max_cloud:g}%) may be excluding every available scene")
    parts.append("try widening the date range")
    return "; ".join(parts) + "."


def _dt_key(item: Item) -> dt.datetime:
    when = item.datetime
    if when is None:
        raw = item.properties.get("start_datetime") or "1970-01-01T00:00:00Z"
        when = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return when


def item_date(item: Item) -> dt.date:
    return _dt_key(item).date()


def describe(items: Iterable[Item]) -> list[dict]:
    """Compact provenance record attached to every analysis response."""
    out = []
    for item in items:
        out.append(
            {
                "id": item.id,
                "collection": item.collection_id,
                "datetime": _dt_key(item).isoformat(),
                "cloud_cover": item.properties.get("eo:cloud_cover"),
                "platform": item.properties.get("platform"),
            }
        )
    return out
