"""OpenStreetMap access via the Overpass API.

OSM supplies the reference layers the platform compares imagery against —
roads, buildings, waterways, hospitals, schools, administrative boundaries.
It is a *reference*, not ground truth: coverage in rural Morocco is uneven,
and an absent feature usually means nobody has mapped it yet, not that it does
not exist. That asymmetry is exactly what makes OSM useful for change
detection, and dangerous for enforcement.
"""

from __future__ import annotations

import logging
import time

import httpx

from app.core.geo import BBox

logger = logging.getLogger(__name__)

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

#: Highway tags worth rendering, in descending order of importance.
ROAD_CLASSES = (
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "service", "track",
)


class OverpassError(RuntimeError):
    pass


def query(overpass_ql: str, *, timeout: float = 60.0) -> dict:
    """Run an Overpass QL query, failing over between mirrors.

    Overpass is a free shared service that rate-limits and occasionally drops
    requests; a second endpoint costs nothing and removes most transient
    failures.
    """
    last_error: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            response = httpx.post(
                endpoint,
                data={"data": overpass_ql},
                timeout=timeout,
                headers={"User-Agent": "AtlasAI-Morocco/0.1 (Earth observation platform)"},
            )
            if response.status_code == 429:
                logger.warning("overpass %s rate-limited, backing off", endpoint)
                time.sleep(2)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - httpx raises a family of errors
            logger.warning("overpass %s failed: %s", endpoint, exc)
            last_error = exc

    raise OverpassError(f"All Overpass endpoints failed: {last_error}")


def _bbox_clause(bbox: BBox) -> str:
    """Overpass wants south,west,north,east — the opposite of GeoJSON order."""
    return f"{bbox.south},{bbox.west},{bbox.north},{bbox.east}"


def roads(bbox: BBox, *, classes: tuple[str, ...] = ROAD_CLASSES) -> list[dict]:
    """Road centrelines as ``{class, coords: [(lon, lat), ...]}``."""
    filter_ = "|".join(classes)
    data = query(
        f"""
        [out:json][timeout:60];
        way["highway"~"^({filter_})$"]({_bbox_clause(bbox)});
        out geom;
        """
    )
    out = []
    for element in data.get("elements", []):
        geometry = element.get("geometry") or []
        if len(geometry) < 2:
            continue
        out.append(
            {
                "class": element.get("tags", {}).get("highway", "unknown"),
                "name": element.get("tags", {}).get("name"),
                "coords": [(point["lon"], point["lat"]) for point in geometry],
            }
        )
    return out


def buildings(bbox: BBox, *, limit: int = 20000) -> list[dict]:
    """Building footprints as closed rings of lon/lat."""
    data = query(
        f"""
        [out:json][timeout:90];
        way["building"]({_bbox_clause(bbox)});
        out geom {limit};
        """
    )
    out = []
    for element in data.get("elements", []):
        geometry = element.get("geometry") or []
        if len(geometry) < 3:
            continue
        out.append(
            {
                "coords": [(point["lon"], point["lat"]) for point in geometry],
                "tags": element.get("tags", {}),
            }
        )
    return out


def amenities(bbox: BBox, kinds: tuple[str, ...] = ("hospital", "school", "clinic")) -> list[dict]:
    """Point features for the context panel — services near a clicked location."""
    filter_ = "|".join(kinds)
    data = query(
        f"""
        [out:json][timeout:60];
        (
          node["amenity"~"^({filter_})$"]({_bbox_clause(bbox)});
          way["amenity"~"^({filter_})$"]({_bbox_clause(bbox)});
        );
        out center;
        """
    )
    out = []
    for element in data.get("elements", []):
        lon = element.get("lon") or (element.get("center") or {}).get("lon")
        lat = element.get("lat") or (element.get("center") or {}).get("lat")
        if lon is None or lat is None:
            continue
        tags = element.get("tags", {})
        out.append(
            {
                "kind": tags.get("amenity"),
                "name": tags.get("name"),
                "lon": lon,
                "lat": lat,
            }
        )
    return out
