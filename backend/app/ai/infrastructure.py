"""Road and infrastructure module.

The honest framing matters here. "Road detection" from satellite imagery is
usually a solved-looking problem that is actually two problems:

1. *Where are the roads?* — For anywhere OpenStreetMap covers, this is already
   answered by people who surveyed the ground. Running a segmentation model to
   rediscover mapped roads is worse than reading the map.
2. *Which roads are missing from the map?* — This is the real question, and it
   is genuinely hard: it needs a trained model to spot linear built features
   that OSM does not know about, in exactly the rural areas where OSM is
   thinnest.

This module answers (1) properly from OSM, computes network statistics, and
provides a spectral candidate layer for (2) while being explicit that a
production answer needs the trained model declared in ``neural_backend``.
"""

from __future__ import annotations

import logging

import numpy as np

from app.ai import osm, sensing
from app.ai.base import (
    AnalysisRequest,
    AnalysisResult,
    ClassArea,
    Layer,
    confidence_from_coverage,
    percent,
)
from app.ai.urban import builtup_probability
from app.core.geo import BBox
from app.eo.catalog import NoDataError
from app.eo.raster import Grid

logger = logging.getLogger(__name__)

ROAD_COLOURS = {
    "motorway": "#e31a1c", "trunk": "#fd8d3c", "primary": "#fecc5c",
    "secondary": "#ffffb2", "tertiary": "#d9d9d9", "unclassified": "#bdbdbd",
    "residential": "#969696", "service": "#737373", "track": "#8c6d31",
}


class RoadsModule:
    id = "roads"
    title = "Road Network"
    question = "What roads are here, and is anything missing from the map?"
    description = (
        "Builds the road network from OpenStreetMap with per-class length statistics and "
        "density, and highlights linear built-up features that may be unmapped tracks."
    )
    sensors = ("sentinel-2",)
    needs_baseline = False
    neural_backend = "D-LinkNet / U-Net road extraction (SpaceNet-trained)"

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        grid = Grid.from_bbox(request.bbox, request.resolution_m or 10)
        caveats: list[str] = []

        try:
            ways = osm.roads(request.bbox)
        except osm.OverpassError as exc:
            raise NoDataError(
                f"OpenStreetMap is unreachable: {exc}",
                hint="Overpass rate-limits heavily. Retry in a minute, or narrow the area.",
            ) from exc

        raster_mask, lengths = _rasterize_roads(ways, grid)
        total_km = sum(lengths.values())
        density = total_km / max(request.bbox.area_km2, 1e-6)

        breakdown = [
            ClassArea(
                id=index, label=road_class, color=ROAD_COLOURS.get(road_class, "#999999"),
                pixels=0, area_km2=round(km, 2),
                share=km / total_km if total_km else 0.0,
            )
            for index, (road_class, km) in enumerate(
                sorted(lengths.items(), key=lambda kv: kv[1], reverse=True)
            )
        ]

        layers = [
            Layer("roads", "OSM road network", "mask", raster_mask.astype(np.float32),
                  colormap="urban", opacity=0.9,
                  description="Roads as mapped in OpenStreetMap, rasterised to the grid."),
        ]

        # Candidate unmapped tracks: built-looking, unvegetated ground away from
        # any known road. Deliberately labelled a candidate layer, not a result.
        try:
            scene = sensing.load_scene(
                request, ["blue", "green", "red", "nir", "swir16"], sensor="sentinel-2", grid=grid
            )
            built = builtup_probability(scene)
            from app.ai.terrain import distance_to

            far_from_road = distance_to(raster_mask, max_steps=6) > 3
            candidates = (built > 0.6) & far_from_road & np.isfinite(built)
            layers.append(
                Layer("unmapped_candidates", "Possible unmapped tracks", "mask",
                      candidates.astype(np.float32), colormap="soil", opacity=0.8,
                      description="Built-looking surface with no OSM road nearby.")
            )
            candidate_km2 = float(np.count_nonzero(candidates)) * grid.pixel_area_km2
            sources = scene.sources()
            dates = scene.dates()
            confidence = confidence_from_coverage(built) * 0.6
        except (NoDataError, ValueError) as exc:
            caveats.append(f"Unmapped-track candidates unavailable ({exc}).")
            candidate_km2 = None
            sources, dates, confidence = [], [], 0.8

        metrics = {
            "total_road_km": round(total_km, 2),
            "road_density_km_per_km2": round(density, 3),
            "way_count": len(ways),
            "paved_km": round(
                sum(km for cls, km in lengths.items()
                    if cls in ("motorway", "trunk", "primary", "secondary")), 2
            ),
            "track_km": round(lengths.get("track", 0.0), 2),
            "unmapped_candidate_km2": round(candidate_km2, 3) if candidate_km2 is not None else None,
        }

        return AnalysisResult(
            module=self.id, title=self.title,
            summary=(
                f"{metrics['total_road_km']} km of road across {len(ways)} OSM ways "
                f"({metrics['road_density_km_per_km2']} km/km²), of which "
                f"{metrics['paved_km']} km is primary or better."
            ),
            grid=grid, metrics=metrics, breakdown=breakdown, layers=layers,
            sources=sources + [{"collection": "openstreetmap", "note": "via Overpass API, ODbL"}],
            caveats=caveats
            + [
                "The road network is read from OpenStreetMap, not detected from imagery. "
                "OSM coverage in rural Morocco is uneven — a missing road on this map most "
                "often means it has not been surveyed by a contributor.",
                "'Possible unmapped tracks' is a coarse spectral candidate layer, not a "
                "road detection. Bare ground, quarries and dry wadis all trigger it. A "
                "usable answer needs the trained road-extraction model.",
                "Lengths are computed on the ellipsoid from OSM geometry; bridges and "
                "tunnels are counted at their plan length.",
            ],
            confidence=confidence, observed_dates=dates,
        )


def _rasterize_roads(ways: list[dict], grid: Grid) -> tuple[np.ndarray, dict[str, float]]:
    """Burn road centrelines into the grid and total their length per class."""
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp_transform

    mask = np.zeros(grid.shape, dtype=bool)
    lengths: dict[str, float] = {}
    wgs84 = CRS.from_epsg(4326)

    for way in ways:
        coords = way["coords"]
        lons = [point[0] for point in coords]
        lats = [point[1] for point in coords]
        lengths[way["class"]] = lengths.get(way["class"], 0.0) + _haversine_length(lons, lats)

        xs, ys = warp_transform(wgs84, grid.crs, lons, lats)
        pixels = [(~grid.transform) * (x, y) for x, y in zip(xs, ys)]
        for (c0, r0), (c1, r1) in zip(pixels, pixels[1:]):
            _draw_line(mask, int(r0), int(c0), int(r1), int(c1))

    return mask, lengths


def _haversine_length(lons: list[float], lats: list[float]) -> float:
    """Polyline length in km on a spherical Earth."""
    radius_km = 6371.0088
    phi = np.radians(lats)
    lam = np.radians(lons)
    dphi = np.diff(phi)
    dlam = np.diff(lam)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi[:-1]) * np.cos(phi[1:]) * np.sin(dlam / 2) ** 2
    return float(np.sum(2 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0, 1)))))


def _draw_line(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    """Bresenham line, clipped to the array."""
    height, width = mask.shape
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    # A segment far outside the grid contributes nothing but a long loop.
    if max(dr, dc) > 4 * max(height, width):
        return
    step_r = 1 if r0 < r1 else -1
    step_c = 1 if c0 < c1 else -1
    error = dc - dr

    while True:
        if 0 <= r0 < height and 0 <= c0 < width:
            mask[r0, c0] = True
        if r0 == r1 and c0 == c1:
            break
        doubled = 2 * error
        if doubled > -dr:
            error -= dr
            c0 += step_c
        if doubled < dc:
            error += dc
            r0 += step_r
