"""Terrain, land cover and renewable-energy siting modules.

Elevation is the quiet dependency of half the platform: floods run downhill,
solar panels care about slope and aspect, wind farms want ridges, and "mountain"
is a terrain fact rather than a land-cover class. This module owns the DEM and
the derivatives everything else borrows.

The land-cover module is deliberately *not* a from-scratch classifier. ESA
WorldCover already provides a validated 10 m global product with ~75% accuracy;
training a worse one from the same Sentinel-2 pixels would be a step backwards.
The genuine work here is reconciling that product with the classes this
platform reports — separating desert from bare soil, and identifying mountain
and beach, which are terrain and position rather than surface type.
"""

from __future__ import annotations

import logging

import numpy as np

from app.ai import sensing
from app.ai.base import (
    AnalysisRequest,
    AnalysisResult,
    Layer,
    class_areas,
    confidence_from_coverage,
    coverage_caveats,
    percent,
    safe_mean,
)
from app.eo.catalog import NoDataError
from app.eo.raster import Grid

logger = logging.getLogger(__name__)

#: The platform's reported land-cover classes.
LANDCOVER_PALETTE: dict[int, tuple[str, str]] = {
    0: ("Water", "#08306b"),
    1: ("Forest", "#00441b"),
    2: ("Agriculture", "#a6d96a"),
    3: ("Rangeland / shrub", "#d9c77d"),
    4: ("Urban", "#7f2704"),
    5: ("Bare soil", "#bf9b6e"),
    6: ("Desert", "#f2e2b6"),
    7: ("Mountain", "#7a7a7a"),
    8: ("Beach / dune", "#ffe9a8"),
    9: ("Wetland", "#4eb3d3"),
}

#: ESA WorldCover code -> platform class, for the codes that map directly.
WORLDCOVER_DIRECT = {
    10: 1,   # tree cover      -> forest
    40: 2,   # cropland        -> agriculture
    50: 4,   # built-up        -> urban
    80: 0,   # permanent water -> water
    90: 9,   # herbaceous wetland
    95: 9,   # mangrove
    20: 3,   # shrubland       -> rangeland
    30: 3,   # grassland       -> rangeland
    100: 3,  # moss & lichen   -> rangeland
    70: 7,   # snow & ice      -> mountain (in Morocco this is the High Atlas)
}


def load_dem(request: AnalysisRequest, grid: Grid) -> np.ndarray:
    """Copernicus GLO-30 elevation resampled onto ``grid``, in metres."""
    dem_request = AnalysisRequest(
        bbox=request.bbox,
        # The DEM is a static mosaic; its items are stamped 2021 and a narrow
        # window would simply miss them.
        start=request.start.replace(year=2010, month=1, day=1),
        end=request.end.replace(year=2030, month=12, day=31),
        resolution_m=request.resolution_m,
        max_items=12,
    )
    scene = sensing.load_scene(
        dem_request, ["elevation"], sensor="dem", grid=grid,
        method="first", mask_clouds=False, max_items=12,
    )
    return scene.band("elevation")


def slope_degrees(elevation: np.ndarray, resolution_m: float) -> np.ndarray:
    """Terrain slope via the Horn (1981) 3x3 kernel, in degrees."""
    padded = np.pad(elevation, 1, mode="edge")
    # Horn's method weights the cardinal neighbours double — less noise-prone
    # than a simple central difference on a real DEM.
    dz_dx = (
        (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / (8 * resolution_m)
    dz_dy = (
        (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])
    ) / (8 * resolution_m)
    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)


def aspect_degrees(elevation: np.ndarray, resolution_m: float) -> np.ndarray:
    """Downslope compass direction, 0 = north, clockwise."""
    padded = np.pad(elevation, 1, mode="edge")
    dz_dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2 * resolution_m)
    dz_dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2 * resolution_m)
    aspect = np.degrees(np.arctan2(-dz_dx, dz_dy))
    return np.mod(450.0 - aspect, 360.0).astype(np.float32)


def relative_elevation(elevation: np.ndarray, window: int = 25) -> np.ndarray:
    """Height above the local minimum — a cheap stand-in for HAND.

    True Height Above Nearest Drainage needs flow routing over a hydrologically
    conditioned DEM. This approximation asks a simpler question — "how far above
    the lowest ground nearby is this pixel?" — which captures the same
    first-order flood signal: valley floors score near zero, hillsides do not.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    radius = max(window // 2, 1)
    padded = np.pad(np.nan_to_num(elevation, nan=np.nanmax(elevation)), radius, mode="edge")
    windows = sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1))
    # Stride the window grid so a large AOI does not build a huge intermediate.
    local_min = windows.min(axis=(-1, -2))
    return (elevation - local_min).astype(np.float32)


def distance_to(mask: np.ndarray, max_steps: int) -> np.ndarray:
    """Approximate pixel distance to the nearest True, capped at ``max_steps``.

    Iterative dilation rather than a full distance transform: we only ever ask
    "is this within N pixels of water", so the cap keeps the cost bounded.
    """
    distance = np.full(mask.shape, np.inf, dtype=np.float32)
    distance[mask] = 0
    frontier = mask.copy()
    for step in range(1, max_steps + 1):
        grown = frontier.copy()
        grown[1:, :] |= frontier[:-1, :]
        grown[:-1, :] |= frontier[1:, :]
        grown[:, 1:] |= frontier[:, :-1]
        grown[:, :-1] |= frontier[:, 1:]
        newly = grown & ~frontier
        if not newly.any():
            break
        distance[newly] = step
        frontier = grown
    return distance


class TerrainModule:
    id = "terrain"
    title = "Terrain Analysis"
    question = "What does the ground look like here — elevation, slope, aspect?"
    description = (
        "Copernicus GLO-30 elevation with derived slope, aspect and height above local "
        "drainage. Feeds the flood, solar and wind modules."
    )
    sensors = ("dem",)
    needs_baseline = False
    neural_backend = None

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        grid = Grid.from_bbox(request.bbox, request.resolution_m or 30)
        elevation = load_dem(request, grid)
        slope = slope_degrees(elevation, grid.resolution_m)
        aspect = aspect_degrees(elevation, grid.resolution_m)
        relative = relative_elevation(elevation)

        finite = elevation[np.isfinite(elevation)]
        metrics = {
            "mean_elevation_m": safe_mean(elevation),
            "min_elevation_m": round(float(finite.min()), 1) if finite.size else None,
            "max_elevation_m": round(float(finite.max()), 1) if finite.size else None,
            "relief_m": round(float(finite.max() - finite.min()), 1) if finite.size else None,
            "mean_slope_deg": safe_mean(slope),
            "steep_share_pct": percent(
                np.count_nonzero(slope > 15) / max(np.count_nonzero(np.isfinite(slope)), 1)
            ),
        }

        return AnalysisResult(
            module=self.id, title=self.title,
            summary=(
                f"Elevation ranges {metrics['min_elevation_m']}-{metrics['max_elevation_m']} m "
                f"(relief {metrics['relief_m']} m); mean slope {metrics['mean_slope_deg']}°, "
                f"with {metrics['steep_share_pct']}% of the area steeper than 15°."
            ),
            grid=grid, metrics=metrics,
            layers=[
                Layer("elevation", "Elevation", "continuous", elevation, "elevation",
                      float(finite.min()) if finite.size else 0,
                      float(finite.max()) if finite.size else 1000, unit="m"),
                Layer("slope", "Slope", "continuous", slope, "risk", 0, 45, unit="degrees"),
                Layer("aspect", "Aspect", "continuous", aspect, "gray", 0, 360, unit="degrees"),
                Layer("hand", "Height above local drainage", "continuous", relative, "water", 0, 60,
                      unit="m", description="Low values mark valley floors and flood-prone ground."),
            ],
            sources=[{"collection": "cop-dem-glo-30", "note": "Copernicus DEM GLO-30, static mosaic"}],
            caveats=[
                "GLO-30 is a surface model: it includes tree canopy and buildings, so "
                "elevations in forest and city are a few metres above true ground.",
                "Height above local drainage is a windowed approximation of HAND, not a "
                "hydrologically routed computation.",
            ],
            confidence=confidence_from_coverage(elevation),
        )


class LandCoverModule:
    id = "landcover"
    title = "Land Cover Classification"
    question = "What is on the ground — and how much of each type?"
    description = (
        "Reconciles ESA WorldCover's validated 10 m classes with terrain and spectral "
        "evidence to report water, forest, agriculture, rangeland, urban, bare soil, "
        "desert, mountain, beach and wetland, with area for each."
    )
    sensors = ("worldcover", "sentinel-2", "dem")
    needs_baseline = False
    neural_backend = "SegFormer / DeepLabV3+ fine-tuned on WorldCover labels"

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        grid = Grid.from_bbox(request.bbox, request.resolution_m or 20)

        lc_request = AnalysisRequest(
            bbox=request.bbox,
            start=request.start.replace(year=2020, month=1, day=1),
            end=request.end.replace(year=2022, month=12, day=31),
            resolution_m=request.resolution_m,
        )
        base = sensing.load_scene(
            lc_request, ["class"], sensor="worldcover", grid=grid,
            method="first", mask_clouds=False, max_items=6,
        )
        codes = base.band("class")

        classes = np.full(codes.shape, -1, dtype=np.int16)
        for wc_code, target in WORLDCOVER_DIRECT.items():
            classes[codes == wc_code] = target

        caveats = [
            "Base classes come from ESA WorldCover 2021 (~75% global accuracy), not from a "
            "model trained here. Its class boundaries are the ones being reported.",
            "'Mountain' and 'Beach' are terrain and position, not surface types — they are "
            "derived from elevation, slope and distance to water, and are applied only to "
            "sparsely vegetated ground so they do not overwrite forest or farmland.",
        ]

        # WorldCover's single "bare / sparse vegetation" class covers Saharan
        # erg, Atlas scree and coastal dune alike. Splitting it needs greenness
        # and terrain.
        bare = codes == 60
        ndvi: np.ndarray | None = None
        try:
            optical = sensing.load_scene(
                request, ["blue", "red", "nir", "swir16"], sensor="sentinel-2", grid=grid
            )
            ndvi = optical.index("ndvi")
            sources = base.sources() + optical.sources()
            dates = optical.dates()
        except (NoDataError, ValueError) as exc:
            logger.info("landcover: optical refinement unavailable (%s)", exc)
            caveats.append(f"Desert/bare-soil split unavailable ({exc}); all bare ground reported as bare soil.")
            sources, dates = base.sources(), []

        if ndvi is not None:
            # Ground that stays this dark green is not merely bare, it is desert.
            classes[bare & (ndvi < 0.10)] = 6
            classes[bare & (ndvi >= 0.10)] = 5
        else:
            classes[bare] = 5

        try:
            elevation = load_dem(request, grid)
            slope = slope_degrees(elevation, grid.resolution_m)
            sparse = np.isin(classes, (3, 5, 6, 7))
            mountain = sparse & (elevation > 1200) & (slope > 12)
            classes[mountain] = 7

            water_mask = classes == 0
            if water_mask.any():
                # ~300 m of the shoreline, below 25 m elevation, unvegetated.
                reach = max(int(300 / grid.resolution_m), 1)
                near_water = distance_to(water_mask, reach) <= reach
                beach = near_water & np.isin(classes, (5, 6)) & (elevation < 25)
                classes[beach] = 8
            sources.append({"collection": "cop-dem-glo-30", "note": "terrain refinement"})
        except (NoDataError, ValueError) as exc:
            logger.info("landcover: terrain refinement unavailable (%s)", exc)
            caveats.append(f"Mountain/beach refinement unavailable ({exc}).")

        breakdown = class_areas(classes, LANDCOVER_PALETTE, grid)
        dominant = max(breakdown, key=lambda c: c.pixels)
        top = sorted(breakdown, key=lambda c: c.pixels, reverse=True)[:3]

        return AnalysisResult(
            module=self.id, title=self.title,
            summary=(
                f"Dominant cover is {dominant.label.lower()} at {dominant.share:.0%} "
                f"({dominant.area_km2:.1f} km²). Top three: "
                + ", ".join(f"{c.label} {c.share:.0%}" for c in top)
                + "."
            ),
            grid=grid,
            metrics={
                "dominant_class": dominant.label,
                "dominant_share_pct": percent(dominant.share),
                "classified_km2": round(
                    sum(c.area_km2 for c in breakdown), 2
                ),
                "class_count": sum(1 for c in breakdown if c.pixels > 0),
            },
            breakdown=breakdown,
            layers=[
                Layer("landcover", "Land cover", "categorical", classes.astype(np.float32),
                      palette=LANDCOVER_PALETTE, opacity=0.8)
            ],
            sources=sources, caveats=caveats,
            confidence=confidence_from_coverage(codes) * 0.9,
            observed_dates=dates,
        )


class SolarPotentialModule:
    id = "solar"
    title = "Solar Potential"
    question = "Where should solar capacity be built here?"
    description = (
        "Scores land for utility-scale solar from slope, aspect, observed cloud "
        "frequency and land-cover availability."
    )
    sensors = ("dem", "worldcover", "sentinel-2")
    needs_baseline = False
    neural_backend = None

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        grid = Grid.from_bbox(request.bbox, request.resolution_m or 30)
        elevation = load_dem(request, grid)
        slope = slope_degrees(elevation, grid.resolution_m)
        aspect = aspect_degrees(elevation, grid.resolution_m)

        # Flat is best; up to ~10° is workable with standard racking.
        slope_score = np.clip(1.0 - slope / 15.0, 0, 1)
        # In the northern hemisphere a south-facing slope gains, north loses.
        aspect_score = np.where(
            slope < 2, 1.0, 0.5 + 0.5 * np.cos(np.radians(aspect - 180.0))
        )

        caveats: list[str] = []
        availability = np.ones(grid.shape, dtype=np.float32)
        try:
            lc = LandCoverModule().run(
                AnalysisRequest(bbox=request.bbox, start=request.start, end=request.end,
                                resolution_m=grid.resolution_m)
            )
            lc_layer = lc.layer("landcover")
            if lc_layer is not None:
                cover = lc_layer.array
                # Exclude water, wetland, forest and existing urban outright;
                # discount farmland, which has a competing use.
                availability = np.where(np.isin(cover, (0, 1, 9, 4)), 0.0, 1.0)
                availability = np.where(cover == 2, 0.35, availability)
        except (NoDataError, ValueError) as exc:
            caveats.append(f"Land availability not applied ({exc}); scores reflect terrain only.")

        score = (0.45 * slope_score + 0.25 * aspect_score + 0.30) * availability
        score = np.clip(score, 0, 1) * 100
        score[~np.isfinite(elevation)] = np.nan

        excellent = np.count_nonzero(score >= 80)
        good = np.count_nonzero((score >= 65) & (score < 80))
        px = grid.pixel_area_km2

        # ~55 MW per km² for fixed-tilt utility-scale PV in this latitude band.
        capacity_mw = (excellent + good) * px * 55

        return AnalysisResult(
            module=self.id, title=self.title,
            summary=(
                f"{excellent * px:.1f} km² rates excellent and {good * px:.1f} km² good for "
                f"utility-scale solar — roughly {capacity_mw:,.0f} MW of installable capacity "
                "on terrain and land-use grounds alone."
            ),
            grid=grid,
            metrics={
                "excellent_km2": round(excellent * px, 2),
                "good_km2": round(good * px, 2),
                "mean_score": safe_mean(score),
                "indicative_capacity_mw": round(capacity_mw, 0),
                "mean_slope_deg": safe_mean(slope),
            },
            layers=[
                Layer("suitability", "Solar suitability", "continuous", score, "burn", 0, 100,
                      unit="score"),
                Layer("slope", "Slope", "continuous", slope, "risk", 0, 30, unit="degrees"),
            ],
            sources=[{"collection": "cop-dem-glo-30"}, {"collection": "esa-worldcover"}],
            caveats=caveats
            + [
                "Terrain and land-cover screening only. It does not consider grid "
                "connection distance, land ownership, protected areas, dust soiling rates "
                "or water availability for cleaning — all of which decide real projects.",
                "Capacity is a density estimate (~55 MW/km²), not a yield simulation. Use "
                "PVGIS or SAM with local irradiance data for energy figures.",
            ],
            confidence=confidence_from_coverage(elevation) * 0.7,
        )


class WindPotentialModule:
    id = "wind"
    title = "Wind Farm Suitability"
    question = "Which terrain here suits wind development?"
    description = (
        "Screens for wind-farm siting using terrain exposure, ridge position, land cover "
        "and distance from settlement. Terrain proxy only — no wind resource data."
    )
    sensors = ("dem", "worldcover")
    needs_baseline = False
    neural_backend = None

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        grid = Grid.from_bbox(request.bbox, request.resolution_m or 30)
        elevation = load_dem(request, grid)
        slope = slope_degrees(elevation, grid.resolution_m)

        # Exposure: how far a pixel stands above its surroundings. Ridges and
        # plateau edges score high; sheltered valleys score low.
        exposure = relative_elevation(elevation, window=41)
        exposure_score = np.clip(exposure / 120.0, 0, 1)
        # Turbines need buildable ground: steep faces are expensive to access.
        buildable = np.clip(1.0 - np.abs(slope - 6.0) / 20.0, 0, 1)

        caveats: list[str] = []
        availability = np.ones(grid.shape, dtype=np.float32)
        try:
            lc = LandCoverModule().run(
                AnalysisRequest(bbox=request.bbox, start=request.start, end=request.end,
                                resolution_m=grid.resolution_m)
            )
            lc_layer = lc.layer("landcover")
            if lc_layer is not None:
                cover = lc_layer.array
                availability = np.where(np.isin(cover, (0, 4, 9)), 0.0, 1.0)
                availability = np.where(cover == 1, 0.4, availability)  # forest: possible, costly
        except (NoDataError, ValueError) as exc:
            caveats.append(f"Land availability not applied ({exc}).")

        score = np.clip(0.55 * exposure_score + 0.45 * buildable, 0, 1) * availability * 100
        score[~np.isfinite(elevation)] = np.nan
        px = grid.pixel_area_km2
        strong = np.count_nonzero(score >= 70)

        return AnalysisResult(
            module=self.id, title=self.title,
            summary=(
                f"{strong * px:.1f} km² of terrain scores strongly for wind siting "
                f"(exposed, buildable and unoccupied)."
            ),
            grid=grid,
            metrics={
                "strong_siting_km2": round(strong * px, 2),
                "mean_score": safe_mean(score),
                "mean_exposure_m": safe_mean(exposure),
            },
            layers=[
                Layer("suitability", "Wind siting score", "continuous", score, "burn", 0, 100,
                      unit="score"),
                Layer("exposure", "Terrain exposure", "continuous", exposure, "elevation", 0, 150,
                      unit="m"),
            ],
            sources=[{"collection": "cop-dem-glo-30"}, {"collection": "esa-worldcover"}],
            caveats=caveats
            + [
                "THIS IS NOT A WIND RESOURCE ASSESSMENT. It contains no wind speed data. "
                "Terrain exposure correlates with wind but does not measure it — Morocco's "
                "best wind sites (Tarfaya, Essaouira, Tangier) are driven by regional "
                "atmospheric patterns this analysis cannot see.",
                "Use the Global Wind Atlas or, for any real project, on-site met-mast "
                "measurement over at least twelve months.",
            ],
            confidence=confidence_from_coverage(elevation) * 0.5,
        )
