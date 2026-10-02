"""Vegetation, drought and agriculture modules.

The three share one physical basis — canopy reflectance in the red, near-infrared
and shortwave-infrared — but ask different questions of it:

* **Vegetation** — how much green biomass is there *now*?
* **Drought** — how does that compare with the same season in normal years?
* **Agriculture** — how are the *cropped* pixels specifically performing?

The drought module is the one that matters most for Morocco, and it is
deliberately built on the Vegetation Condition Index rather than on raw NDVI.
Raw NDVI cannot distinguish "this land is dry" from "this land is a desert and
always has been". VCI rescales each pixel against its own multi-year seasonal
range, so it reports *departure from normal* — which is what drought means.
"""

from __future__ import annotations

import logging

import numpy as np

from app.ai import sensing
from app.ai.base import (
    AnalysisRequest,
    AnalysisResult,
    ClassArea,
    Layer,
    class_areas,
    classify,
    confidence_from_coverage,
    coverage_caveats,
    percent,
    safe_mean,
    trend_label,
)
from app.eo.catalog import NoDataError
from app.eo.raster import Grid

logger = logging.getLogger(__name__)

VEGETATION_PALETTE: dict[int, tuple[str, str]] = {
    0: ("Bare / non-vegetated", "#a6611a"),
    1: ("Sparse vegetation", "#dfc27d"),
    2: ("Moderate vegetation", "#a6d96a"),
    3: ("Dense vegetation", "#1a9641"),
    4: ("Very dense vegetation", "#00441b"),
}
#: NDVI breakpoints between the classes above.
VEGETATION_THRESHOLDS = (0.10, 0.20, 0.35, 0.55)

DROUGHT_PALETTE: dict[int, tuple[str, str]] = {
    0: ("Extreme drought", "#7f0000"),
    1: ("Severe drought", "#d7191c"),
    2: ("Moderate drought", "#fdae61"),
    3: ("Normal", "#1a9641"),
}
#: Vegetation Condition Index breakpoints (Kogan 1990, as used operationally
#: by FAO/ASIS and the USDA). VCI is a percentage.
VCI_THRESHOLDS = (10.0, 20.0, 40.0)


class VegetationModule:
    id = "vegetation"
    title = "Vegetation Monitoring"
    question = "How much healthy vegetation is there, and is it gaining or losing ground?"
    description = (
        "Maps green biomass and canopy moisture from Sentinel-2, scores overall "
        "vegetation health, and — given a baseline window — reports gain, loss and "
        "desertification pressure."
    )
    sensors = ("sentinel-2", "landsat")
    needs_baseline = False
    neural_backend = "SegFormer-B2 (land-cover fine-tuned)"

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        scene = sensing.load_with_fallback(request, ["blue", "red", "nir", "swir16"])

        ndvi = scene.index("ndvi")
        savi = scene.index("savi")
        evi = scene.index("evi")
        ndmi = scene.index("ndmi")

        health = vegetation_health(savi, ndmi)
        classes = classify(ndvi, VEGETATION_THRESHOLDS)
        breakdown = class_areas(classes, VEGETATION_PALETTE, scene.grid)

        vegetated = np.count_nonzero(classes >= 1)
        observed = max(np.count_nonzero(classes >= 0), 1)
        vegetated_share = vegetated / observed

        metrics: dict[str, object] = {
            "vegetation_health_score": safe_mean(health),
            "mean_ndvi": safe_mean(ndvi),
            "mean_savi": safe_mean(savi),
            "mean_evi": safe_mean(evi),
            "mean_ndmi": safe_mean(ndmi),
            "vegetated_share_pct": percent(vegetated_share),
            "vegetated_area_km2": round(vegetated * scene.grid.pixel_area_km2, 2),
            "aoi_area_km2": round(request.bbox.area_km2, 1),
        }

        layers = [
            Layer("ndvi", "NDVI", "continuous", ndvi, "vegetation", -0.2, 0.9,
                  description="Greenness. Above ~0.3 indicates active canopy."),
            Layer("health", "Vegetation health score", "continuous", health, "vegetation", 0, 100,
                  unit="score", description="Greenness and canopy moisture combined, 0-100."),
            Layer("ndmi", "Canopy moisture (NDMI)", "continuous", ndmi, "moisture", -0.6, 0.6,
                  description="Water content of the canopy — falls before greenness does."),
            Layer("classes", "Vegetation density", "categorical", classes.astype(np.float32),
                  palette=VEGETATION_PALETTE,
                  description="NDVI binned into density classes."),
        ]

        caveats = coverage_caveats(scene.coverage(), request)
        summary_parts = [
            f"Mean NDVI is {metrics['mean_ndvi']}, with {metrics['vegetated_share_pct']}% of the "
            f"observed area carrying vegetation ({metrics['vegetated_area_km2']} km²)."
        ]

        # A baseline turns a snapshot into a claim about change.
        if request.has_baseline():
            try:
                baseline = sensing.load_scene(
                    request,
                    ["red", "nir"],
                    sensor=scene.sensor,
                    start=request.baseline_start,  # type: ignore[arg-type]
                    end=request.baseline_end,  # type: ignore[arg-type]
                    grid=scene.grid,
                )
                base_ndvi = baseline.index("ndvi")
                delta = ndvi - base_ndvi
                loss = np.count_nonzero(delta < -0.10)
                gain = np.count_nonzero(delta > 0.10)
                metrics.update(
                    {
                        "baseline_mean_ndvi": safe_mean(base_ndvi),
                        "ndvi_change": safe_mean(delta),
                        "vegetation_loss_km2": round(loss * scene.grid.pixel_area_km2, 2),
                        "vegetation_gain_km2": round(gain * scene.grid.pixel_area_km2, 2),
                        "trend": trend_label(safe_mean(delta) or 0.0),
                    }
                )
                layers.append(
                    Layer("change", "NDVI change vs baseline", "continuous", delta, "change",
                          -0.4, 0.4, description="Blue = greening, red = vegetation loss.")
                )
                summary_parts.append(
                    f"Against the baseline window, NDVI is {metrics['trend']} "
                    f"({metrics['ndvi_change']:+}); {metrics['vegetation_loss_km2']} km² lost "
                    f"and {metrics['vegetation_gain_km2']} km² gained vegetation."
                )
                caveats.append(
                    "Change is measured between two cloud-free composites, not two single dates; "
                    "sub-window variation is averaged out."
                )
            except NoDataError as exc:
                caveats.append(f"Baseline comparison unavailable — {exc}")

        return AnalysisResult(
            module=self.id,
            title=self.title,
            summary=" ".join(summary_parts),
            grid=scene.grid,
            metrics=metrics,
            breakdown=breakdown,
            layers=layers,
            sources=scene.sources(),
            caveats=caveats,
            confidence=confidence_from_coverage(ndvi),
            observed_dates=scene.dates(),
        )


class DroughtModule:
    id = "drought"
    title = "Drought Detection"
    question = "Which areas are drier than they should be for this time of year?"
    description = (
        "Computes the Vegetation Condition Index against the same calendar window in "
        "previous years, so it reports departure from seasonal normal rather than "
        "absolute dryness. Deserts do not register as drought; failing farmland does."
    )
    sensors = ("sentinel-2", "landsat")
    needs_baseline = False
    neural_backend = "Temporal CNN + Transformer over index time series"

    #: How many prior years form the "normal" range. More years is a better
    #: climatology but costs one extra imagery search each.
    baseline_years = 3

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        scene = sensing.load_with_fallback(request, ["blue", "red", "nir", "swir16"])
        ndvi = scene.index("ndvi")
        ndmi = scene.index("ndmi")

        history = self._seasonal_history(request, scene.grid, scene.sensor)
        caveats: list[str] = coverage_caveats(scene.coverage(), request)

        if history is None:
            # Without a climatology, fall back to an absolute-dryness read and
            # say so plainly rather than pretending it is a drought index.
            vci = np.clip((ndvi + 0.1) / 0.6, 0, 1) * 100
            caveats.insert(
                0,
                "No usable imagery from previous years was found, so this is an absolute "
                "dryness index, not a true departure-from-normal drought index. Naturally "
                "arid terrain will read as dry here.",
            )
            method = "absolute-dryness-fallback"
            years_used = 0
        else:
            ndvi_min, ndvi_max, ndvi_mean, years_used = history
            span = ndvi_max - ndvi_min
            with np.errstate(divide="ignore", invalid="ignore"):
                vci = np.where(span < 0.05, np.nan, (ndvi - ndvi_min) / span * 100)
            # A pixel whose multi-year NDVI range is tiny (rock, water, tarmac)
            # has no meaningful "condition" — leave it unclassified.
            vci = np.clip(vci, 0, 100)
            method = f"VCI vs {years_used}-year seasonal baseline"
            caveats.append(
                f"Baseline built from {years_used} prior year(s) of the same calendar window. "
                "Operational drought products typically use 20+ years; a short baseline "
                "over-reacts to an unusual reference year."
            )

        classes = classify(vci, VCI_THRESHOLDS)
        breakdown = class_areas(classes, DROUGHT_PALETTE, scene.grid)
        in_drought = sum(c.pixels for c in breakdown if c.id <= 2)
        observed = max(int(np.count_nonzero(classes >= 0)), 1)

        metrics: dict[str, object] = {
            "method": method,
            "baseline_years": years_used,
            "mean_vci": safe_mean(vci),
            "drought_affected_share_pct": percent(in_drought / observed),
            "drought_affected_km2": round(in_drought * scene.grid.pixel_area_km2, 2),
            "severe_or_worse_km2": round(
                sum(c.pixels for c in breakdown if c.id <= 1) * scene.grid.pixel_area_km2, 2
            ),
            "mean_ndvi": safe_mean(ndvi),
            "mean_canopy_moisture": safe_mean(ndmi),
        }
        if history is not None:
            metrics["ndvi_anomaly_vs_normal"] = safe_mean(ndvi - history[2])

        worst = max(breakdown, key=lambda c: c.share if c.id <= 2 else -1)
        summary = (
            f"{metrics['drought_affected_share_pct']}% of the observed area shows drought "
            f"conditions ({metrics['drought_affected_km2']} km²), dominated by "
            f"'{worst.label}'. Mean VCI is {metrics['mean_vci']}."
        )

        return AnalysisResult(
            module=self.id,
            title=self.title,
            summary=summary,
            grid=scene.grid,
            metrics=metrics,
            breakdown=breakdown,
            layers=[
                Layer("severity", "Drought severity", "categorical", classes.astype(np.float32),
                      palette=DROUGHT_PALETTE,
                      description="VCI binned into drought categories."),
                Layer("vci", "Vegetation Condition Index", "continuous", vci, "drought", 0, 100,
                      unit="%", description="0 = worst on record for this season, 100 = best."),
                Layer("ndmi", "Canopy moisture", "continuous", ndmi, "moisture", -0.6, 0.6,
                      description="Leading indicator — moisture drops before greenness does."),
            ],
            sources=scene.sources(),
            caveats=caveats,
            confidence=confidence_from_coverage(vci) * (0.7 if history is None else 1.0),
            observed_dates=scene.dates(),
        )

    def _seasonal_history(
        self, request: AnalysisRequest, grid: Grid, sensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
        """Per-pixel NDVI min/max/mean over the same season in prior years."""
        stack: list[np.ndarray] = []
        for start, end in sensing.seasonal_baseline_windows(request.end, years=self.baseline_years):
            try:
                past = sensing.load_scene(
                    request, ["red", "nir"], sensor=sensor,
                    start=start, end=end, grid=grid, max_items=6,
                )
                stack.append(past.index("ndvi"))
            except (NoDataError, ValueError) as exc:
                logger.info("drought baseline %s..%s unavailable: %s", start, end, exc)

        if not stack:
            return None
        cube = np.stack(stack)
        with np.errstate(all="ignore"):
            return (
                np.nanmin(cube, axis=0),
                np.nanmax(cube, axis=0),
                np.nanmean(cube, axis=0),
                len(stack),
            )


class AgricultureModule:
    id = "agriculture"
    title = "Agricultural Monitoring"
    question = "How are the crops doing, and which fields are irrigated?"
    description = (
        "Restricts vegetation analysis to pixels that ESA WorldCover identifies as "
        "cropland, then scores crop vigour and flags likely irrigation — fields that "
        "stay green while the surrounding landscape dries out."
    )
    sensors = ("sentinel-2", "worldcover")
    needs_baseline = False
    neural_backend = "U-Net crop-type classifier over Sentinel-2 time series"

    #: ESA WorldCover code for cropland.
    CROPLAND_CODE = 40

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        scene = sensing.load_with_fallback(request, ["blue", "red", "nir", "swir16"])
        ndvi = scene.index("ndvi")
        ndmi = scene.index("ndmi")

        caveats = coverage_caveats(scene.coverage(), request)
        cropland = self._cropland_mask(request, scene.grid)
        if cropland is None:
            cropland = np.isfinite(ndvi)
            caveats.insert(0, "Cropland map unavailable — statistics cover the whole area, not just fields.")
            crop_source = "whole AOI (no cropland mask)"
        else:
            crop_source = "ESA WorldCover cropland class"

        crop_ndvi = np.where(cropland, ndvi, np.nan)
        crop_ndmi = np.where(cropland, ndmi, np.nan)
        crop_pixels = int(np.count_nonzero(cropland & np.isfinite(ndvi)))

        # Vigour classes tuned for cropland rather than for natural vegetation:
        # a field below 0.25 NDVI mid-season is failing, not merely sparse.
        vigour = classify(crop_ndvi, (0.15, 0.25, 0.40, 0.60))
        vigour_palette = {
            0: ("Failed / fallow", "#a6611a"),
            1: ("Stressed", "#dfc27d"),
            2: ("Fair", "#ffffbf"),
            3: ("Good", "#a6d96a"),
            4: ("Vigorous", "#1a9641"),
        }
        breakdown = class_areas(vigour, vigour_palette, scene.grid)

        # Irrigation proxy: high canopy moisture where the wider landscape is dry.
        landscape_ndmi = safe_mean(ndmi) or 0.0
        irrigated = cropland & np.isfinite(crop_ndmi) & (crop_ndmi > landscape_ndmi + 0.15) & (crop_ndvi > 0.3)
        irrigated_km2 = float(np.count_nonzero(irrigated)) * scene.grid.pixel_area_km2

        metrics = {
            "cropland_source": crop_source,
            "cropland_km2": round(crop_pixels * scene.grid.pixel_area_km2, 2),
            "cropland_share_pct": percent(crop_pixels / max(ndvi.size, 1)),
            "mean_crop_ndvi": safe_mean(crop_ndvi),
            "mean_crop_moisture": safe_mean(crop_ndmi),
            "stressed_share_pct": percent(
                sum(c.pixels for c in breakdown if c.id <= 1) / max(crop_pixels, 1)
            ),
            "likely_irrigated_km2": round(irrigated_km2, 2),
        }

        return AnalysisResult(
            module=self.id,
            title=self.title,
            summary=(
                f"{metrics['cropland_km2']} km² of cropland observed; mean crop NDVI "
                f"{metrics['mean_crop_ndvi']}, with {metrics['stressed_share_pct']}% of fields "
                f"stressed or failed and ~{metrics['likely_irrigated_km2']} km² showing an "
                "irrigation signature."
            ),
            grid=scene.grid,
            metrics=metrics,
            breakdown=breakdown,
            layers=[
                Layer("vigour", "Crop vigour", "categorical", vigour.astype(np.float32),
                      palette=vigour_palette, description="NDVI over cropland pixels only."),
                Layer("crop_ndvi", "Crop NDVI", "continuous", crop_ndvi, "vegetation", 0, 0.9),
                Layer("irrigation", "Likely irrigated", "mask", irrigated.astype(np.float32),
                      colormap="water", description="Canopy moisture well above the local landscape."),
            ],
            sources=scene.sources(),
            caveats=caveats
            + [
                "Irrigation is inferred from a moisture contrast, not observed directly; "
                "naturally wet valley bottoms produce the same signature.",
                "Crop *type* classification needs a trained temporal model — not included.",
            ],
            confidence=confidence_from_coverage(crop_ndvi),
            observed_dates=scene.dates(),
        )

    def _cropland_mask(self, request: AnalysisRequest, grid: Grid) -> np.ndarray | None:
        try:
            lc = sensing.load_scene(
                request, ["class"], sensor="worldcover",
                start=request.start.replace(year=2020, month=1, day=1),
                end=request.end.replace(year=2022, month=12, day=31),
                grid=grid, method="first", mask_clouds=False, max_items=4,
            )
            return lc.band("class") == self.CROPLAND_CODE
        except (NoDataError, ValueError) as exc:
            logger.info("worldcover cropland mask unavailable: %s", exc)
            return None


def vegetation_health(savi: np.ndarray, ndmi: np.ndarray) -> np.ndarray:
    """Composite 0-100 health score.

    SAVI carries most of the weight because it handles the bright-soil
    backgrounds that dominate Moroccan landscapes, where NDVI overestimates
    bare ground as sparse vegetation. NDMI supplies the moisture term, which
    turns down first when a canopy comes under stress.
    """
    green = np.clip((savi - 0.05) / 0.55, 0, 1)
    moist = np.clip((ndmi + 0.2) / 0.6, 0, 1)
    return (100.0 * (0.65 * green + 0.35 * moist)).astype(np.float32)
