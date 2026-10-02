"""Water detection, water quality and coastal-change modules.

Surface water is the easiest thing to see from orbit and the hardest to
interpret casually. Three traps shape the code below:

* **Terrain shadow looks like water** in NDWI. The Atlas and Rif are full of
  it, so detection uses an ensemble that includes AWEI, which was designed to
  suppress exactly that confusion.
* **"Water is here" and "water is always here" are different claims.** A
  single date cannot separate a reservoir from a flooded field. Permanence is
  measured as the fraction of observations in which a pixel was wet.
* **Water quality from space is an estimate, not a measurement.** These
  indices track optical properties — turbidity, pigment — not potability.
  Nothing here can tell you whether water is safe to drink, and the module
  says so in its own output.
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
    classify,
    confidence_from_coverage,
    coverage_caveats,
    percent,
    safe_mean,
)
from app.eo import raster
from app.eo.catalog import NoDataError

logger = logging.getLogger(__name__)

WATER_PALETTE: dict[int, tuple[str, str]] = {
    0: ("Dry land", "#e8e0d0"),
    1: ("Seasonal / intermittent water", "#74c2e1"),
    2: ("Permanent water", "#08306b"),
}


def water_probability(scene: sensing.Scene) -> np.ndarray:
    """Ensemble water likelihood in 0-1 from three complementary indices.

    MNDWI is the strongest single discriminator over inland water; AWEI adds
    shadow rejection in mountainous terrain; NDWI contributes sensitivity to
    thin or sediment-laden water. Averaging their individually-calibrated
    logistic responses is more robust than thresholding any one of them.
    """
    mndwi = scene.index("mndwi")
    awei = scene.index("awei")
    ndwi = scene.index("ndwi")

    def logistic(array: np.ndarray, midpoint: float, steepness: float) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-steepness * (array - midpoint)))

    votes = np.stack(
        [
            logistic(mndwi, 0.05, 18.0),
            logistic(awei, 0.0, 8.0),
            logistic(ndwi, 0.0, 15.0),
        ]
    )
    with np.errstate(all="ignore"):
        return np.nanmean(votes, axis=0).astype(np.float32)


class WaterModule:
    id = "water"
    title = "Water Detection"
    question = "Where is there surface water, and is it permanent or seasonal?"
    description = (
        "Detects open water with a three-index ensemble, then separates permanent from "
        "seasonal water by measuring how often each pixel was wet across the available "
        "observations. Reports reservoir surface area and its change against a baseline."
    )
    sensors = ("sentinel-2", "landsat")
    needs_baseline = False
    neural_backend = "U-Net / DeepLabV3+ water segmentation"

    #: Wet in this share of observations or more counts as permanent.
    PERMANENT_FRACTION = 0.8
    #: Ensemble probability above which a single observation counts as wet.
    WET_THRESHOLD = 0.5

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        scene = sensing.load_with_fallback(
            request, ["green", "red", "nir", "swir16", "swir22"]
        )
        probability = water_probability(scene)
        occurrence = self._occurrence(request, scene)

        classes = np.full(probability.shape, -1, dtype=np.int16)
        observed = np.isfinite(probability)
        classes[observed] = 0
        classes[observed & (occurrence > 0.15)] = 1
        classes[observed & (occurrence >= self.PERMANENT_FRACTION)] = 2

        breakdown = class_areas(classes, WATER_PALETTE, scene.grid)
        permanent_km2 = next(c.area_km2 for c in breakdown if c.id == 2)
        seasonal_km2 = next(c.area_km2 for c in breakdown if c.id == 1)

        metrics: dict[str, object] = {
            "water_area_km2": round(permanent_km2 + seasonal_km2, 3),
            "permanent_water_km2": round(permanent_km2, 3),
            "seasonal_water_km2": round(seasonal_km2, 3),
            "water_share_pct": percent(
                (permanent_km2 + seasonal_km2) / max(request.bbox.area_km2, 1e-6)
            ),
            "observations_used": len(scene.items),
            "mean_mndwi": safe_mean(scene.index("mndwi")),
        }

        caveats = coverage_caveats(scene.coverage(), request) + [
            "Permanence is measured only across the scenes in this time window. A window "
            "shorter than a year cannot distinguish a genuinely permanent lake from one "
            "that happens to be full this season.",
        ]
        summary = (
            f"{metrics['water_area_km2']} km² of surface water detected — "
            f"{metrics['permanent_water_km2']} km² permanent and "
            f"{metrics['seasonal_water_km2']} km² seasonal."
        )

        layers = [
            Layer("classes", "Water permanence", "categorical", classes.astype(np.float32),
                  palette=WATER_PALETTE, description="Permanent vs seasonal surface water."),
            Layer("probability", "Water probability", "continuous", probability, "water", 0, 1,
                  description="Ensemble of MNDWI, AWEI and NDWI."),
            Layer("occurrence", "Wet-observation frequency", "continuous", occurrence, "water", 0, 1,
                  unit="fraction", description="Share of clear observations in which the pixel was wet."),
        ]

        # Reservoir monitoring: the headline question is whether storage is falling.
        if request.has_baseline():
            try:
                baseline = sensing.load_scene(
                    request, ["green", "red", "nir", "swir16", "swir22"],
                    sensor=scene.sensor, start=request.baseline_start,  # type: ignore[arg-type]
                    end=request.baseline_end, grid=scene.grid,  # type: ignore[arg-type]
                )
                base_water = water_probability(baseline) > self.WET_THRESHOLD
                now_water = probability > self.WET_THRESHOLD
                base_km2 = float(np.count_nonzero(base_water)) * scene.grid.pixel_area_km2
                now_km2 = float(np.count_nonzero(now_water)) * scene.grid.pixel_area_km2
                change_pct = ((now_km2 - base_km2) / base_km2 * 100) if base_km2 > 0.01 else None

                # -1 dried out, 0 unchanged, +1 newly flooded.
                delta = np.zeros(probability.shape, dtype=np.float32)
                delta[base_water & ~now_water] = -1.0
                delta[~base_water & now_water] = 1.0

                metrics.update(
                    {
                        "baseline_water_km2": round(base_km2, 3),
                        "current_water_km2": round(now_km2, 3),
                        "water_change_km2": round(now_km2 - base_km2, 3),
                        "water_change_pct": round(change_pct, 2) if change_pct is not None else None,
                        "lost_water_km2": round(
                            float(np.count_nonzero(delta < 0)) * scene.grid.pixel_area_km2, 3
                        ),
                        "gained_water_km2": round(
                            float(np.count_nonzero(delta > 0)) * scene.grid.pixel_area_km2, 3
                        ),
                    }
                )
                layers.append(
                    Layer("change", "Water gain / loss", "continuous", delta, "change", -1, 1,
                          description="Red = water lost since baseline, blue = water gained.")
                )
                if change_pct is not None:
                    direction = "shrunk" if change_pct < 0 else "grown"
                    summary += (
                        f" Surface water has {direction} {abs(change_pct):.1f}% since the "
                        f"baseline window ({base_km2:.2f} → {now_km2:.2f} km²)."
                    )
            except NoDataError as exc:
                caveats.append(f"Baseline comparison unavailable — {exc}")

        return AnalysisResult(
            module=self.id, title=self.title, summary=summary, grid=scene.grid,
            metrics=metrics, breakdown=breakdown, layers=layers,
            sources=scene.sources(), caveats=caveats,
            confidence=confidence_from_coverage(probability),
            observed_dates=scene.dates(),
        )

    def _occurrence(self, request: AnalysisRequest, scene: sensing.Scene) -> np.ndarray:
        """Per-pixel fraction of clear observations that were wet.

        Recomputed per scene rather than from the composite: a median composite
        of a shoreline that moves would report a single averaged edge, which is
        precisely the information permanence needs.
        """
        collection = scene.collection
        wet_count = np.zeros(scene.shape, dtype=np.float32)
        clear_count = np.zeros(scene.shape, dtype=np.float32)

        for item in scene.items:
            try:
                bands = raster.load_bands(
                    [item], collection, ["green", "red", "nir", "swir16", "swir22"],
                    scene.grid, method="first",
                )
            except (NoDataError, ValueError) as exc:
                logger.info("occurrence: skipping %s (%s)", item.id, exc)
                continue
            single = sensing.Scene(
                sensor=scene.sensor, collection=collection, grid=scene.grid,
                bands=bands, items=[item], start=scene.start, end=scene.end,
            )
            probability = water_probability(single)
            clear = np.isfinite(probability)
            clear_count += clear
            wet_count += np.where(clear & (probability > self.WET_THRESHOLD), 1.0, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(clear_count > 0, wet_count / clear_count, np.nan).astype(np.float32)


class WaterQualityModule:
    id = "water_quality"
    title = "Water Quality Estimation"
    question = "How turbid or algae-rich is this water body?"
    description = (
        "Estimates optical water-quality proxies — turbidity and chlorophyll-a — from "
        "Sentinel-2 spectral bands over detected water only. These are optical "
        "properties, not chemical or microbiological measurements."
    )
    sensors = ("sentinel-2",)
    needs_baseline = False
    neural_backend = None

    #: Non-negotiable. Every response carries these.
    HARD_CAVEATS = (
        "These are optical proxies derived from reflectance, not laboratory measurements. "
        "They must be calibrated against in-situ samples before being used for any decision.",
        "Satellite imagery CANNOT determine whether water is safe to drink. Pathogens, "
        "heavy metals, nitrates and dissolved chemicals are all optically invisible. "
        "Do not use this module to assess potability.",
        "Optically shallow water returns bottom reflectance, which inflates apparent "
        "turbidity in the shallows of a reservoir or lagoon.",
    )

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        scene = sensing.load_scene(
            request, ["blue", "green", "red", "nir", "swir16", "swir22"], sensor="sentinel-2"
        )
        probability = water_probability(scene)
        water_mask = probability > 0.5

        water_pixels = int(np.count_nonzero(water_mask))
        if water_pixels < 25:
            raise NoDataError(
                "Not enough open water in this area to assess quality.",
                hint="Point the analysis at a reservoir, lagoon or coastal strip.",
            )

        turbidity = np.where(water_mask, scene.index("ndti"), np.nan)
        chlorophyll = np.where(water_mask, scene.index("chl"), np.nan)

        turbidity_classes = classify(turbidity, (-0.10, 0.0, 0.10))
        turbidity_palette = {
            0: ("Clear", "#08306b"),
            1: ("Slightly turbid", "#4292c6"),
            2: ("Turbid", "#fdd0a2"),
            3: ("Highly turbid", "#e6550d"),
        }
        breakdown = class_areas(turbidity_classes, turbidity_palette, scene.grid)

        # High blue/green ratio means blue light is *not* being absorbed, i.e.
        # low pigment. The bloom signal is therefore a *low* ratio.
        bloom_risk = np.where(water_mask & (chlorophyll < 0.9) & (scene.index("ndvi") > -0.1), 1.0, 0.0)
        bloom_km2 = float(np.count_nonzero(bloom_risk)) * scene.grid.pixel_area_km2

        metrics = {
            "water_area_km2": round(water_pixels * scene.grid.pixel_area_km2, 3),
            "mean_turbidity_index": safe_mean(turbidity),
            "mean_chlorophyll_ratio": safe_mean(chlorophyll),
            "turbid_share_pct": percent(
                sum(c.pixels for c in breakdown if c.id >= 2) / max(water_pixels, 1)
            ),
            "possible_bloom_km2": round(bloom_km2, 3),
            "recreational_indication": self._recreational_hint(breakdown, water_pixels, bloom_km2),
        }

        return AnalysisResult(
            module=self.id, title=self.title,
            summary=(
                f"Across {metrics['water_area_km2']} km² of water, "
                f"{metrics['turbid_share_pct']}% of the surface reads as turbid and "
                f"{metrics['possible_bloom_km2']} km² shows a possible algal signature. "
                "These are optical estimates requiring ground validation."
            ),
            grid=scene.grid, metrics=metrics, breakdown=breakdown,
            layers=[
                Layer("turbidity", "Turbidity class", "categorical",
                      turbidity_classes.astype(np.float32), palette=turbidity_palette),
                Layer("ndti", "Turbidity index", "continuous", turbidity, "turbidity", -0.3, 0.4),
                Layer("chlorophyll", "Chlorophyll proxy", "continuous", chlorophyll, "algae", 0.5, 2.0,
                      description="Blue/green ratio; lower values suggest more pigment."),
                Layer("bloom", "Possible algal bloom", "mask", bloom_risk, colormap="algae"),
            ],
            sources=scene.sources(),
            caveats=list(self.HARD_CAVEATS) + coverage_caveats(scene.coverage(), request),
            confidence=confidence_from_coverage(turbidity) * 0.8,
            observed_dates=scene.dates(),
        )

    @staticmethod
    def _recreational_hint(breakdown, water_pixels: int, bloom_km2: float) -> str:
        turbid_share = sum(c.pixels for c in breakdown if c.id >= 2) / max(water_pixels, 1)
        if bloom_km2 > 0.5:
            return "Possible bloom detected — verify on the ground before recreational use."
        if turbid_share > 0.5:
            return "Optically turbid — reduced clarity, cause unknown from imagery alone."
        return "Optically clear in this observation. Clarity is not a safety assessment."


class CoastalModule:
    id = "coastal"
    title = "Coastal Monitoring"
    question = "Is the shoreline advancing or retreating?"
    description = (
        "Extracts the land/water boundary at two epochs and measures erosion and "
        "accretion between them, for Morocco's ~3,500 km Atlantic and Mediterranean coast."
    )
    sensors = ("sentinel-2", "landsat")
    needs_baseline = True
    neural_backend = "Siamese U-Net shoreline change detection"

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        if not request.has_baseline():
            raise ValueError("Coastal change analysis requires a baseline window to compare against.")

        bands = ["green", "red", "nir", "swir16", "swir22"]
        current = sensing.load_with_fallback(request, bands)
        past = sensing.load_scene(
            request, bands, sensor=current.sensor,
            start=request.baseline_start, end=request.baseline_end,  # type: ignore[arg-type]
            grid=current.grid,
        )

        now_water = water_probability(current) > 0.5
        then_water = water_probability(past) > 0.5
        valid = current.observed_mask() & past.observed_mask()

        eroded = valid & ~then_water & now_water   # land became sea
        accreted = valid & then_water & ~now_water  # sea became land

        px = current.grid.pixel_area_km2
        eroded_km2 = float(np.count_nonzero(eroded)) * px
        accreted_km2 = float(np.count_nonzero(accreted)) * px

        change = np.zeros(now_water.shape, dtype=np.float32)
        change[eroded] = -1.0
        change[accreted] = 1.0

        palette = {0: ("Erosion (land lost)", "#d73027"), 1: ("Accretion (land gained)", "#4393c3")}
        classes = np.full(change.shape, -1, dtype=np.int16)
        classes[eroded] = 0
        classes[accreted] = 1

        # Mean shoreline displacement, from area change per unit of coast length.
        shoreline_px = int(np.count_nonzero(_edges(now_water & valid)))
        coast_length_km = shoreline_px * current.grid.resolution_m / 1000
        net_km2 = accreted_km2 - eroded_km2
        mean_shift_m = (net_km2 * 1e6 / (coast_length_km * 1000)) if coast_length_km > 0.1 else None

        metrics = {
            "eroded_km2": round(eroded_km2, 4),
            "accreted_km2": round(accreted_km2, 4),
            "net_land_change_km2": round(net_km2, 4),
            "shoreline_length_km": round(coast_length_km, 2),
            "mean_shoreline_shift_m": round(mean_shift_m, 2) if mean_shift_m is not None else None,
            "direction": "accreting" if net_km2 > 0 else "eroding" if net_km2 < 0 else "stable",
        }

        return AnalysisResult(
            module=self.id, title=self.title,
            summary=(
                f"Shoreline is {metrics['direction']}: {eroded_km2:.3f} km² of land lost and "
                f"{accreted_km2:.3f} km² gained along {coast_length_km:.1f} km of coast."
            ),
            grid=current.grid, metrics=metrics,
            breakdown=class_areas(classes, palette, current.grid),
            layers=[
                Layer("change", "Shoreline change", "categorical", classes.astype(np.float32),
                      palette=palette),
                Layer("water_now", "Water extent (current)", "mask",
                      (now_water & valid).astype(np.float32), colormap="water"),
            ],
            sources=current.sources() + past.sources(),
            caveats=coverage_caveats(min(current.coverage(), past.coverage()), request)
            + [
                "Tide state is not corrected for. On a gently sloping beach, the tidal "
                "difference between two acquisitions can exceed a decade of real erosion — "
                "compare epochs of several months to average it out, and treat single-date "
                "comparisons as unreliable.",
                "Wave run-up and wet sand both read as water at 10 m resolution.",
            ],
            confidence=confidence_from_coverage(change) * 0.75,
            observed_dates=current.dates() + past.dates(),
        )


def _edges(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a boolean mask, via a 4-neighbour difference."""
    padded = np.pad(mask.astype(np.int8), 1, mode="edge")
    neighbours = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    )
    return mask & (neighbours < 4)
