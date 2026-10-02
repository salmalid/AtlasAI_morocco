"""Flood and wildfire modules.

Both are disaster-response products, which changes the engineering priorities:

* **Radar, not optics, for floods.** A flood arrives with the storm that caused
  it, and Sentinel-2 sees only cloud tops. Sentinel-1 penetrates cloud and works
  at night, which is why it is the operational instrument for flood mapping
  worldwide. Calm water is a specular reflector — it bounces radar away from the
  sensor, so flooded ground appears as a dark patch. That single physical fact
  is what makes the detection work.

* **Burn severity needs a before.** dNBR is a difference; there is no such thing
  as a single-date burn severity map. The module requires a pre-fire window and
  refuses to guess one.
"""

from __future__ import annotations

import datetime as dt
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
from app.ai.terrain import load_dem, relative_elevation, slope_degrees
from app.eo.catalog import NoDataError
from app.eo.raster import Grid

logger = logging.getLogger(__name__)

#: VV backscatter below this (dB) is almost certainly a smooth water surface.
WATER_DB_THRESHOLD = -16.0

FLOOD_PALETTE: dict[int, tuple[str, str]] = {
    0: ("Dry", "#f2efe9"),
    1: ("Permanent water", "#08306b"),
    2: ("Flooded", "#e6550d"),
}

#: USGS/FIREMON dNBR severity breakpoints (Key & Benson 2006).
BURN_PALETTE: dict[int, tuple[str, str]] = {
    0: ("Unburned", "#f2efe9"),
    1: ("Low severity", "#ffffb2"),
    2: ("Moderate-low severity", "#fecc5c"),
    3: ("Moderate-high severity", "#fd8d3c"),
    4: ("High severity", "#bd0026"),
}
DNBR_THRESHOLDS = (0.10, 0.27, 0.44, 0.66)


def to_db(power: np.ndarray) -> np.ndarray:
    """Linear radar power to decibels, guarding log(0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (10.0 * np.log10(np.where(power > 1e-9, power, np.nan))).astype(np.float32)


class FloodModule:
    id = "flood"
    title = "Flood Detection & Risk"
    question = "Where is water standing now, and which ground is exposed to flooding?"
    description = (
        "Maps standing water from Sentinel-1 radar — which sees through the cloud that "
        "accompanies any flood — separates it from permanent water bodies, and scores "
        "terrain susceptibility from elevation above local drainage and slope."
    )
    sensors = ("sentinel-1", "dem", "worldcover")
    needs_baseline = False
    neural_backend = "U-Net over SAR (VV/VH) flood segmentation"

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        grid = Grid.from_bbox(request.bbox, request.resolution_m or 20)
        caveats: list[str] = []

        scene = sensing.load_scene(
            request, ["vv", "vh"], sensor="sentinel-1", grid=grid,
            method="min", mask_clouds=False,
        )
        vv_db = to_db(scene.band("vv"))
        vh_db = to_db(scene.band("vh"))

        # Smooth water is dark in both polarisations; requiring both suppresses
        # radar shadow, which darkens VV but not VH the same way.
        water_now = (vv_db < WATER_DB_THRESHOLD) & (vh_db < WATER_DB_THRESHOLD - 5)

        permanent = self._permanent_water(request, grid)
        if permanent is None:
            permanent = np.zeros(grid.shape, dtype=bool)
            caveats.append(
                "Permanent-water reference unavailable — rivers and reservoirs may be "
                "reported as flooding."
            )

        flooded = water_now & ~permanent

        # Terrain susceptibility, independent of whether it is flooded today.
        try:
            elevation = load_dem(request, grid)
            hand = relative_elevation(elevation, window=31)
            slope = slope_degrees(elevation, grid.resolution_m)
            # Flat, low ground floods; steep, high ground does not.
            low_lying = np.clip(1.0 - hand / 25.0, 0, 1)
            flat = np.clip(1.0 - slope / 8.0, 0, 1)
            risk = np.clip(0.6 * low_lying + 0.4 * flat, 0, 1) * 100
            risk[~np.isfinite(elevation)] = np.nan
            # Ground that actually flooded is, definitionally, at risk.
            risk[flooded] = np.maximum(risk[flooded], 85)
        except (NoDataError, ValueError) as exc:
            caveats.append(f"Terrain susceptibility unavailable ({exc}).")
            risk = np.where(flooded, 85.0, np.nan).astype(np.float32)

        classes = np.full(grid.shape, -1, dtype=np.int16)
        observed = np.isfinite(vv_db)
        classes[observed] = 0
        classes[permanent & observed] = 1
        classes[flooded] = 2
        breakdown = class_areas(classes, FLOOD_PALETTE, grid)

        px = grid.pixel_area_km2
        flooded_km2 = float(np.count_nonzero(flooded)) * px
        high_risk_km2 = float(np.count_nonzero(risk >= 70)) * px

        metrics = {
            "flooded_km2": round(flooded_km2, 3),
            "permanent_water_km2": round(float(np.count_nonzero(permanent & observed)) * px, 3),
            "high_risk_km2": round(high_risk_km2, 2),
            "high_risk_share_pct": percent(high_risk_km2 / max(request.bbox.area_km2, 1e-6)),
            "mean_risk_score": safe_mean(risk),
            "radar_scenes": len(scene.items),
        }

        return AnalysisResult(
            module=self.id, title=self.title,
            summary=(
                f"{flooded_km2:.2f} km² of standing water detected outside permanent water "
                f"bodies. {high_risk_km2:.1f} km² of terrain scores high for flood "
                f"susceptibility."
            ),
            grid=grid, metrics=metrics, breakdown=breakdown,
            layers=[
                Layer("flood", "Flood extent", "categorical", classes.astype(np.float32),
                      palette=FLOOD_PALETTE),
                Layer("risk", "Flood susceptibility", "continuous", risk, "risk", 0, 100,
                      unit="score", description="Terrain exposure: low ground, flat, near drainage."),
                Layer("vv_db", "Radar backscatter (VV)", "continuous", vv_db, "gray", -25, 5,
                      unit="dB", description="Dark = smooth surface, typically water."),
            ],
            sources=scene.sources(),
            caveats=caveats
            + coverage_caveats(scene.coverage(), request)
            + [
                "Radar also reads dry smooth surfaces — tarmac, salt flats, sabkha, airport "
                "runways — as dark. These are classic false positives for SAR flood mapping.",
                "Flooding under dense canopy or inside built-up areas is largely invisible to "
                "this method: vegetation and buildings scatter the signal back regardless of "
                "water beneath.",
                "Susceptibility is a terrain screening score, not a hydraulic flood model. It "
                "carries no rainfall, river discharge or drainage-capacity information.",
            ],
            confidence=confidence_from_coverage(vv_db) * 0.85,
            observed_dates=scene.dates(),
        )

    def _permanent_water(self, request: AnalysisRequest, grid: Grid) -> np.ndarray | None:
        try:
            lc = sensing.load_scene(
                AnalysisRequest(
                    bbox=request.bbox,
                    start=dt.date(2020, 1, 1), end=dt.date(2022, 12, 31),
                    resolution_m=request.resolution_m,
                ),
                ["class"], sensor="worldcover", grid=grid,
                method="first", mask_clouds=False, max_items=4,
            )
            return lc.band("class") == 80
        except (NoDataError, ValueError) as exc:
            logger.info("flood: permanent water reference unavailable (%s)", exc)
            return None


class WildfireModule:
    id = "wildfire"
    title = "Wildfire & Burned Area"
    question = "What burned, and how badly?"
    description = (
        "Computes the differenced Normalised Burn Ratio between pre- and post-fire "
        "imagery, classifies burn severity on the USGS scale, and cross-checks against "
        "MODIS thermal anomaly detections."
    )
    sensors = ("sentinel-2", "modis-fire", "modis-burn")
    needs_baseline = True
    neural_backend = "U-Net burn-scar segmentation"

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        if not request.has_baseline():
            raise ValueError(
                "Burn severity is a difference between pre-fire and post-fire imagery. "
                "Provide baseline_start and baseline_end covering a pre-fire window."
            )

        bands = ["blue", "green", "red", "nir", "swir16", "swir22"]
        post = sensing.load_with_fallback(request, bands)
        pre = sensing.load_scene(
            request, bands, sensor=post.sensor,
            start=request.baseline_start, end=request.baseline_end,  # type: ignore[arg-type]
            grid=post.grid,
        )

        dnbr = pre.index("nbr") - post.index("nbr")
        # Vegetation must actually have been present to burn.
        had_vegetation = pre.index("ndvi") > 0.15
        dnbr = np.where(had_vegetation, dnbr, np.nan)

        classes = classify(dnbr, DNBR_THRESHOLDS)
        breakdown = class_areas(classes, BURN_PALETTE, post.grid)

        px = post.grid.pixel_area_km2
        burned_pixels = sum(c.pixels for c in breakdown if c.id >= 1)
        burned_km2 = burned_pixels * px
        severe_km2 = sum(c.area_km2 for c in breakdown if c.id >= 3)

        caveats = coverage_caveats(min(pre.coverage(), post.coverage()), request)
        active_fires, fire_layer = self._active_fires(request, post.grid)
        if active_fires is None:
            caveats.append("MODIS active-fire cross-check unavailable for this window.")

        metrics = {
            "burned_area_km2": round(burned_km2, 3),
            "high_severity_km2": round(severe_km2, 3),
            "mean_dnbr": safe_mean(dnbr),
            "burned_share_pct": percent(burned_pixels / max(int(np.count_nonzero(classes >= 0)), 1)),
            "active_fire_detections": active_fires,
        }

        layers = [
            Layer("severity", "Burn severity", "categorical", classes.astype(np.float32),
                  palette=BURN_PALETTE),
            Layer("dnbr", "dNBR", "continuous", dnbr, "burn", -0.2, 1.0,
                  description="Pre-fire minus post-fire NBR; higher = more severe."),
        ]
        if fire_layer is not None:
            layers.append(fire_layer)

        return AnalysisResult(
            module=self.id, title=self.title,
            summary=(
                f"{burned_km2:.2f} km² shows a burn signature, of which {severe_km2:.2f} km² "
                f"is moderate-high or high severity. Mean dNBR {metrics['mean_dnbr']}."
            ),
            grid=post.grid, metrics=metrics, breakdown=breakdown, layers=layers,
            sources=pre.sources() + post.sources(),
            caveats=caveats
            + [
                "dNBR responds to any abrupt loss of vegetation and moisture — harvest, "
                "clear-cutting, and even ploughing produce a burn-like signal. Cross-check "
                "against thermal detections before reporting an area as burned.",
                "Severity thresholds are the standard USGS values, calibrated for North "
                "American forests. Mediterranean scrub and Moroccan cork oak burn "
                "differently; local calibration against field plots would shift the classes.",
                "Smoke and active flame are not detected here — this maps the scar after "
                "the fact, using a post-fire image clear enough to read.",
            ],
            confidence=confidence_from_coverage(dnbr) * 0.85,
            observed_dates=pre.dates() + post.dates(),
        )

    def _active_fires(self, request: AnalysisRequest, grid: Grid) -> tuple[int | None, Layer | None]:
        """MODIS thermal anomalies, an independent confirmation of real fire."""
        try:
            scene = sensing.load_scene(
                request, ["fire_mask"], sensor="modis-fire", grid=grid,
                method="max", mask_clouds=False, max_items=20,
            )
        except (NoDataError, ValueError) as exc:
            logger.info("wildfire: MODIS active fire unavailable (%s)", exc)
            return None, None

        # FireMask 7/8/9 = fire at low/nominal/high confidence.
        fire = scene.band("fire_mask")
        detections = (fire >= 7).astype(np.float32)
        return int(np.count_nonzero(detections)), Layer(
            "active_fire", "MODIS active fire", "mask", detections, colormap="burn", opacity=0.95,
            description="Thermal anomalies at 1 km — independent confirmation of combustion.",
        )
