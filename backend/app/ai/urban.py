"""Urban expansion, construction detection and night-lights modules.

Urban change detection is the platform's most operationally sensitive output —
"this building is new and unpermitted" is an accusation, not an observation.
The code therefore separates two claims carefully:

* *A built-up surface appeared here between date A and date B.* This is
  measurable from reflectance, and the module reports it.
* *That construction is illegal.* This is a legal determination that requires
  cadastral and permit records the platform does not have. The module produces
  review candidates, never verdicts, and refuses to imply otherwise.
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

logger = logging.getLogger(__name__)


def builtup_probability(scene: sensing.Scene) -> np.ndarray:
    """Likelihood that a pixel is an impervious built surface, 0-1.

    Built-up surfaces are bright in SWIR relative to NIR (high NDBI), not
    vegetated (low NDVI), and moderately bright overall. Bare desert satisfies
    the first two conditions too — which is why the BSI term is subtracted
    rather than added. Distinguishing concrete from bare rock is the genuinely
    hard part of this problem in an arid country, and it is where a trained
    model earns its keep.
    """
    ndbi = scene.index("ndbi")
    ndvi = scene.index("ndvi")
    bsi = scene.index("bsi")

    built = np.clip((ndbi + 0.05) / 0.35, 0, 1)
    unvegetated = np.clip((0.25 - ndvi) / 0.25, 0, 1)
    not_bare_soil = np.clip((0.35 - bsi) / 0.45, 0, 1)

    score = 0.45 * built + 0.30 * unvegetated + 0.25 * not_bare_soil
    score[~np.isfinite(ndbi) | ~np.isfinite(ndvi)] = np.nan
    return score.astype(np.float32)


class UrbanExpansionModule:
    id = "urban"
    title = "Urban Expansion Detection"
    question = "Where has the city grown, and by how much?"
    description = (
        "Compares built-up surface between two time windows to find new construction, "
        "new roads and industrial development, and quantifies the growth rate."
    )
    sensors = ("sentinel-2", "landsat")
    needs_baseline = True
    neural_backend = "Siamese CNN / change-detection network (e.g. BIT, ChangeFormer)"

    BUILT_THRESHOLD = 0.55

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        if not request.has_baseline():
            raise ValueError(
                "Urban expansion analysis needs two time windows. Provide baseline_start "
                "and baseline_end to compare against."
            )

        bands = ["blue", "green", "red", "nir", "swir16"]
        current = sensing.load_with_fallback(request, bands)
        past = sensing.load_scene(
            request, bands, sensor=current.sensor,
            start=request.baseline_start, end=request.baseline_end,  # type: ignore[arg-type]
            grid=current.grid,
        )

        now_score = builtup_probability(current)
        then_score = builtup_probability(past)
        valid = np.isfinite(now_score) & np.isfinite(then_score)

        # Thresholding each epoch independently is the classic change-detection
        # error: sun angle, atmosphere and phenology shift the whole score
        # distribution between dates, so a fixed cutoff reports that shift as
        # construction. Relative radiometric normalisation removes it — the
        # baseline is rescaled so its median and spread match the current
        # epoch's before either is thresholded.
        then_score, normalisation = _normalise_to(then_score, now_score, valid)

        now_built = valid & (now_score > self.BUILT_THRESHOLD)
        then_built = valid & (then_score > self.BUILT_THRESHOLD)

        # Requiring a real jump in the score, not just a threshold crossing,
        # suppresses pixels that merely wobble around the boundary.
        new_built = now_built & ~then_built & ((now_score - then_score) > 0.15)
        removed = then_built & ~now_built & ((then_score - now_score) > 0.15)

        px = current.grid.pixel_area_km2
        new_km2 = float(np.count_nonzero(new_built)) * px
        then_km2 = float(np.count_nonzero(then_built)) * px
        now_km2 = float(np.count_nonzero(now_built)) * px

        years = max((request.end - request.baseline_end).days / 365.25, 0.01)  # type: ignore[operator]
        # Growth relative to a near-zero baseline is a meaningless percentage —
        # it says more about the detector than about the city. Require the
        # baseline to be a substantial share of the current extent first.
        comparable = then_km2 > 0.05 and then_km2 > 0.25 * now_km2
        growth_pct = (new_km2 / then_km2 * 100) if comparable else None

        classes = np.full(now_score.shape, -1, dtype=np.int16)
        classes[valid] = 0
        classes[then_built] = 1
        classes[new_built] = 2
        classes[removed] = 3
        palette = {
            0: ("Not built-up", "#f2efe9"),
            1: ("Existing built-up", "#969696"),
            2: ("New construction", "#d94801"),
            3: ("Cleared / demolished", "#4393c3"),
        }

        metrics = {
            "baseline_builtup_km2": round(then_km2, 3),
            "current_builtup_km2": round(now_km2, 3),
            "new_construction_km2": round(new_km2, 3),
            "cleared_km2": round(float(np.count_nonzero(removed)) * px, 3),
            "growth_pct": round(growth_pct, 2) if growth_pct is not None else None,
            "annual_growth_km2": round(new_km2 / years, 3),
            "years_elapsed": round(years, 2),
            "urbanization_score": round(min(now_km2 / max(request.bbox.area_km2, 1e-6), 1.0) * 100, 1),
        }

        growth_text = (
            f"{metrics['growth_pct']}% growth" if growth_pct is not None else "growth rate unavailable"
        )
        instability_caveats: list[str] = []
        if not comparable:
            instability_caveats.append(
                f"Detected built-up area differs sharply between the two epochs "
                f"({then_km2:.2f} vs {now_km2:.2f} km²), which is far more than urban growth "
                "can explain. The detector is responding to differences between the two "
                "composites rather than to construction, so the growth percentage is "
                "withheld. Compare windows from the same season, and prefer a baseline with "
                "similar cloud-free coverage."
            )
        if normalisation is not None:
            instability_caveats.append(
                f"Baseline scores were radiometrically normalised to the current epoch "
                f"(scale {normalisation[0]:.2f}, offset {normalisation[1]:+.3f}). This "
                "removes date-to-date brightness differences but slightly damps genuine "
                "area-wide change."
            )

        return AnalysisResult(
            module=self.id, title=self.title,
            summary=(
                f"{new_km2:.2f} km² of new built-up surface appeared over {years:.1f} years "
                f"({metrics['annual_growth_km2']} km²/yr, {growth_text}). Built-up area went "
                f"from {then_km2:.2f} to {now_km2:.2f} km²."
            ),
            grid=current.grid, metrics=metrics,
            breakdown=class_areas(classes, palette, current.grid),
            layers=[
                Layer("change", "Urban change", "categorical", classes.astype(np.float32),
                      palette=palette),
                Layer("new_construction", "New construction", "mask",
                      new_built.astype(np.float32), colormap="urban", opacity=0.9),
                Layer("builtup_now", "Built-up probability (current)", "continuous",
                      now_score, "urban", 0, 1),
            ],
            sources=current.sources() + past.sources(),
            caveats=coverage_caveats(min(current.coverage(), past.coverage()), request)
            + instability_caveats
            + [
                "Bare soil, quarries and dry riverbeds share a spectral signature with "
                "concrete in arid terrain; some flagged pixels will be ground disturbance "
                "rather than buildings.",
                "At 10-20 m resolution, individual houses are below the detection limit. "
                "This measures built-up *surface*, not building count.",
                "Seasonal effects matter: a field harvested between the two dates can read "
                "as newly built. Compare like seasons where possible.",
            ],
            confidence=confidence_from_coverage(now_score, then_score) * 0.85,
            observed_dates=current.dates() + past.dates(),
        )


class ConstructionReviewModule:
    id = "construction_review"
    title = "Construction Change Review"
    question = "Which new construction sites warrant a planning review?"
    description = (
        "Ranks newly built-up areas by size and isolation to produce a prioritised list "
        "of sites for human review. It identifies construction that is new — it cannot "
        "and does not determine whether construction is authorised."
    )
    sensors = ("sentinel-2",)
    needs_baseline = True
    neural_backend = "Siamese CNN + building footprint segmentation"

    #: Ignore specks: below this, a cluster is more likely noise than a site.
    MIN_SITE_KM2 = 0.004

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        base = UrbanExpansionModule().run(request)
        new_layer = base.layer("new_construction")
        assert new_layer is not None
        new_built = new_layer.array > 0.5

        sites = _cluster(new_built, min_pixels=int(self.MIN_SITE_KM2 / base.grid.pixel_area_km2))
        ranked = sorted(sites, key=lambda s: s["pixels"], reverse=True)[:50]

        for site in ranked:
            row, col = site["centroid"]
            lon, lat = _pixel_to_lonlat(base.grid, row, col)
            site["lon"], site["lat"] = round(lon, 6), round(lat, 6)
            site["area_km2"] = round(site["pixels"] * base.grid.pixel_area_km2, 4)
            site.pop("centroid")
            site.pop("pixels")

        metrics = {
            "sites_detected": len(sites),
            "sites_returned": len(ranked),
            "total_new_construction_km2": base.metrics.get("new_construction_km2"),
            "largest_site_km2": ranked[0]["area_km2"] if ranked else 0.0,
            "review_sites": ranked,
        }

        return AnalysisResult(
            module=self.id, title=self.title,
            summary=(
                f"{len(sites)} distinct new-construction sites detected, totalling "
                f"{base.metrics.get('new_construction_km2')} km². The {len(ranked)} largest "
                "are listed with coordinates for field verification."
            ),
            grid=base.grid, metrics=metrics, breakdown=base.breakdown,
            layers=[layer for layer in base.layers if layer.key in ("new_construction", "change")],
            sources=base.sources,
            caveats=base.caveats
            + [
                "THIS MODULE DOES NOT DETECT ILLEGAL CONSTRUCTION. It detects construction "
                "that is new. Legality depends on permits, cadastral boundaries and zoning "
                "records that are not part of this analysis.",
                "Output is a prioritised list of review candidates for a human planner. "
                "Enforcement action must never be taken on this signal alone.",
            ],
            confidence=base.confidence * 0.8,
            observed_dates=base.observed_dates,
        )


class NightLightsModule:
    id = "nightlights"
    title = "Night Light Analysis"
    question = "Which areas are lit at night, and where is electrification spreading?"
    description = (
        "Measures nighttime radiance as a proxy for electrification, urban activity and "
        "economic development. Requires a VIIRS data source to be configured."
    )
    sensors = ()
    needs_baseline = False
    neural_backend = None

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        # Being straight about this rather than shipping a plausible-looking
        # fake: the Planetary Computer STAC catalog this platform reads carries
        # no VIIRS nighttime-lights collection, so there is nothing to analyse
        # until an operator wires up a source.
        raise NoDataError(
            "No nighttime-lights data source is configured.",
            hint=(
                "VIIRS DNB is not in the Planetary Computer STAC catalog. To enable this "
                "module, register a source in app/ai/urban.py — NASA Black Marble "
                "(VNP46A2, via LAADS DAAC, requires a free Earthdata login) or the NOAA/"
                "Colorado School of Mines annual VNL composites are the usual choices. "
                "See docs/DATA_SOURCES.md."
            ),
        )


def _normalise_to(
    source: np.ndarray, reference: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Rescale ``source`` so its centre and spread match ``reference``.

    Relative radiometric normalisation, the standard remedy for comparing two
    acquisition dates. Median and median-absolute-deviation are used rather than
    mean and standard deviation because the pixels that genuinely changed are
    outliers, and robust statistics are exactly what stops those outliers from
    driving the correction that is supposed to preserve them.

    The assumption is that most of the scene did not change between epochs —
    true over a few years of urban growth, false after a flood or a fire. The
    correction is returned so the module can disclose it.
    """
    src_values = source[valid & np.isfinite(source)]
    ref_values = reference[valid & np.isfinite(reference)]
    if src_values.size < 100 or ref_values.size < 100:
        return source, None

    src_median = float(np.median(src_values))
    ref_median = float(np.median(ref_values))
    src_spread = float(np.median(np.abs(src_values - src_median)))
    ref_spread = float(np.median(np.abs(ref_values - ref_median)))

    # A degenerate spread means a near-constant scene; scaling it would amplify
    # noise into signal, so only the offset is applied.
    scale = (ref_spread / src_spread) if src_spread > 1e-4 and ref_spread > 1e-4 else 1.0
    scale = float(np.clip(scale, 0.5, 2.0))
    offset = ref_median - scale * src_median

    return (source * scale + offset).astype(np.float32), (scale, offset)


def _cluster(mask: np.ndarray, min_pixels: int = 5) -> list[dict]:
    """Connected-component labelling over a boolean mask (8-connectivity).

    Hand-rolled with a union-find over a two-pass scan so the platform does not
    pull in SciPy for one function. Returns cluster pixel counts and centroids.
    """
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    next_label = 1
    for row in range(height):
        for col in range(width):
            if not mask[row, col]:
                continue
            neighbours = []
            if row > 0:
                for dc in (-1, 0, 1):
                    c = col + dc
                    if 0 <= c < width and labels[row - 1, c]:
                        neighbours.append(int(labels[row - 1, c]))
            if col > 0 and labels[row, col - 1]:
                neighbours.append(int(labels[row, col - 1]))

            if neighbours:
                label = min(neighbours)
                labels[row, col] = label
                for other in neighbours:
                    union(label, other)
            else:
                labels[row, col] = next_label
                parent[next_label] = next_label
                next_label += 1

    if next_label == 1:
        return []

    resolved = np.zeros(next_label, dtype=np.int32)
    for label in range(1, next_label):
        resolved[label] = find(label)
    flat = resolved[labels]

    clusters: list[dict] = []
    for label in np.unique(flat):
        if label == 0:
            continue
        rows, cols = np.nonzero(flat == label)
        if rows.size < min_pixels:
            continue
        clusters.append(
            {"pixels": int(rows.size), "centroid": (float(rows.mean()), float(cols.mean()))}
        )
    return clusters


def _pixel_to_lonlat(grid, row: float, col: float) -> tuple[float, float]:
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp_transform

    x, y = grid.transform * (col + 0.5, row + 0.5)
    lons, lats = warp_transform(grid.crs, CRS.from_epsg(4326), [x], [y])
    return lons[0], lats[0]
