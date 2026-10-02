"""Natural-language interface to the analysis modules.

The assistant's job is narrow and unglamorous: turn a question into a
*plan* — which module, over which area, across which dates — then execute it and
report what the pixels actually said. It never answers from its own knowledge.
Every number in a reply comes from imagery analysed for that request, and a
question the modules cannot answer gets a plain "I can't answer that", not a
plausible guess.

There are two planners. A deterministic intent router handles the common shapes
with no external dependency, so the feature works out of the box. When a
Gemini API key is present, Gemini plans instead — it handles paraphrase,
implicit time references and multi-part questions far better. Both produce the
same :class:`Plan`, and execution is identical either way, so the LLM never
touches the numbers.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

from app.ai import registry
from app.ai.base import AnalysisRequest
from app.core.config import settings
from app.core.geo import BBox
from app.eo.catalog import NoDataError
from app.services import regions

logger = logging.getLogger(__name__)

Mode = Literal["single", "scan"]

#: Keyword -> module, checked in order. Earlier entries win, so put the
#: specific phrases ("water quality") ahead of the general ones ("water").
INTENT_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("water quality", "turbid", "algae", "pollut", "swim", "drink"), "water_quality"),
    (("drought", "dry spell", "water stress", "arid"), "drought"),
    (("reservoir", "dam", "lake", "water level", "losing water", "storage"), "water"),
    (("flood", "inundat", "standing water", "overflow"), "flood"),
    (("fire", "wildfire", "burn", "burnt"), "wildfire"),
    (("urban", "city grow", "expansion", "sprawl", "construction", "new neighbour", "new neighbor"), "urban"),
    (("illegal", "unpermitted", "encroach", "review"), "construction_review"),
    (("crop", "farm", "agricultur", "harvest", "irrigat", "field"), "agriculture"),
    (("coast", "shoreline", "erosion", "beach"), "coastal"),
    (("road", "highway", "track", "infrastructure"), "roads"),
    (("solar", "photovolta", "pv "), "solar"),
    (("wind farm", "wind turbine", "wind power"), "wind"),
    (("night light", "nighttime light", "electrification", "lit at night"), "nightlights"),
    (("land cover", "landcover", "land use", "what is there", "classify"), "landcover"),
    (("elevation", "terrain", "slope", "altitude", "mountain"), "terrain"),
    (("vegetation", "forest", "green", "ndvi", "deforest", "desertif", "tree"), "vegetation"),
    (("satellite image", "imagery", "what does it look like", "picture", "photo"), "imagery"),
)

#: Phrases that mean "compare against an earlier period".
CHANGE_WORDS = (
    "chang", "increas", "decreas", "grow", "shrink", "lost", "losing", "gain",
    "compare", "since", "trend", "expand", "new ", "evolv", "before", "versus", " vs ",
)

#: Phrases that mean "look at many places and rank them".
SCAN_WORDS = (
    "which", "where", "rank", "list", "find", "cities", "regions", "provinces",
    "reservoirs", "areas", "top ", "worst", "most", "highest", "lowest",
)


@dataclass
class Plan:
    """A resolved, executable interpretation of a question."""

    module: str
    mode: Mode = "single"
    places: list[regions.Place] = field(default_factory=list)
    bbox: BBox | None = None
    start: dt.date = dt.date.today() - dt.timedelta(days=90)
    end: dt.date = dt.date.today()
    baseline_start: dt.date | None = None
    baseline_end: dt.date | None = None
    #: How the plan was produced, surfaced to the caller for transparency.
    planner: str = "rules"
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "mode": self.mode,
            "places": [p.id for p in self.places],
            "window": {"start": self.start.isoformat(), "end": self.end.isoformat()},
            "baseline": (
                {"start": self.baseline_start.isoformat(), "end": self.baseline_end.isoformat()}
                if self.baseline_start and self.baseline_end
                else None
            ),
            "planner": self.planner,
            "reasoning": self.reasoning,
        }


class UnanswerableError(ValueError):
    """The question is outside what the platform's modules can answer."""


# --- rule-based planner -----------------------------------------------------


def route(question: str, *, today: dt.date | None = None) -> Plan:
    """Deterministic intent routing. No network, no model, no surprises."""
    today = today or dt.date.today()
    text = question.lower()

    module = next(
        (mod for keywords, mod in INTENT_RULES if any(k in text for k in keywords)),
        None,
    )
    # "Compare Casablanca in 2022 vs 2026" names no subject, only a place and
    # two dates. The honest reading is "show me both" — a visual comparison.
    if module is None and any(word in text for word in ("compare", " vs ", "versus")):
        module = "imagery"
    if module is None:
        raise UnanswerableError(
            "I couldn't tell which kind of analysis this needs. Try naming the subject "
            "directly — drought, water, vegetation, flooding, fire, urban growth, "
            "cropland, land cover, terrain, coastline, roads or solar."
        )

    end, start = _parse_window(text, today)
    baseline_start, baseline_end = _parse_baseline(text, start, module)
    places, mode = _parse_places(text, module)

    return Plan(
        module=module, mode=mode, places=places,
        start=start, end=end,
        baseline_start=baseline_start, baseline_end=baseline_end,
        planner="rules",
        reasoning=f"Matched on keywords for '{module}'.",
    )


def _parse_window(text: str, today: dt.date) -> tuple[dt.date, dt.date]:
    """Resolve the observation window, defaulting to a recent 90-day span."""
    years = sorted({int(y) for y in re.findall(r"\b(20[0-2]\d)\b", text)})
    if years:
        latest = years[-1]
        if latest >= today.year:
            return today, today - dt.timedelta(days=90)
        return dt.date(latest, 12, 31), dt.date(latest, 1, 1)

    if "last month" in text or "past month" in text:
        return today, today - dt.timedelta(days=35)
    if "last year" in text or "past year" in text:
        return today, today - dt.timedelta(days=365)
    if "this year" in text:
        return today, dt.date(today.year, 1, 1)
    if "decade" in text:
        return today, today - dt.timedelta(days=3650)
    return today, today - dt.timedelta(days=90)


def _parse_baseline(
    text: str, start: dt.date, module: str
) -> tuple[dt.date | None, dt.date | None]:
    """Pick a comparison window when the question implies change."""
    module_obj = registry.get(module)
    wants_change = any(word in text for word in CHANGE_WORDS)
    if not wants_change and not module_obj.needs_baseline:
        return None, None

    years = sorted({int(y) for y in re.findall(r"\b(20[0-2]\d)\b", text)})
    if len(years) >= 2:
        # "Compare Casablanca in 2022 vs 2026" — the earlier year is the baseline.
        return dt.date(years[0], 1, 1), dt.date(years[0], 12, 31)

    # Default: the same season one year earlier, which controls for phenology.
    anchor = start - dt.timedelta(days=365)
    return anchor - dt.timedelta(days=45), anchor + dt.timedelta(days=45)


def _parse_places(text: str, module: str) -> tuple[list[regions.Place], Mode]:
    """Resolve named places, or pick a sensible set to scan."""
    #: {place: the term that matched it}, so overlaps can be resolved below.
    matched: dict[str, tuple[regions.Place, set[str]]] = {}
    for place in regions.ALL_PLACES:
        hits = {
            term
            for term in (place.name.lower(), *place.aliases)
            if _mentions(text, term)
        }
        if hits:
            matched[place.id] = (place, hits)

    if matched:
        # "Marrakech" matches both the city and the Marrakech-Safi region, which
        # would silently turn a single-city question into a two-area scan. When
        # a region matched only on a term that also named a city or reservoir,
        # the specific place is what the user meant.
        specific_terms = {
            term
            for place, hits in matched.values()
            if place.kind != "region"
            for term in hits
        }
        resolved = {
            place_id: place
            for place_id, (place, hits) in matched.items()
            if place.kind != "region" or not hits <= specific_terms
        }
        return list(resolved.values()), "single" if len(resolved) == 1 else "scan"

    if any(word in text for word in SCAN_WORDS):
        return list(_default_scan_set(text, module)), "scan"

    raise UnanswerableError(
        "I couldn't work out which area you mean. Name a city, region or reservoir "
        "(for example Marrakech, Souss-Massa, or Barrage Al Massira), or ask a "
        "'which…' question and I'll scan the country."
    )


def _mentions(text: str, term: str) -> bool:
    """Whole-word containment, so 'safi' doesn't match inside 'safirst'."""
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def _default_scan_set(text: str, module: str) -> tuple[regions.Place, ...]:
    if "reservoir" in text or "dam" in text or module in ("water", "water_quality"):
        return regions.RESERVOIRS[:8]
    if "cities" in text or "city" in text or module in ("urban", "construction_review", "roads"):
        return regions.CITIES[:8]
    return regions.REGIONS


# --- LLM planner (optional) -------------------------------------------------

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {
            "type": "boolean",
            "description": "False if no available module can answer this question.",
        },
        "module": {"type": "string", "description": "Module id, or empty if not answerable."},
        "mode": {"type": "string", "enum": ["single", "scan"]},
        "places": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Place ids from the supplied list. Empty means scan defaults.",
        },
        "start": {"type": "string", "description": "Observation window start, YYYY-MM-DD."},
        "end": {"type": "string", "description": "Observation window end, YYYY-MM-DD."},
        "baseline_start": {"type": "string", "description": "Comparison window start, or empty."},
        "baseline_end": {"type": "string", "description": "Comparison window end, or empty."},
        "reasoning": {"type": "string", "description": "One sentence on the interpretation."},
    },
    "required": [
        "answerable", "module", "mode", "places",
        "start", "end", "baseline_start", "baseline_end", "reasoning",
    ],
    "additionalProperties": False,
}


def plan_with_gemini(question: str, *, today: dt.date | None = None) -> Plan:
    """Ask Gemini to map the question onto a module, area and time window.

    The model plans; it never analyses. Its entire output is the schema above —
    a routing decision — so a hallucinated statistic has nowhere to enter the
    response. If planning fails for any reason, the caller falls back to the
    deterministic router rather than surfacing an error.
    """
    import json as _json

    from google import genai  # imported lazily so the package stays optional
    from google.genai import types as genai_types

    today = today or dt.date.today()
    client = genai.Client(api_key=settings.gemini_api_key)

    modules = "\n".join(
        f"- {m['id']}: {m['question']} (needs a baseline window: {m['needs_baseline']})"
        for m in registry.catalog()
    )
    places = "\n".join(f"- {p.id} ({p.kind}): {p.name}" for p in regions.ALL_PLACES)

    system = (
        "You route questions about Morocco to Earth-observation analysis modules on the "
        "AtlasAI platform. You do not answer the question yourself and you never state "
        "figures — another system computes those from satellite imagery.\n\n"
        f"Today is {today.isoformat()}.\n\n"
        f"Available modules:\n{modules}\n\n"
        f"Known places:\n{places}\n\n"
        "Rules:\n"
        "- Pick exactly one module: the one whose question best matches.\n"
        "- Set answerable=false when no module fits, rather than forcing a poor match.\n"
        "- mode='scan' for 'which/where/rank' questions across many places; "
        "'single' when one place is named.\n"
        "- Leave places empty on a scan to use the platform's default set for that module.\n"
        "- Supply a baseline window whenever the question implies change over time, and "
        "always for modules that require one. Prefer the same calendar season in an "
        "earlier year so seasonal cycles do not masquerade as change.\n"
        "- Optical imagery starts in 2017. Never choose a window earlier than that.\n"
        "- Use empty strings for baseline fields when no comparison is needed.\n"
        "- Respond ONLY with a valid JSON object matching the required schema. No prose."
    )

    response = client.models.generate_content(
        model=settings.assistant_model,
        contents=question,
        config=genai_types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=PLAN_SCHEMA,
            temperature=0.0,
        ),
    )

    text = response.text or ""
    raw = _json.loads(text)

    if not raw.get("answerable"):
        raise UnanswerableError(
            raw.get("reasoning")
            or "No module on this platform can answer that question."
        )

    module = raw["module"]
    if module not in registry.REGISTRY:
        raise UnanswerableError(f"The planner chose an unknown module '{module}'.")

    resolved = [regions.get(pid) for pid in raw.get("places", [])]
    resolved = [p for p in resolved if p is not None]
    mode: Mode = "scan" if raw.get("mode") == "scan" else "single"
    if not resolved and mode == "scan":
        resolved = list(_default_scan_set(question.lower(), module))
    if not resolved:
        raise UnanswerableError("The planner did not identify an area to analyse.")

    return Plan(
        module=module,
        mode=mode,
        places=resolved,
        start=_date(raw["start"], today - dt.timedelta(days=90)),
        end=_date(raw["end"], today),
        baseline_start=_date(raw.get("baseline_start"), None),
        baseline_end=_date(raw.get("baseline_end"), None),
        planner=f"gemini ({settings.assistant_model})",
        reasoning=raw.get("reasoning", ""),
    )


def _date(value: object, default: dt.date | None) -> dt.date | None:
    if not value or not isinstance(value, str):
        return default
    try:
        return dt.date.fromisoformat(value.strip())
    except ValueError:
        return default


def make_plan(question: str, *, today: dt.date | None = None) -> Plan:
    """Plan with Gemini when configured, falling back to rules on any failure."""
    if settings.has_llm:
        try:
            return plan_with_gemini(question, today=today)
        except UnanswerableError:
            raise
        except Exception as exc:  # noqa: BLE001 - planning must never hard-fail
            logger.warning("LLM planner unavailable, using rules: %s", exc)
    return route(question, today=today)


# --- execution --------------------------------------------------------------


def execute(plan: Plan, *, resolution_m: float | None = None) -> dict:
    """Run the plan and assemble a grounded answer.

    A scan runs the module over each place concurrently and ranks the results by
    the module's headline metric. Places whose imagery is unusable are reported
    as gaps rather than silently dropped — an absent reservoir in a "which
    reservoirs are shrinking" answer is misleading.
    """
    from app.api.routes.analysis import run_module

    module = registry.get(plan.module)
    if module.needs_baseline and not (plan.baseline_start and plan.baseline_end):
        anchor = plan.start - dt.timedelta(days=365)
        plan.baseline_start = anchor - dt.timedelta(days=45)
        plan.baseline_end = anchor + dt.timedelta(days=45)

    def _run(place: regions.Place) -> dict:
        request = AnalysisRequest(
            bbox=place.as_bbox(),
            start=plan.start,
            end=plan.end,
            baseline_start=plan.baseline_start,
            baseline_end=plan.baseline_end,
            # Scans trade detail for breadth: eight areas at 10 m would be
            # minutes of reads for a ranking that 60 m resolves perfectly well.
            resolution_m=resolution_m or (60 if plan.mode == "scan" else 20),
            max_cloud=40,
            max_items=4 if plan.mode == "scan" else 8,
        )
        try:
            key, result = run_module(plan.module, request)
        except NoDataError as exc:
            return {"place": place.to_dict(), "status": "no_data", "detail": str(exc), "hint": exc.hint}
        except Exception as exc:  # noqa: BLE001 - one bad area must not sink the scan
            logger.exception("assistant: %s failed for %s", plan.module, place.id)
            return {"place": place.to_dict(), "status": "error", "detail": str(exc)}

        return {
            "place": place.to_dict(),
            "status": "ok",
            "summary": result.summary,
            "metrics": result.metrics,
            "confidence": round(result.confidence, 3),
            "caveats": result.caveats,
            "observed_dates": result.observed_dates,
            "result_key": key,
            "layers": [layer.descriptor() for layer in result.layers],
        }

    with ThreadPoolExecutor(max_workers=min(6, max(len(plan.places), 1))) as pool:
        results = list(pool.map(_run, plan.places))

    ranked = _rank(plan.module, [r for r in results if r["status"] == "ok"])
    gaps = [r for r in results if r["status"] != "ok"]

    return {
        "plan": plan.to_dict(),
        "module": {"id": module.id, "title": module.title, "question": module.question},
        "results": ranked,
        "gaps": gaps,
        "answer": _narrate(plan, module, ranked, gaps),
    }


#: Metric each module is ranked by in a scan, and whether higher is "more".
RANK_METRIC: dict[str, tuple[str, bool]] = {
    "drought": ("drought_affected_share_pct", True),
    "vegetation": ("vegetation_health_score", False),
    "water": ("water_change_pct", False),
    "water_quality": ("turbid_share_pct", True),
    "flood": ("high_risk_share_pct", True),
    "wildfire": ("burned_area_km2", True),
    "urban": ("new_construction_km2", True),
    "construction_review": ("total_new_construction_km2", True),
    "agriculture": ("stressed_share_pct", True),
    "coastal": ("eroded_km2", True),
    "solar": ("excellent_km2", True),
    "wind": ("strong_siting_km2", True),
    "roads": ("road_density_km_per_km2", True),
    "terrain": ("mean_elevation_m", True),
}


def _rank(module_id: str, results: list[dict]) -> list[dict]:
    metric, descending = RANK_METRIC.get(module_id, ("", True))
    if not metric:
        return results

    def key(entry: dict) -> float:
        value = entry.get("metrics", {}).get(metric)
        if value is None:
            # Missing metric sorts last in either direction.
            return float("-inf") if descending else float("inf")
        return float(value)

    ranked = sorted(results, key=key, reverse=descending)
    for position, entry in enumerate(ranked, start=1):
        entry["rank"] = position
        entry["ranked_by"] = {"metric": metric, "value": entry.get("metrics", {}).get(metric)}
    return ranked


def _narrate(plan: Plan, module, ranked: list[dict], gaps: list[dict]) -> str:
    """Plain-language answer, built only from computed results."""
    window = f"{plan.start.isoformat()} to {plan.end.isoformat()}"

    if not ranked:
        detail = gaps[0].get("detail", "no usable imagery") if gaps else "no usable imagery"
        return (
            f"I ran {module.title} over {len(plan.places)} area(s) for {window} but got no "
            f"usable result — {detail}. Widening the date range usually fixes this."
        )

    if plan.mode == "single" or len(ranked) == 1:
        first = ranked[0]
        text = f"{first['place']['name']} — {first['summary']}"
        if first["confidence"] < 0.5:
            text += f" Confidence is low ({first['confidence']:.0%}) because of limited clear-sky coverage."
        return text

    metric_name = ranked[0].get("ranked_by", {}).get("metric", "")
    lines = [
        f"Ranked {len(ranked)} areas by {module.title.lower()} over {window}"
        + (f", ordered by {metric_name.replace('_', ' ')}." if metric_name else "."),
    ]
    for entry in ranked[:5]:
        value = entry.get("ranked_by", {}).get("value")
        suffix = f" ({value})" if value is not None else ""
        lines.append(f"{entry['rank']}. {entry['place']['name']}{suffix} — {entry['summary']}")

    if gaps:
        names = ", ".join(g["place"]["name"] for g in gaps[:4])
        lines.append(
            f"No usable imagery for {len(gaps)} area(s) ({names}) — they are absent from this "
            "ranking, not necessarily unaffected."
        )
    return "\n".join(lines)


def suggestions() -> list[str]:
    """Example questions the platform can actually answer, for the UI."""
    return [
        "Which reservoirs are losing water?",
        "Show drought conditions in Souss-Massa",
        "Where did vegetation decrease in Marrakech this year?",
        "Show new urban areas in Casablanca since 2020",
        "Which regions have the highest flood risk?",
        "How turbid is the water at Barrage Al Massira?",
        "Compare Tangier in 2019 vs 2024",
        "What land cover is around Agadir?",
        "Where is solar potential highest in Drâa-Tafilalet?",
        "Is the coastline near Essaouira eroding?",
    ]
