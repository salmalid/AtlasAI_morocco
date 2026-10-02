"""AI assistant endpoint — ask a question, get an answer computed from imagery."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.ai import assistant
from app.api.schemas import AssistantQuery
from app.core.config import settings
from app.core.geo import BBox
from app.eo.catalog import NoDataError
from app.services.regions import Place

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post("", summary="Ask a question in natural language")
def ask(query: AssistantQuery) -> dict:
    try:
        plan = assistant.make_plan(query.question)
    except assistant.UnanswerableError as exc:
        # 200, not 4xx: "I can't answer that" is a valid answer, and the client
        # should render it as one rather than as a failed request.
        return {
            "answer": str(exc),
            "answerable": False,
            "suggestions": assistant.suggestions(),
        }

    # An explicit bbox from the map overrides whatever place the planner picked.
    if query.bbox:
        try:
            plan.bbox = BBox.parse(query.bbox)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid bbox: {exc}") from exc
        lon, lat = plan.bbox.centroid
        plan.places = [
            Place(
                id="map-selection", name="Selected area", kind="city",
                lon=lon, lat=lat, bbox=plan.bbox.as_tuple(),
            )
        ]
        plan.mode = "single"

    if query.start:
        plan.start = query.start
    if query.end:
        plan.end = query.end
    if plan.end < plan.start:
        raise HTTPException(status_code=422, detail="`end` must not precede `start`.")

    try:
        payload = assistant.execute(plan)
    except NoDataError as exc:
        return {
            "answer": f"{exc} {exc.hint}".strip(),
            "answerable": True,
            "plan": plan.to_dict(),
            "results": [],
        }

    payload["answerable"] = True
    return payload


@router.get("/suggestions", summary="Example questions the platform can answer")
def suggestions() -> dict:
    return {
        "suggestions": assistant.suggestions(),
        "planner": "claude" if settings.has_llm else "rules",
        "note": (
            "Answers are computed from satellite imagery for each request. The assistant "
            "routes the question to an analysis module; it never answers from memory."
            if settings.has_llm
            else "Set ANTHROPIC_API_KEY to enable LLM-based question planning. The built-in "
            "keyword router handles the phrasings listed above."
        ),
    }
