"""PostGIS schema.

Persistence is **optional**. Without `ATLASAI_DATABASE_URL` the platform runs
entirely from the live satellite archive and an in-process cache, which is the
right default for development and for a single-instance deployment.

The database earns its place at the point where you need things the archive
cannot give you: a record of what was analysed and when, alerts that fire on a
threshold crossing, official administrative boundaries to replace the
approximate envelopes in ``app/services/regions.py``, and time series that
would otherwise cost a fresh imagery read on every request.
"""

from __future__ import annotations

import datetime as dt

from geoalchemy2 import Geometry
from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Region(Base):
    """Official administrative boundary — the authoritative version of a place.

    Load real polygons here (from the Haut-Commissariat au Plan or GADM) and
    zonal statistics become exact rather than bounding-box approximations.
    """

    __tablename__ = "regions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(256))
    name_ar: Mapped[str | None] = mapped_column(String(256))
    level: Mapped[str] = mapped_column(String(32))  # region | province | commune
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("regions.id"))
    population: Mapped[int | None] = mapped_column(Integer)
    area_km2: Mapped[float | None] = mapped_column(Float)
    geom: Mapped[object] = mapped_column(Geometry("MULTIPOLYGON", srid=4326))

    children: Mapped[list["Region"]] = relationship(back_populates="parent")
    parent: Mapped["Region | None"] = relationship(back_populates="children", remote_side=[id])

    __table_args__ = (Index("ix_regions_geom", "geom", postgresql_using="gist"),)


class AreaOfInterest(Base):
    """A user-defined area to monitor over time."""

    __tablename__ = "areas_of_interest"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    description: Mapped[str | None] = mapped_column(Text)
    owner: Mapped[str | None] = mapped_column(String(256), index=True)
    geom: Mapped[object] = mapped_column(Geometry("POLYGON", srid=4326))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    runs: Mapped[list["AnalysisRun"]] = relationship(back_populates="aoi")

    __table_args__ = (Index("ix_aoi_geom", "geom", postgresql_using="gist"),)


class AnalysisRun(Base):
    """Provenance record for one execution of one module.

    This is what makes a published figure defensible six months later: the
    exact module, window, resolution and scene ids behind it. Without it, a
    number in a report is unreproducible the moment the cache expires.
    """

    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    module: Mapped[str] = mapped_column(String(64), index=True)
    aoi_id: Mapped[int | None] = mapped_column(ForeignKey("areas_of_interest.id"))
    bbox: Mapped[object] = mapped_column(Geometry("POLYGON", srid=4326))

    window_start: Mapped[dt.date] = mapped_column(Date, index=True)
    window_end: Mapped[dt.date] = mapped_column(Date, index=True)
    baseline_start: Mapped[dt.date | None] = mapped_column(Date)
    baseline_end: Mapped[dt.date | None] = mapped_column(Date)

    resolution_m: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    metrics: Mapped[dict] = mapped_column(JSON)
    breakdown: Mapped[list] = mapped_column(JSON, default=list)
    caveats: Mapped[list] = mapped_column(JSON, default=list)
    #: STAC item ids, so the exact input scenes can be re-fetched.
    scene_ids: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    aoi: Mapped["AreaOfInterest | None"] = relationship(back_populates="runs")

    __table_args__ = (
        Index("ix_runs_bbox", "bbox", postgresql_using="gist"),
        Index("ix_runs_module_window", "module", "window_end"),
    )


class IndexObservation(Base):
    """One spectral-index value for one area at one date.

    Rebuilding a ten-year NDVI curve costs ~40 imagery reads. Persisting each
    point as it is computed turns the second request for that curve into a
    single indexed query.
    """

    __tablename__ = "index_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    aoi_id: Mapped[int | None] = mapped_column(ForeignKey("areas_of_interest.id"), index=True)
    region_id: Mapped[int | None] = mapped_column(ForeignKey("regions.id"), index=True)
    index_key: Mapped[str] = mapped_column(String(32), index=True)
    observed_on: Mapped[dt.date] = mapped_column(Date, index=True)

    mean: Mapped[float | None] = mapped_column(Float)
    median: Mapped[float | None] = mapped_column(Float)
    p10: Mapped[float | None] = mapped_column(Float)
    p90: Mapped[float | None] = mapped_column(Float)
    valid_fraction: Mapped[float] = mapped_column(Float)
    scene_count: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        UniqueConstraint("aoi_id", "region_id", "index_key", "observed_on", name="uq_observation"),
    )


class Alert(Base):
    """A threshold crossing detected by a monitoring job.

    ``acknowledged`` exists because an alert nobody has triaged is different
    from one that has been reviewed and dismissed, and an operations team needs
    to tell them apart.
    """

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    module: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(16), index=True)  # info | warning | critical
    title: Mapped[str] = mapped_column(String(512))
    detail: Mapped[str] = mapped_column(Text)
    metric: Mapped[str | None] = mapped_column(String(64))
    value: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)

    geom: Mapped[object] = mapped_column(Geometry("POLYGON", srid=4326))
    run_id: Mapped[int | None] = mapped_column(ForeignKey("analysis_runs.id"))
    detected_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    __table_args__ = (Index("ix_alerts_geom", "geom", postgresql_using="gist"),)
