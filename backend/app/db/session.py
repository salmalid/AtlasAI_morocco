"""Database session management — optional, and silent when unconfigured.

The platform must start and serve analysis with no database at all, so nothing
here raises at import time. Callers check :func:`is_available` first, or use
:func:`session_scope`, which is a no-op context manager when persistence is off.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.models import Base

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_factory: sessionmaker[Session] | None = None


def is_available() -> bool:
    return settings.has_database


def get_engine() -> Engine | None:
    global _engine, _factory
    if not is_available():
        return None
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,  # survives the connection drops a pooler introduces
            pool_size=5,
            max_overflow=10,
            future=True,
        )
        _factory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


@contextlib.contextmanager
def session_scope() -> Iterator[Session | None]:
    """Transactional scope, or a `None` yield when no database is configured."""
    if get_engine() is None or _factory is None:
        yield None
        return

    session = _factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_schema() -> bool:
    """Enable PostGIS and create tables. Safe to call repeatedly."""
    engine = get_engine()
    if engine is None:
        logger.info("No database configured — persistence disabled.")
        return False

    with engine.begin() as connection:
        # GeoAlchemy2's Geometry columns need the extension present before
        # create_all runs, or the CREATE TABLE fails on an unknown type.
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    Base.metadata.create_all(engine)
    logger.info("Database schema ready.")
    return True
