-- AtlasAI Morocco — database bootstrap.
-- Runs once, on first start of an empty postgres volume.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- fuzzy place-name search

-- Application tables are created by SQLAlchemy (app/db/session.py:init_schema),
-- so the schema lives in one place rather than drifting between an ORM model
-- and a hand-maintained DDL file.

-- ---------------------------------------------------------------------------
-- Loading official administrative boundaries
-- ---------------------------------------------------------------------------
-- The `regions` table in app/db/models.py expects real polygons. The bounding
-- boxes in app/services/regions.py are approximate envelopes for place lookup
-- only — never use them for area statistics you intend to publish.
--
--   ogr2ogr -f PostgreSQL \
--     PG:"host=localhost dbname=atlasai user=atlasai password=atlasai" \
--     morocco_regions.geojson \
--     -nln regions_import -nlt MULTIPOLYGON -t_srs EPSG:4326
--
-- Then map the import into `regions`, adjusting column names to your source:
--
--   INSERT INTO regions (code, name, level, geom, area_km2)
--   SELECT shapeiso, shapename, 'region', ST_Multi(wkb_geometry),
--          ST_Area(wkb_geometry::geography) / 1e6
--   FROM regions_import
--   ON CONFLICT (code) DO UPDATE
--     SET geom = EXCLUDED.geom, name = EXCLUDED.name;
--
-- Sources: geoBoundaries (https://www.geoboundaries.org, ADM1/ADM2, CC-BY),
-- or the Haut-Commissariat au Plan for the authoritative national dataset.
