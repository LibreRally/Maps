-- LibreRally.Maps — Database Initialization
-- Creates extensions, schemas, and core tables

-- ── Extensions ──────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS postgis CASCADE;
CREATE EXTENSION IF NOT EXISTS postgis_topology CASCADE;
CREATE EXTENSION IF NOT EXISTS pointcloud CASCADE;
CREATE EXTENSION IF NOT EXISTS pointcloud_postgis CASCADE;

-- ── Schemas ─────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS pointcloud;
CREATE SCHEMA IF NOT EXISTS osm;
CREATE SCHEMA IF NOT EXISTS tiles;

-- ── Point Cloud Storage ─────────────────────────────────────────
-- Raw ingested point cloud tiles (LAS → pgpointcloud)
CREATE TABLE IF NOT EXISTS pointcloud.tiles (
    id              SERIAL PRIMARY KEY,
    tile_x          INTEGER NOT NULL,
    tile_y          INTEGER NOT NULL,
    source          TEXT NOT NULL,           -- e.g. 'melbourne_2018', 'kartaview'
    crs             INTEGER NOT NULL,        -- EPSG code of source data
    bounds          geometry(Polygon, 4326), -- WGS84 bounds
    point_count     BIGINT,
    file_size_bytes BIGINT,
    pc_patch        pcpatch,                -- pgpointcloud compressed patch
    metadata        JSONB,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(tile_x, tile_y, source)
);

-- Spatial index on bounds
CREATE INDEX IF NOT EXISTS idx_pointcloud_tiles_bounds 
    ON pointcloud.tiles USING GIST (bounds);

-- Import-first ingest records
CREATE TABLE IF NOT EXISTS pointcloud.imports (
    id                           UUID PRIMARY KEY,
    original_file_name           TEXT NOT NULL,
    stored_file_path             TEXT,
    source                       TEXT NOT NULL,
    crs                          INTEGER NOT NULL,
    point_signature              TEXT,
    total_point_count            BIGINT,
    unique_point_count           BIGINT,
    new_point_count              BIGINT,
    duplicate_point_count        BIGINT,
    duplicate_within_import_count BIGINT,
    file_size_bytes              BIGINT,
    status                       TEXT NOT NULL DEFAULT 'pending',
    rejection_reason             TEXT,
    imported_at                  TIMESTAMPTZ DEFAULT NOW(),
    deleted_at                   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_pointcloud_imports_status
    ON pointcloud.imports (status, imported_at DESC);

ALTER TABLE IF EXISTS pointcloud.imports
    ADD COLUMN IF NOT EXISTS point_signature TEXT;

ALTER TABLE IF EXISTS pointcloud.imports
    ADD COLUMN IF NOT EXISTS alignment_yaw_degrees DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS alignment_offset_x DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS alignment_offset_y DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS alignment_source TEXT,
    ADD COLUMN IF NOT EXISTS alignment_updated_at TIMESTAMPTZ;

UPDATE pointcloud.imports
SET alignment_yaw_degrees = -90.0,
    alignment_offset_x = COALESCE(alignment_offset_x, 0.0),
    alignment_offset_y = COALESCE(alignment_offset_y, 0.0),
    alignment_source = COALESCE(alignment_source, 'scaniverse_default'),
    alignment_updated_at = COALESCE(alignment_updated_at, NOW())
WHERE alignment_yaw_degrees IS NULL
  AND original_file_name ILIKE '%scaniverse%';

CREATE INDEX IF NOT EXISTS idx_pointcloud_imports_signature
    ON pointcloud.imports (point_signature, status, imported_at);

CREATE TABLE IF NOT EXISTS pointcloud.points (
    id              BIGSERIAL PRIMARY KEY,
    x               DOUBLE PRECISION NOT NULL,
    y               DOUBLE PRECISION NOT NULL,
    z               DOUBLE PRECISION NOT NULL,
    red             INTEGER,
    green           INTEGER,
    blue            INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (x, y, z)
);

CREATE TABLE IF NOT EXISTS pointcloud.import_points (
    import_id       UUID NOT NULL REFERENCES pointcloud.imports(id) ON DELETE CASCADE,
    point_id        BIGINT NOT NULL REFERENCES pointcloud.points(id) ON DELETE CASCADE,
    PRIMARY KEY (import_id, point_id)
);

CREATE INDEX IF NOT EXISTS idx_pointcloud_import_points_point
    ON pointcloud.import_points (point_id);

-- ── OSM Data (loaded via osm2pgsql) ─────────────────────────────
-- Standard osm2pgsql output tables go in 'osm' schema
-- These are created by osm2pgsql, not here

-- ── 3D Tile Catalog ─────────────────────────────────────────────
-- Generated 3D Tiles metadata
CREATE TABLE IF NOT EXISTS tiles.catalog (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tileset_path    TEXT NOT NULL,           -- path to tileset.json in object storage
    tile_x          INTEGER,
    tile_y          INTEGER,
    zoom_level      INTEGER,
    bounds          geometry(Polygon, 4326),
    feature_types   TEXT[],                  -- e.g. {'building', 'road', 'vegetation'}
    mesh_count      INTEGER,
    vertex_count    BIGINT,
    file_size_bytes BIGINT,
    lod_min         REAL,
    lod_max         REAL,
    segment_count   INTEGER,
    segments_path   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    -- Source point cloud tiles used
    source_tile_ids INTEGER[]
);

CREATE INDEX IF NOT EXISTS idx_tiles_catalog_bounds 
    ON tiles.catalog USING GIST (bounds);

CREATE INDEX IF NOT EXISTS idx_tiles_catalog_zoom 
    ON tiles.catalog (zoom_level, tile_x, tile_y);

ALTER TABLE IF EXISTS tiles.catalog
    ADD COLUMN IF NOT EXISTS segment_count INTEGER,
    ADD COLUMN IF NOT EXISTS segments_path TEXT,
    ADD COLUMN IF NOT EXISTS source_import_ids UUID[],
    ADD COLUMN IF NOT EXISTS alignment_yaw_degrees DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS alignment_offset_x DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS alignment_offset_y DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS alignment_source TEXT,
    ADD COLUMN IF NOT EXISTS alignment_updated_at TIMESTAMPTZ;

-- ── Segment Catalog ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tiles.segments (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tile_id           UUID NOT NULL REFERENCES tiles.catalog(id) ON DELETE CASCADE,
    source_tile_id    INTEGER REFERENCES pointcloud.tiles(id),
    source_import_ids UUID[],
    local_segment_id  INTEGER NOT NULL,
    predicted_label   TEXT NOT NULL,
    reviewed_label    TEXT,
    point_count       INTEGER NOT NULL,
    confidence        REAL,
    osm_feature_type  TEXT,
    osm_name          TEXT,
    osm_identifier    BIGINT,
    osm_match_status  TEXT NOT NULL DEFAULT 'unmatched',
    osm_match_score   REAL,
    preview_path      TEXT,
    artifact_path     TEXT,
    bounds_min_x      DOUBLE PRECISION,
    bounds_min_y      DOUBLE PRECISION,
    bounds_min_z      DOUBLE PRECISION,
    bounds_max_x      DOUBLE PRECISION,
    bounds_max_y      DOUBLE PRECISION,
    bounds_max_z      DOUBLE PRECISION,
    centroid_x        DOUBLE PRECISION,
    centroid_y        DOUBLE PRECISION,
    centroid_z        DOUBLE PRECISION,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tile_id, local_segment_id)
);

CREATE INDEX IF NOT EXISTS idx_tiles_segments_tile
    ON tiles.segments (tile_id, local_segment_id);

CREATE INDEX IF NOT EXISTS idx_tiles_segments_review
    ON tiles.segments (osm_match_status, predicted_label);

ALTER TABLE IF EXISTS tiles.segments
    ADD COLUMN IF NOT EXISTS source_import_ids UUID[],
    ADD COLUMN IF NOT EXISTS geometry_source TEXT DEFAULT 'scan',
    ADD COLUMN IF NOT EXISTS has_osm_fill BOOLEAN DEFAULT FALSE;

UPDATE tiles.segments
SET geometry_source = COALESCE(geometry_source, 'scan'),
    has_osm_fill = COALESCE(has_osm_fill, FALSE)
WHERE geometry_source IS NULL OR has_osm_fill IS NULL;

ALTER TABLE IF EXISTS tiles.segments
    ALTER COLUMN geometry_source SET DEFAULT 'scan',
    ALTER COLUMN has_osm_fill SET DEFAULT FALSE;

-- ── Segment Reviews / Training Feedback ────────────────────────────
CREATE TABLE IF NOT EXISTS tiles.segment_reviews (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    segment_id        UUID NOT NULL REFERENCES tiles.segments(id) ON DELETE CASCADE,
    correction_type   TEXT NOT NULL,         -- 'reclassify', 'split', 'merge'
    previous_label    TEXT,
    requested_label   TEXT,
    related_segment_ids TEXT[],
    notes             TEXT,
    submitted_by      TEXT,
    submitted_at      TIMESTAMPTZ DEFAULT NOW(),
    reviewed          BOOLEAN DEFAULT FALSE NOT NULL,
    approved          BOOLEAN DEFAULT FALSE NOT NULL,
    reviewed_at       TIMESTAMPTZ,
    reviewer_notes    TEXT,
    exported_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_tiles_segment_reviews_segment
    ON tiles.segment_reviews (segment_id, reviewed, approved);

-- ── Feedback / Corrections ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS tiles.corrections (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tile_id         UUID REFERENCES tiles.catalog(id),
    feature_index   INTEGER,                -- index of feature in the tile's glTF
    correction_type TEXT NOT NULL,           -- 'reclassify', 'reposition', 'remove', 'add'
    old_label       TEXT,
    new_label       TEXT,
    geometry_correction JSONB,              -- {x, y, z} delta or new geometry
    submitted_by    TEXT,
    submitted_at    TIMESTAMPTZ DEFAULT NOW(),
    reviewed        BOOLEAN DEFAULT FALSE,
    approved        BOOLEAN DEFAULT FALSE,
    reviewed_at     TIMESTAMPTZ,
    reviewer_notes  TEXT
);

ALTER TABLE IF EXISTS tiles.corrections
    ADD COLUMN IF NOT EXISTS approved BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS reviewer_notes TEXT;

ALTER TABLE IF EXISTS tiles.corrections
    ALTER COLUMN reviewed SET DEFAULT FALSE,
    ALTER COLUMN approved SET DEFAULT FALSE;

UPDATE tiles.corrections
SET reviewed = COALESCE(reviewed, FALSE),
    approved = COALESCE(approved, FALSE)
WHERE reviewed IS NULL OR approved IS NULL;

ALTER TABLE IF EXISTS tiles.corrections
    ALTER COLUMN reviewed SET NOT NULL,
    ALTER COLUMN approved SET NOT NULL;

-- ── Version Tracking ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pointcloud.versions (
    id              SERIAL PRIMARY KEY,
    version_tag     TEXT NOT NULL,
    description     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Track which point cloud tiles are in which version
CREATE TABLE IF NOT EXISTS pointcloud.version_tiles (
    version_id      INTEGER REFERENCES pointcloud.versions(id),
    tile_id         INTEGER REFERENCES pointcloud.tiles(id),
    PRIMARY KEY (version_id, tile_id)
);
