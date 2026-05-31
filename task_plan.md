# LibreRally.Maps — Architecture & Phased Roadmap

## Goal
Open-source geospatial SDK for Stride engine that:
- Ingests lidar point clouds (LAS/LAZ) + KartaView photo sequences
- Classifies map elements via PointNet2 + OSM enrichment
- Generates 3D Tiles (glTF with metadata) for runtime streaming
- Stores point clouds (source of truth) + generated meshes (served to client)
- Supports user corrections feeding back into ML training
- Orchestrated via .NET Aspire

## Architecture Decisions (Proposed)

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| ML runtime | Python microservice (gRPC) behind PDAL | PointNet2/PyTorch3D/PCL are Python/C++ native; TorchSharp can't train |
| Point cloud processing | PDAL (via Docker or CLI) | Production-grade, extensible via `filters.python` |
| Raw point cloud storage | pgpointcloud + PostGIS | Mature, spatial indexes, 1ms queries at 5B points |
| Mesh/tile storage | Local filesystem storage with metadata in PostgreSQL | Meshes are large blobs |
| 3D Tile generation | Custom .NET service using SharpGLTF | Writes valid 3D Tiles glTF with EXT_mesh_features metadata |
| Client rendering | Stride engine + SharpGLTF.Ext.3DTiles | Full .NET stack, engine-agnostic library |
| Orchestration | .NET Aspire | Manages PostgreSQL, PDAL workers, ML service, tile server |
| OSM integration | osm2pgsql → PostGIS spatial joins | Standard tooling, rich query capability |

## Data Inventory

| Source | Format | Size | Status |
|--------|--------|------|--------|
| Melbourne 2018 (CoM) | 215 LAS 1.2 tiles | ~8 GB | Downloaded, ready |
| Victoria OSM | PBF | TBD (Geofabrik) | Not downloaded |
| KartaView sequences | Photo + GPS | N/A | Library started (KartaViewSharp) |

## Phases

### Phase 0: Data Exploration & Feasibility
Status: **in_progress**

- [x] Inventory Melbourne LAS data (215 tiles, ~3.6M pts/tile, Point Format 2 = XYZ+RGB+Classification)
- [ ] Inspect LAS classification field — what's pre-classified?
- [ ] Determine coordinate system (MGA Zone 55 → WGS84 reprojection needed)
- [ ] Verify PDAL pipeline can process Melbourne tiles end-to-end
- [ ] Test pgpointcloud install (Docker) with sample tile
- [ ] Verify SharpGLTF.Ext.3DTiles reads/writes tileset.json + batched b3dm

### Phase 1: Ingestion Pipeline
Status: **pending**

- [ ] PDAL pipeline: LAS → ground classification (SMRF) → clustering → reproject to WGS84
- [ ] pgpointcloud schema: load processed point clouds with spatial indexing
- [ ] osm2pgsql: load Victoria OSM PBF into PostGIS
- [ ] Spatial join: cluster ↔ OSM feature attribution
- [ ] KartaView integration: photo-to-point cloud pipeline (via KartaViewSharp)

### Phase 2: ML Classification Service
Status: **pending**

- [ ] Set up Python microservice (FastAPI/gRPC) with PointNet2 (PyTorch)
- [ ] PDAL `filters.python` → call ML service for per-cluster classification
- [ ] Training data: pre-classified LAS + OSM labels → train PointNet2
- [ ] Output: point clusters tagged with OSM feature type + confidence
- [ ] Surface reconstruction: Poisson/marching cubes → mesh

### Phase 3: 3D Tile Generation
Status: **pending**

- [ ] Mesh → glTF conversion with EXT_mesh_features metadata
- [ ] Tiling: split meshes into 3D Tiles hierarchy (tileset.json)
- [ ] Store generated tiles in object storage
- [ ] Tile metadata in PostgreSQL (bounds, LOD, feature count)
- [ ] Regeneration trigger: new point cloud data in tile area

### Phase 4: Runtime Serving & Client
Status: **pending**

- [ ] .NET Aspire orchestration: PostgreSQL + ML service + tile server
- [ ] Tile server: spatial queries → stream 3D Tiles to client
- [ ] Stride client: SharpGLTF.Ext.3DTiles loader + custom renderer
- [ ] LOD management, frustum culling, tile priority queue
- [ ] Interactivity: per-feature collision (signposts knockable, buildings solid)

### Phase 5: Feedback Loop
Status: **pending**

- [ ] User correction logging (client → server)
- [ ] Correction → training data augmentation
- [ ] Model retraining pipeline (triggered on threshold)
- [ ] Tile regeneration from corrected classifications

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Librarian agents: OpenRouter key limit | 1 | Fell back to direct web searches (websearch_web_search_exa) |
| Metis pre-planning agent timed out | 1 | Synthesized architecture analysis manually |
