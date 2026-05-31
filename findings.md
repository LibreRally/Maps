# LibreRally.Maps — Research Findings

## pgpointcloud

- **Status**: Active, maintained. Latest release v1.2.5 (Sep 2023), last commit Sep 2025.
- **Support**: PostgreSQL 13-17, PostGIS 3.3. Windows, Linux, macOS, Docker.
- **Architecture**: Stores point groups as `pcPatch` with compression. Spatial indexing via PostGIS bounding boxes.
- **Performance**: 1ms queries on 5B point dataset. BUT data output is SLOW: ~100k pts/sec.
- **Implication**: Viable for storage + spatial queries. NOT viable for streaming raw points to client. Need pre-generation pipeline.
- **Repo**: https://github.com/pgpointcloud/pointcloud (425 stars, 47 open issues)
- **Docker**: pgpointcloud/pointcloud on Docker Hub

## SharpGLTF & 3D Tiles

- **Status**: Active. v1.0.6 (Dec 2025), 571 stars, MIT license.
- **Packages**: SharpGLTF.Core (read/write), SharpGLTF.Runtime (rendering helpers), SharpGLTF.Toolkit (utilities)
- **3D Tiles**: SharpGLTF.Ext.3DTiles supports:
  - CESIUM_primitive_outline
  - EXT_Mesh_Features (feature IDs, property tables)
  - EXT_Instance_Features
  - EXT_Structural_Metadata
  - Read + Write support, validated against Cesium 3D Tiles Validator
- **Stride integration**: Not built-in. vpenades keeps library engine-agnostic. Need custom Stride rendering layer.
- **Key PR**: bertt's PR #213 added comprehensive 3D Tiles API support (Jan 2024, 16K lines, merged)
- **Repo**: https://github.com/vpenades/SharpGLTF

## PDAL (Point Data Abstraction Library)

- **Classification filters**: SMRF (recommended), PMF, CSF for ground; outlier/ELM for noise; neighborclassifier for consensus
- **Clustering**: Euclidean distance, DBSCAN, k-means
- **Integration**: `filters.python` allows embedding Python code in pipeline — key integration point for PointNet2
- **Pipeline**: JSON-based declarative pipelines, streamable mode for memory efficiency
- **Output**: LAS, GDAL rasters, ASCII, and custom writers possible
- **Key insight**: PDAL can be the orchestration backbone for the point cloud processing tier

## TorchSharp

- **Status**: Active (.NET Foundation), 2K+ stars. Latest based on libtorch 2.5.1.
- **Capability**: Tensor operations, model loading (TorchScript), inference
- **Gap**: NOT a full ML framework. No training loop, data loading, augmentation. No pre-built PointNet2.
- **Viable for**: Loading traced PointNet2 model for inference
- **Not viable for**: Training, dataset management, complex ML pipelines
- **Recommendation**: Python microservice for ML; TorchSharp only for client-side inference if needed

## PointNet2

- **Multiple PyTorch implementations**: zhulf0804/Pointnet2.PyTorch, yanx27/Pointnet_Pointnet2_pytorch, sshaoshuai/Pointnet2.PyTorch
- **Classification accuracy**: ~91.8% on ModelNet40 (1024 points)
- **Limitation**: Classification only — doesn't generate meshes. Need separate surface reconstruction (PCL/Poisson)
- **Training data**: Currently trained on ModelNet40/ShapeNet — need custom dataset with OSM labels for map elements

## Unity/Stride 3D Tiles Alternatives

- **GeoTileLoader** (mhama): MIT, pure C# + glTFast, WebGL compatible, early stage. For Unity.
- **Unity3DTiles** (NASA-AMMOS): Older, depends on Newtonsoft + UnityGLTF. For Unity.
- **Neither is Stride-native** — we build custom Stride integration with SharpGLTF

## Melbourne Point Cloud Data

- **Source**: City of Melbourne 2018 Aerial Lidar Survey
- **Format**: LAS 1.2, Point Format 2 (XYZ + RGB + Classification + Return info)
- **Extent**: 215 tiles, 18×18 tile grid (partial), ~500m tiles
- **Geographic**: Melbourne CBD, 144.89°E–144.97°E, 37.77°S–37.85°S
- **Projection**: MGA Zone 55 (GDA94), needs reprojection to WGS84 for 3D Tiles
- **Size**: ~8 GB uncompressed, ~4.3 GB zipped
- **Density**: ~3.6M points per dense tile (~30-95 MB/tile)
- **Classification**: LAS classification field populated — extent of pre-classification TBD

## Key Architecture Insights

1. **pgpointcloud is for storage, not streaming** — output bottleneck (100k pts/sec) means raw point streaming is non-viable
2. **PDAL + Python filter is the pragmatic ML bridge** — avoids building a separate Python microservice for initial processing
3. **SharpGLTF.Ext.3DTiles covers the 3D Tiles spec well** — metadata, feature IDs, property tables all supported
4. **TorchSharp is inference-only** — training requires Python ecosystem; hybrid architecture is necessary
5. **Surface reconstruction is the biggest gap** — PointNet2 classifies but doesn't generate meshes; need PCL or alternative
6. **The feedback loop is novel** — no existing system does user-correction → retraining → tile regeneration; this is greenfield
