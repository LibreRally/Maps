"""LibreRally.Maps ML Service — Open3D/Open3D-ML semantic pipeline."""
import json
import os
import queue
import threading
from typing import Any, List

import numpy as np
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from classifier import FEATURE_NAMES as _ASPRS_NAME
from open3d_classifier import ensure_model_checkpoint, reconstruct_with_open3d
from open3d_pipeline import classify_semantic_labels_with_source, get_runtime_info
from pipeline import process_input_file

app = FastAPI(title="LibreRally.Maps ML Service", version="0.2.0")
CONTAINER_DATA_ROOT = os.environ.get("CONTAINER_DATA_ROOT", "/data")
HOST_DATA_ROOT_RELATIVE = os.environ.get("HOST_DATA_ROOT_RELATIVE", "data").replace("\\", "/").strip("/")


@app.on_event("startup")
async def warm_open3d_ml_runtime():
    runtime = get_runtime_info()
    if runtime["open3d_ml_available"]:
        ensure_model_checkpoint()


# ── Request/Response Models ───────────────────────────────────────

class PointCloudRequest(BaseModel):
    """Input point cloud for classification."""
    points: List[List[float]]  # Nx3 (x, y, z) or Nx6 (x, y, z, r, g, b)
    crs: str = "EPSG:28355"    # Source coordinate system

class ClassificationResult(BaseModel):
    """Classification output per point."""
    labels: List[int]           # ASPRS classification per point
    confidence: List[float]     # Confidence score per point
    feature_types: List[str]    # e.g. "building", "ground", "vegetation"
    source: str = "osm-guided"  # Classification method used

class ReconstructRequest(BaseModel):
    """Input for surface reconstruction."""
    points: List[List[float]]   # Nx3 classified points
    labels: List[int]           # Per-point labels
    method: str = "alpha_shape" # "alpha_shape", "poisson", "convex_hull"

class MeshResult(BaseModel):
    """Reconstructed mesh output."""
    vertices: List[List[float]]  # Nx3
    faces: List[List[int]]       # Mx3 triangle indices
    vertex_count: int
    face_count: int

class ProcessRequest(BaseModel):
    """Input for full point-bucket processing inside the ML container."""
    input_path: str
    max_points: int = 0
    crs: str = "EPSG:28355"


# ── Health ────────────────────────────────────────────────────────

@app.get("/health")
async def health(response: Response):
    runtime = get_runtime_info()
    service_status = "healthy" if runtime["open3d_available"] else "unhealthy"
    if service_status != "healthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": service_status,
        "service": "librerallymaps-ml",
        "open3d": runtime["open3d_available"],
        "open3dMl": runtime["open3d_ml_available"],
        "modelPresent": runtime["model_present"],
    }

@app.get("/info")
async def info():
    runtime = get_runtime_info()
    return {"service": "librerallymaps-ml", "version": "0.2.0", **runtime}


# ── Classify ──────────────────────────────────────────────────────

@app.post("/classify", response_model=ClassificationResult)
async def classify(pc: PointCloudRequest):
    """Classify point cloud using the Open3D/Open3D-ML production path."""
    pts = np.array(pc.points, dtype=np.float64)

    try:
        colors = pts[:, 3:6] if pts.shape[1] >= 6 else None
        xyz = pts[:, :3]
        labels, confidence, source = classify_semantic_labels_with_source(xyz, colors)
        return {
            "labels": labels.tolist(),
            "confidence": confidence.tolist(),
            "feature_types": [_ASPRS_NAME.get(int(label), f"class_{int(label)}") for label in labels],
            "source": source,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Reconstruct ───────────────────────────────────────────────────

@app.post("/reconstruct", response_model=MeshResult)
async def reconstruct(req: ReconstructRequest):
    """Generate a mesh using Open3D reconstruction."""
    try:
        pts = np.array(req.points, dtype=np.float64)
        verts, faces = reconstruct_with_open3d(pts, method=req.method)
        if verts is None or faces is None or len(faces) == 0:
            raise RuntimeError("Open3D reconstruction did not produce a valid mesh.")

        return {
            "vertices": verts.tolist(),
            "faces": faces.tolist(),
            "vertex_count": int(len(verts)),
            "face_count": int(len(faces)),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/process")
async def process(req: ProcessRequest) -> dict[str, Any]:
    """Run the full Open3D pipeline inside the ML container."""
    try:
        container_input_path = _resolve_container_path(req.input_path)
        result = process_input_file(container_input_path, max_points=req.max_points, input_crs=req.crs)
        return _normalize_result_paths(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/process/stream")
async def process_stream(req: ProcessRequest) -> StreamingResponse:
    """Run the full Open3D pipeline and stream status/result events as NDJSON."""
    container_input_path = _resolve_container_path(req.input_path)
    event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def status_callback(payload: dict[str, Any]) -> None:
        event_queue.put({"type": "status", **payload})

    def worker() -> None:
        try:
            result = process_input_file(
                container_input_path,
                max_points=req.max_points,
                input_crs=req.crs,
                status_callback=status_callback,
            )
            event_queue.put({"type": "result", "result": _normalize_result_paths(result)})
        except Exception as exc:
            event_queue.put({"type": "error", "detail": str(exc)})
        finally:
            event_queue.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            payload = event_queue.get()
            if payload is None:
                break

            yield json.dumps(payload, separators=(",", ":")) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def _resolve_container_path(input_path: str) -> str:
    normalized = input_path.replace("\\", "/").strip()
    if os.path.isabs(normalized):
        return normalized

    prefix = f"{HOST_DATA_ROOT_RELATIVE}/" if HOST_DATA_ROOT_RELATIVE else ""
    relative_path = normalized[len(prefix):] if prefix and normalized.startswith(prefix) else normalized
    return os.path.join(CONTAINER_DATA_ROOT, *[part for part in relative_path.split("/") if part])


def _normalize_result_paths(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    normalized["glbPath"] = _to_host_relative_path(normalized["glbPath"])
    if normalized.get("segmentDataPath"):
        normalized["segmentDataPath"] = _to_host_relative_path(normalized["segmentDataPath"])
    if normalized.get("segmentMetadataPath"):
        normalized["segmentMetadataPath"] = _to_host_relative_path(normalized["segmentMetadataPath"])
    return normalized


def _to_host_relative_path(path: str) -> str:
    absolute_path = os.path.abspath(path)
    container_root = os.path.abspath(CONTAINER_DATA_ROOT)
    if absolute_path == container_root:
        return HOST_DATA_ROOT_RELATIVE

    container_prefix = container_root + os.sep
    if absolute_path.startswith(container_prefix):
        suffix = os.path.relpath(absolute_path, container_root).replace("\\", "/")
        return f"{HOST_DATA_ROOT_RELATIVE}/{suffix}" if HOST_DATA_ROOT_RELATIVE else suffix

    return path.replace("\\", "/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
