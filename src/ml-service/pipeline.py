"""
LibreRally.Maps — segment-first point cloud classification + mesh generation pipeline.

The pipeline now:
1. Loads the LAS/LAZ tile
2. Predicts per-point classes
3. Builds stable geometry-first segments
4. Reconstructs one mesh per segment
5. Emits segment artifacts for review / training export
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Callable

import laspy
import numpy as np
from pyproj import Transformer

from classifier import (
    CLASS_BUILDING,
    CLASS_GROUND,
    CLASS_LOW_VEG,
    CLASS_MED_VEG,
    CLASS_ROAD,
    FEATURE_NAMES as CLASS_NAMES,
)
from open3d_pipeline import (
    build_segment_ids,
    classify_segment_geometry,
    classify_semantic_labels,
    estimate_ground_plane,
    reconstruct_segment_meshes_open3d,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LAS_DIR = os.environ.get("LAS_DIR", os.path.join(REPO_ROOT, "data", "raw"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(REPO_ROOT, "data", "tiles"))
DEFAULT_ZOOM = 14
MAX_RECONSTRUCTION_POINTS = int(os.environ.get("PIPELINE_MAX_RECONSTRUCTION_POINTS", "0"))
GLB_PROCESS_FACE_LIMIT = int(os.environ.get("PIPELINE_GLB_PROCESS_FACE_LIMIT", "250000"))
GLB_NORMAL_FIX_FACE_LIMIT = int(os.environ.get("PIPELINE_GLB_NORMAL_FIX_FACE_LIMIT", "75000"))

FEATURE_COLORS = {
    "building": [200, 180, 160],
    "ground": [90, 140, 80],
    "high_vegetation": [40, 120, 40],
    "medium_vegetation": [60, 150, 60],
    "low_vegetation": [80, 170, 90],
    "road": [80, 80, 90],
    "water": [40, 80, 180],
    "wire": [180, 180, 100],
    "tower": [190, 150, 110],
    "unclassified": [170, 170, 170],
}


def emit_status(status, message, completed_steps=None, total_steps=None):
    payload = {"status": status, "message": message}
    if completed_steps is not None:
        payload["completedSteps"] = int(completed_steps)
    if total_steps is not None:
        payload["totalSteps"] = int(total_steps)
    print(f"PIPELINE_STATUS {json.dumps(payload)}", flush=True)


def emit_result(payload):
    print(f"PIPELINE_RESULT {json.dumps(payload)}", flush=True)


def publish_status(status, message, completed_steps=None, total_steps=None, status_callback=None):
    payload = {"status": status, "message": message}
    if completed_steps is not None:
        payload["completedSteps"] = int(completed_steps)
    if total_steps is not None:
        payload["totalSteps"] = int(total_steps)

    if status_callback is not None:
        status_callback(payload)
        return

    emit_status(status, message, completed_steps=completed_steps, total_steps=total_steps)


def latlon_to_tile(lat, lon, zoom):
    """Convert lat/lon to 3D Tiles quadtree (x, y, zoom)."""
    import math

    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y, zoom


def classify_points_locally(xyz, rgb=None):
    """Classify in-process using the Open3D/Open3D-ML production path."""
    labels, confidence = classify_semantic_labels(xyz, rgb)
    return np.asarray(labels, dtype=np.int32), np.asarray(confidence, dtype=np.float32)


def classify_points(xyz, rgb=None):
    """Classify point cloud in-process inside the ML container."""
    return classify_points_locally(xyz, rgb)


def choose_segment_class(point_labels):
    usable = [int(label) for label in point_labels if int(label) > 1]
    if usable:
        return Counter(usable).most_common(1)[0][0]

    usable = [int(label) for label in point_labels]
    return Counter(usable).most_common(1)[0][0] if usable else 1


def compute_segment_geometry(pts):
    pts64 = np.asarray(pts, dtype=np.float64)
    if pts64.ndim != 2 or pts64.shape[0] == 0:
        raise ValueError("Segment geometry requires at least one 3D point.")

    bbox_min = np.min(pts64, axis=0)
    bbox_max = np.max(pts64, axis=0)
    centroid = bbox_min + np.mean(pts64 - bbox_min, axis=0, dtype=np.float64)
    centroid = np.clip(centroid, bbox_min, bbox_max)

    if not (
        np.all(np.isfinite(bbox_min))
        and np.all(np.isfinite(bbox_max))
        and np.all(np.isfinite(centroid))
    ):
        raise ValueError("Segment geometry contains non-finite coordinates.")

    return bbox_min, bbox_max, centroid


def refine_segment_class(asprs_class, pts, plane_model=None):
    if len(pts) < 25:
        return int(asprs_class)

    bbox_min, bbox_max, _ = compute_segment_geometry(pts)
    size = bbox_max - bbox_min
    width, length = sorted(size[:2])
    footprint = float(size[0] * size[1])
    height = float(size[2])
    aspect_ratio = float(width / max(length, 1e-6))

    if asprs_class == CLASS_ROAD:
        geometry_class, _ = classify_segment_geometry(pts, plane_model)
        if geometry_class == CLASS_GROUND:
            return CLASS_GROUND
        return int(asprs_class)

    if asprs_class != CLASS_BUILDING:
        return int(asprs_class)

    if len(pts) < 200 or ((footprint < 12.0 or aspect_ratio <= 0.12) and len(pts) < 800):
        geometry_class, _ = classify_segment_geometry(pts, plane_model)
        if geometry_class in (CLASS_GROUND, CLASS_ROAD, CLASS_LOW_VEG, CLASS_MED_VEG):
            return int(geometry_class)
        return CLASS_LOW_VEG

    if footprint < 120.0 or height > 3.5:
        return int(asprs_class)

    geometry_class, _ = classify_segment_geometry(pts, plane_model)
    if geometry_class in (CLASS_GROUND, CLASS_ROAD, CLASS_LOW_VEG, CLASS_MED_VEG):
        return int(geometry_class)

    return int(asprs_class)


def build_segment_summary(local_id, pts, point_labels, confidence, plane_model=None):
    bbox_min, bbox_max, centroid = compute_segment_geometry(pts)
    asprs_class = choose_segment_class(point_labels)
    asprs_class = refine_segment_class(asprs_class, pts, plane_model)
    feature_type = CLASS_NAMES.get(asprs_class, f"class_{asprs_class}")
    return {
        "localId": int(local_id),
        "asprsClass": int(asprs_class),
        "featureType": feature_type,
        "pointCount": int(len(pts)),
        "confidence": float(np.mean(confidence, dtype=np.float64)) if len(confidence) else 0.0,
        "boundsMin": [float(value) for value in bbox_min],
        "boundsMax": [float(value) for value in bbox_max],
        "centroid": [float(value) for value in centroid],
        "vertexCount": 0,
        "faceCount": 0,
    }


def build_segments(xyz, labels, confidence):
    segment_ids = build_segment_ids(xyz, labels)
    segments = []
    plane_model = estimate_ground_plane(xyz)

    for local_id in sorted(int(value) for value in np.unique(segment_ids) if int(value) >= 0):
        member_indices = np.where(segment_ids == local_id)[0]
        if len(member_indices) < 10:
            continue

        segments.append(
            build_segment_summary(
                local_id,
                xyz[member_indices],
                labels[member_indices],
                confidence[member_indices],
                plane_model,
            )
        )

    return segment_ids, segments


def reconstruct_segment_meshes(xyz, rgb, segment_ids, segments):
    return reconstruct_segment_meshes_open3d(
        xyz,
        normalize_rgb_for_display(rgb),
        segment_ids,
        segments,
        max_reconstruction_points=MAX_RECONSTRUCTION_POINTS,
    )


def normalize_rgb_for_display(rgb):
    if rgb is None:
        return None

    colors = np.asarray(rgb, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[0] == 0 or colors.shape[1] < 3:
        return None

    max_color = float(np.max(colors[:, :3]))
    if max_color <= 1.0:
        scaled = colors[:, :3] * 255.0
    elif max_color <= 255.0:
        scaled = colors[:, :3]
    else:
        scaled = colors[:, :3] * (255.0 / max_color)

    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


def generate_glb(meshes, output_path):
    """Generate a GLB file that preserves one geometry per segment.

    Applies per-vertex colours from the point cloud and bakes PBR textures
    when available. Falls back to flat feature colours for segments without
    colour data.
    """
    import trimesh

    if not meshes:
        return False, None

    export_meshes = [mesh for mesh in meshes if mesh.get("name") != "noise"]
    if not export_meshes:
        return False, None

    all_bounds = []
    for mesh in export_meshes:
        verts = np.asarray(mesh["vertices"], dtype=np.float32)
        if len(verts) >= 3:
            all_bounds.append(np.array([verts.min(axis=0), verts.max(axis=0)], dtype=np.float32))

    if not all_bounds:
        return False, None

    scene_origin = np.mean(np.stack(all_bounds, axis=0), axis=(0, 1), dtype=np.float64)
    scene = trimesh.Scene()
    for mesh in export_meshes:
        verts = np.array(mesh["vertices"], dtype=np.float32) - scene_origin.astype(np.float32)
        faces = np.array(mesh["faces"], dtype=np.int32)
        if len(verts) < 3 or len(faces) < 1:
            continue

        try:
            has_texture = mesh.get("baseColorTexture") is not None
            has_vertex_colors = mesh.get("vertexColors") is not None
            should_process = len(faces) <= GLB_PROCESS_FACE_LIMIT and not has_texture and not has_vertex_colors
            should_fix_normals = len(faces) <= GLB_NORMAL_FIX_FACE_LIMIT

            tm = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                process=should_process,
                validate=should_process,
            )
            if should_fix_normals:
                trimesh.repair.fix_normals(tm, multibody=True)

            if has_texture:
                # ── PBR material with baked base colour + normal textures ──
                import io

                from PIL import Image

                base_img = Image.open(io.BytesIO(mesh["baseColorTexture"]))
                normal_img = None
                if mesh.get("normalTexture") is not None:
                    normal_img = Image.open(io.BytesIO(mesh["normalTexture"]))

                pbr_material = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=base_img,
                    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                    metallicFactor=0.0,
                    roughnessFactor=0.9,
                )
                if normal_img is not None:
                    pbr_material.normalTexture = normal_img

                uvs = np.asarray(mesh["uvs"], dtype=np.float32)
                tm.visual = trimesh.visual.texture.TextureVisuals(
                    uv=uvs,
                    material=pbr_material,
                )
            elif has_vertex_colors:
                # ── Per-vertex colours from point cloud ──
                vertex_colors = np.asarray(mesh["vertexColors"], dtype=np.uint8)
                if len(vertex_colors) == len(tm.vertices):
                    tm.visual.vertex_colors = vertex_colors
                else:
                    # Vertex count mismatch (unlikely without processing) — fall back to flat
                    color = mesh.get("displayColor") or FEATURE_COLORS.get(
                        mesh["name"], FEATURE_COLORS["unclassified"]
                    )
                    tm.visual.vertex_colors = np.tile(
                        np.asarray(color, dtype=np.uint8), (len(tm.vertices), 1)
                    )
            else:
                # ── Flat feature colour (fallback) ──
                color = mesh.get("displayColor") or FEATURE_COLORS.get(
                    mesh["name"], FEATURE_COLORS["unclassified"]
                )
                tm.visual.vertex_colors = np.tile(
                    np.asarray(color, dtype=np.uint8), (len(tm.vertices), 1)
                )

            scene.add_geometry(
                tm,
                node_name=f"segment-{mesh['localId']:04d}-{mesh['name']}",
            )
        except Exception as exc:
            print(f"Skipping segment {mesh['localId']}: {exc}", file=sys.stderr, flush=True)

    if not scene.geometry:
        return False, None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scene.export(output_path)
    return True, scene_origin


def save_segment_artifacts(output_dir, file_name, xyz, rgb, labels, segment_ids, segments, point_ids=None, scene_origin=None):
    colors = rgb.astype(np.float32) if rgb is not None else np.empty((0, 3), dtype=np.float32)
    segment_data_path = os.path.join(output_dir, "segments.npz")
    archive_payload = {
        "points": np.asarray(xyz, dtype=np.float64),
        "colors": colors,
        "labels": labels.astype(np.int32),
        "segment_ids": segment_ids.astype(np.int32),
    }
    if point_ids is not None:
        archive_payload["point_ids"] = point_ids.astype(np.int64)
    np.savez_compressed(segment_data_path, **archive_payload)

    metadata = {
        "fileName": file_name,
        "segmentCount": len(segments),
        "segments": segments,
    }
    if scene_origin is not None:
        metadata["sceneOrigin"] = [float(value) for value in np.asarray(scene_origin, dtype=np.float64)]
    segment_metadata_path = os.path.join(output_dir, "segments.json")
    with open(segment_metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)

    return segment_data_path, segment_metadata_path


def load_input_points(input_path, max_points=None):
    if input_path.lower().endswith(".csv"):
        rows = np.loadtxt(input_path, delimiter=",", dtype=np.float64)
        rows = np.atleast_2d(rows)
        total_points = len(rows)

        if max_points and max_points > 0 and total_points > max_points:
            step = max(total_points // max_points, 1)
            idx = slice(0, total_points, step)
        else:
            idx = slice(None)

        sampled = rows[idx]
        point_ids = sampled[:, 0].astype(np.int64) if sampled.shape[1] >= 7 else None
        xyz = sampled[:, 1:4].astype(np.float64) if sampled.shape[1] >= 7 else sampled[:, :3].astype(np.float64)

        if sampled.shape[1] >= 7:
            rgb_values = sampled[:, 4:7]
        elif sampled.shape[1] >= 6:
            rgb_values = sampled[:, 3:6]
        else:
            rgb_values = None

        if rgb_values is not None and np.any(rgb_values >= 0):
            rgb = rgb_values.astype(np.float32)
        else:
            rgb = None

        return point_ids, xyz, rgb, total_points

    las = laspy.read(input_path)
    total_points = len(las.x)

    if max_points and max_points > 0 and total_points > max_points:
        step = max(total_points // max_points, 1)
        idx = slice(0, total_points, step)
    else:
        idx = slice(None)

    xyz = np.vstack((
        np.asarray(las.x[idx], dtype=np.float64),
        np.asarray(las.y[idx], dtype=np.float64),
        np.asarray(las.z[idx], dtype=np.float64),
    )).T
    if all(hasattr(las, channel) for channel in ("red", "green", "blue")):
        rgb = np.vstack((np.array(las.red[idx]), np.array(las.green[idx]), np.array(las.blue[idx]))).T.astype(np.float32)
    else:
        rgb = None

    return None, xyz, rgb, total_points


def process_input_file(
    input_path,
    max_points=None,
    input_crs="EPSG:28355",
    status_callback: Callable[[dict[str, Any]], None] | None = None,
):
    """Process a LAS file or exported CSV point bucket through the segment-first pipeline."""
    publish_status("loading_las", f"Reading {os.path.basename(input_path)}", status_callback=status_callback)

    start = time.time()
    point_ids, xyz, rgb, total_points = load_input_points(input_path, max_points=max_points)

    processed_points = len(xyz)
    publish_status(
        "classifying",
        f"Loaded {processed_points:,} / {total_points:,} points in {time.time() - start:.1f}s.",
        completed_steps=processed_points,
        total_steps=total_points,
        status_callback=status_callback,
    )

    classify_start = time.time()
    labels, confidence = classify_points(xyz, rgb)
    class_time = time.time() - classify_start

    distribution = defaultdict(int)
    for label in labels:
        distribution[int(label)] += 1

    summary = ", ".join(
        f"{CLASS_NAMES.get(key, f'class_{key}')}: {value:,}"
        for key, value in sorted(distribution.items())
    )

    publish_status(
        "segmenting",
        f"Classification finished in {class_time:.1f}s ({summary}).",
        status_callback=status_callback,
    )

    segment_start = time.time()
    segment_ids, segments = build_segments(xyz, labels, confidence)
    segment_time = time.time() - segment_start
    publish_status(
        "reconstructing",
        f"Built {len(segments)} segments in {segment_time:.1f}s.",
        status_callback=status_callback,
    )

    reconstruct_start = time.time()
    meshes = reconstruct_segment_meshes(xyz, rgb, segment_ids, segments)
    reconstruct_time = time.time() - reconstruct_start

    total_vertices = sum(len(mesh["vertices"]) for mesh in meshes)
    total_faces = sum(len(mesh["faces"]) for mesh in meshes)
    publish_status(
        "generating_glb",
        f"Reconstructed {len(meshes)} segment meshes in {reconstruct_time:.1f}s ({total_vertices:,} vertices, {total_faces:,} faces).",
        status_callback=status_callback,
    )

    transformer = Transformer.from_crs(input_crs, "EPSG:4326", always_xy=True)
    center_lon, center_lat = transformer.transform(float(xyz[:, 0].mean()), float(xyz[:, 1].mean()))
    tile_x, tile_y, zoom = latlon_to_tile(center_lat, center_lon, DEFAULT_ZOOM)

    glb_dir = os.path.join(OUTPUT_DIR, f"{zoom}.{tile_x}.{tile_y}")
    glb_path = os.path.join(glb_dir, f"{zoom}.{tile_x}.{tile_y}.glb")
    os.makedirs(glb_dir, exist_ok=True)

    success, scene_origin = generate_glb(meshes, glb_path)
    if not success:
        raise RuntimeError("No valid segment meshes were produced for GLB export.")

    segment_data_path, segment_metadata_path = save_segment_artifacts(
        glb_dir,
        os.path.basename(input_path),
        xyz,
        rgb,
        labels,
        segment_ids,
        segments,
        point_ids=point_ids,
        scene_origin=scene_origin,
    )

    result = {
        "glbPath": glb_path,
        "zoom": int(zoom),
        "x": int(tile_x),
        "y": int(tile_y),
        "meshCount": int(len(meshes)),
        "segmentCount": int(len(segments)),
        "vertexCount": int(total_vertices),
        "faceCount": int(total_faces),
        "featureTypes": sorted({segment["featureType"] for segment in segments}),
        "segmentDataPath": segment_data_path,
        "segmentMetadataPath": segment_metadata_path,
        "sceneOrigin": [float(value) for value in np.asarray(scene_origin, dtype=np.float64)],
        "segments": segments,
    }

    publish_status(
        "generated_glb",
        f"Generated GLB for tile ({tile_x},{tile_y}) at zoom {zoom}.",
        status_callback=status_callback,
    )
    if status_callback is None:
        emit_result(result)
    return result


def resolve_input_path(explicit_path):
    if explicit_path:
        return explicit_path

    if os.path.exists(LAS_DIR):
        candidates = sorted(
            [name for name in os.listdir(LAS_DIR) if name.lower().endswith((".las", ".laz"))]
        )
        if candidates:
            return os.path.join(LAS_DIR, candidates[0])

    raise FileNotFoundError("No LAS files found. Upload via the web UI first.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--max-points", dest="max_points", type=int, default=0)
    args = parser.parse_args()

    input_path = resolve_input_path(args.input_path or os.environ.get("PIPELINE_INPUT_PATH"))
    process_input_file(input_path, max_points=args.max_points)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        sys.exit(1)
