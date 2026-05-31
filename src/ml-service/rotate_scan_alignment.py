import argparse
import json
import math

import numpy as np
import trimesh


def transform_local_xy(points: np.ndarray, rotation_degrees: float, translate_xy: np.ndarray) -> np.ndarray:
    radians = math.radians(rotation_degrees)
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)
    rotation = np.array(
        [
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ],
        dtype=np.float64,
    )

    transformed = np.asarray(points, dtype=np.float64).copy()
    transformed[:, :2] = transformed[:, :2] @ rotation.T
    transformed[:, :2] += translate_xy
    return transformed


def transform_world_xy(
    points: np.ndarray,
    pivot_xy: np.ndarray,
    rotation_degrees: float,
    translate_xy: np.ndarray,
) -> np.ndarray:
    transformed = np.asarray(points, dtype=np.float64).copy()
    transformed[:, :2] -= pivot_xy
    transformed = transform_local_xy(transformed, rotation_degrees, translate_xy)
    transformed[:, :2] += pivot_xy
    return transformed


def compute_segment_geometry(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    centroid = bbox_min + np.mean(points - bbox_min, axis=0, dtype=np.float64)
    centroid = np.clip(centroid, bbox_min, bbox_max)
    return bbox_min, bbox_max, centroid


def transform_glb_in_place(glb_path: str, rotation_degrees: float, translate_xy: np.ndarray) -> None:
    scene = trimesh.load(glb_path, force="scene")
    for name, geometry in scene.geometry.items():
        if name.lower().endswith("-osmfill"):
            continue

        vertices = np.asarray(geometry.vertices, dtype=np.float64)
        if len(vertices) == 0:
            continue

        geometry.vertices = transform_local_xy(vertices, rotation_degrees, translate_xy)

    scene.export(glb_path)


def transform_segment_archive(
    segment_path: str,
    metadata_path: str,
    pivot_xy: np.ndarray,
    rotation_degrees: float,
    translate_xy: np.ndarray,
) -> list[dict]:
    archive = np.load(segment_path)
    points = np.asarray(archive["points"], dtype=np.float64)
    segment_ids = np.asarray(archive["segment_ids"], dtype=np.int32)

    transformed_points = transform_world_xy(points, pivot_xy, rotation_degrees, translate_xy)

    payload = {
        "points": transformed_points,
        "colors": archive["colors"],
        "labels": archive["labels"],
        "segment_ids": segment_ids,
    }
    if "point_ids" in archive.files:
        payload["point_ids"] = archive["point_ids"]

    np.savez_compressed(segment_path, **payload)

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    updated_segments: list[dict] = []
    for segment in metadata.get("segments", []):
        local_id = int(segment.get("localId", -1))
        member_points = transformed_points[segment_ids == local_id]
        if len(member_points) == 0:
            updated_segments.append(segment)
            continue

        bounds_min, bounds_max, centroid = compute_segment_geometry(member_points)
        updated = dict(segment)
        updated["boundsMin"] = [float(value) for value in bounds_min]
        updated["boundsMax"] = [float(value) for value in bounds_max]
        updated["centroid"] = [float(value) for value in centroid]
        updated_segments.append(updated)

    metadata["segments"] = updated_segments
    metadata["segmentCount"] = len(updated_segments)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)

    return updated_segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--pivot-x", required=True, type=float)
    parser.add_argument("--pivot-y", required=True, type=float)
    parser.add_argument("--rotation-degrees", required=True, type=float)
    parser.add_argument("--translate-x", type=float, default=0.0)
    parser.add_argument("--translate-y", type=float, default=0.0)
    args = parser.parse_args()

    translate_xy = np.array([args.translate_x, args.translate_y], dtype=np.float64)
    transform_glb_in_place(args.glb, args.rotation_degrees, translate_xy)
    segments = transform_segment_archive(
        args.segments,
        args.metadata,
        np.array([args.pivot_x, args.pivot_y], dtype=np.float64),
        args.rotation_degrees,
        translate_xy,
    )

    print(
        json.dumps(
            {
                "rotated": True,
                "rotationDegrees": args.rotation_degrees,
                "translateX": args.translate_x,
                "translateY": args.translate_y,
                "segments": segments,
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
