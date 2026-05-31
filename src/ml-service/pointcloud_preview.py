import argparse
import json
import os
import sys

import laspy
import numpy as np


def build_segment_colors(segment_ids):
    palette = np.array(
        [
            [0.90, 0.45, 0.35],
            [0.35, 0.75, 0.45],
            [0.30, 0.55, 0.90],
            [0.90, 0.80, 0.30],
            [0.75, 0.45, 0.90],
            [0.35, 0.85, 0.85],
            [0.95, 0.60, 0.20],
        ],
        dtype=np.float32,
    )
    colors = np.zeros((len(segment_ids), 3), dtype=np.float32)
    for index, segment_id in enumerate(segment_ids):
        if segment_id < 0:
            continue
        colors[index] = palette[int(segment_id) % len(palette)]
    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--segments")
    parser.add_argument("--csv")
    parser.add_argument("--max-points", type=int, default=100000)
    args = parser.parse_args()

    if not args.input and not args.segments and not args.csv:
        raise SystemExit("Provide --input, --segments, or --csv.")

    if args.segments:
        archive = np.load(args.segments)
        points = archive["points"].astype(np.float32)
        segment_ids = archive["segment_ids"].astype(np.int32)
        total_points = len(points)
        
        metadata_path = args.segments.replace("segments.npz", "segments.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    if "sceneOrigin" in metadata:
                        scene_origin = np.array(metadata["sceneOrigin"], dtype=np.float32)
                        points = points - scene_origin
            except Exception as e:
                print(f"Warning: Failed to apply sceneOrigin: {e}", file=sys.stderr)

        if args.max_points > 0 and total_points > args.max_points:
            step = max(total_points // args.max_points, 1)
            idx = slice(0, total_points, step)
        else:
            idx = slice(None)

        xyz = points[idx]
        sampled_points = len(xyz)
        colors = build_segment_colors(segment_ids[idx]).reshape(-1).tolist()
        sampled_segment_ids = segment_ids[idx].tolist()
        file_name = os.path.basename(args.segments)
    elif args.csv:
        rows = np.loadtxt(args.csv, delimiter=",", dtype=np.float64)
        rows = np.atleast_2d(rows)
        total_points = len(rows)

        if args.max_points > 0 and total_points > args.max_points:
            step = max(total_points // args.max_points, 1)
            idx = slice(0, total_points, step)
        else:
            idx = slice(None)

        sampled = rows[idx]
        if sampled.shape[1] >= 7:
            xyz = sampled[:, 1:4].astype(np.float32)
            rgb_values = sampled[:, 4:7]
        else:
            xyz = sampled[:, :3].astype(np.float32)
            rgb_values = sampled[:, 3:6] if sampled.shape[1] >= 6 else None

        sampled_points = len(xyz)
        colors = None
        sampled_segment_ids = None
        if rgb_values is not None and np.any(rgb_values >= 0):
            rgb = rgb_values.astype(np.float32)
            max_color = max(float(rgb.max()), 1.0)
            colors = (rgb / max_color).reshape(-1).tolist()

        file_name = os.path.basename(args.csv)
    else:
        las = laspy.read(args.input)
        total_points = len(las.x)

        if args.max_points > 0 and total_points > args.max_points:
            step = max(total_points // args.max_points, 1)
            idx = slice(0, total_points, step)
        else:
            idx = slice(None)

        xyz = np.vstack((np.array(las.x[idx]), np.array(las.y[idx]), np.array(las.z[idx]))).T.astype(np.float32)
        sampled_points = len(xyz)

        colors = None
        sampled_segment_ids = None
        if all(hasattr(las, channel) for channel in ("red", "green", "blue")):
            rgb = np.vstack((np.array(las.red[idx]), np.array(las.green[idx]), np.array(las.blue[idx]))).T.astype(np.float32)
            max_color = max(float(rgb.max()), 1.0)
            colors = (rgb / max_color).reshape(-1).tolist()
        file_name = os.path.basename(args.input)

    payload = {
        "fileName": file_name,
        "totalPoints": int(total_points),
        "sampledPoints": int(sampled_points),
        "points": xyz.reshape(-1).tolist(),
        "colors": colors,
        "segmentIds": sampled_segment_ids,
    }

    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
