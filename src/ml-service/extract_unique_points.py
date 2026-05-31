import argparse
import hashlib
import json

import laspy
import numpy as np


def first_unique_xyz_indices(xyz):
    if len(xyz) == 0:
        return np.array([], dtype=np.int64)

    structured = np.ascontiguousarray(xyz).view(
        np.dtype([("x", np.float64), ("y", np.float64), ("z", np.float64)])
    ).reshape(-1)
    _, indices = np.unique(structured, return_index=True)
    return np.sort(indices.astype(np.int64))


def build_point_signature(unique_xyz):
    if len(unique_xyz) == 0:
        return None

    order = np.lexsort((unique_xyz[:, 2], unique_xyz[:, 1], unique_xyz[:, 0]))
    normalized = np.ascontiguousarray(unique_xyz[order], dtype=np.float64)
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    las = laspy.read(args.input)
    xyz = np.vstack(
        (
            np.asarray(las.x, dtype=np.float64),
            np.asarray(las.y, dtype=np.float64),
            np.asarray(las.z, dtype=np.float64),
        )
    ).T

    total_points = int(len(xyz))
    unique_indices = first_unique_xyz_indices(xyz)
    unique_xyz = xyz[unique_indices]

    if all(hasattr(las, channel) for channel in ("red", "green", "blue")):
        rgb = np.vstack(
            (
                np.asarray(las.red, dtype=np.int32),
                np.asarray(las.green, dtype=np.int32),
                np.asarray(las.blue, dtype=np.int32),
            )
        ).T
        unique_rgb = rgb[unique_indices]
    else:
        unique_rgb = np.full((len(unique_xyz), 3), -1, dtype=np.int32)

    exported = np.column_stack((unique_xyz, unique_rgb))
    np.savetxt(
        args.output_csv,
        exported,
        delimiter=",",
        fmt=["%.17g", "%.17g", "%.17g", "%d", "%d", "%d"],
    )

    payload = {
        "totalPoints": total_points,
        "uniquePoints": int(len(unique_xyz)),
        "duplicateWithinImportPoints": int(total_points - len(unique_xyz)),
        "pointSignature": build_point_signature(unique_xyz),
    }

    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


if __name__ == "__main__":
    main()
