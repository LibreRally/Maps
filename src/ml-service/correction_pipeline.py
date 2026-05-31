"""
Segment-review to training-data export.

Approved reclassification reviews export actual segment point subsets.
Approved split/merge reviews export segmentation-feedback metadata for later model work.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import requests


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TILE_SERVER = os.environ.get("TILESERVER_URL", "http://localhost:5034")
DATA_DIR = os.path.join(REPO_ROOT, "data", "training")
TRAINING_DATASET = os.path.join(DATA_DIR, "segment_reviews_dataset.json")

LABEL_MAP = {
    "building": 6,
    "ground": 2,
    "low_vegetation": 3,
    "medium_vegetation": 4,
    "high_vegetation": 5,
    "road": 11,
    "water": 9,
    "wire": 14,
    "tower": 15,
    "unclassified": 1,
}


def fetch_training_reviews():
    response = requests.get(f"{TILE_SERVER}/api/segments/reviews/training")
    response.raise_for_status()
    payload = response.json()
    print(
        f"Fetched {payload['count']} approved segment reviews awaiting export "
        f"({payload.get('exportedCount', 0)} already exported)"
    )
    return payload


def mark_reviews_exported(review_ids):
    if not review_ids:
        return None

    response = requests.post(
        f"{TILE_SERVER}/api/segments/reviews/exported",
        json={"reviewIds": review_ids},
    )
    response.raise_for_status()
    return response.json()


def load_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        return None

    with open(dataset_path, encoding="utf-8") as handle:
        return json.load(handle)


def merge_records(existing_records, exported_records):
    merged = {record["reviewId"]: record for record in existing_records}
    for record in exported_records:
        merged[record["reviewId"]] = record

    return sorted(
        merged.values(),
        key=lambda record: (
            record.get("reviewedAt") or "",
            record.get("reviewId") or "",
        ),
        reverse=True,
    )


def resolve_artifact_path(path):
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def export_review(review, output_dir):
    artifact_path = resolve_artifact_path(review.get("artifactPath"))
    if not artifact_path or not os.path.exists(artifact_path):
        return None

    archive = np.load(artifact_path)
    points = archive["points"]
    segment_ids = archive["segment_ids"]
    local_segment_id = review["localSegmentId"]
    mask = segment_ids == local_segment_id
    if not np.any(mask):
        return None

    segment_points = points[mask]
    segment_file = os.path.join(output_dir, f"{review['segmentId']}.npz")

    record = {
        "reviewId": review["id"],
        "segmentId": review["segmentId"],
        "tileId": review["tileId"],
        "sourceImportIds": review.get("segmentSourceImportIds") or review.get("tileSourceImportIds") or [],
        "localSegmentId": local_segment_id,
        "correctionType": review["correctionType"],
        "predictedLabel": review.get("previousLabel") or review.get("predictedLabel"),
        "reviewedLabel": review.get("reviewedLabel"),
        "requestedLabel": review.get("requestedLabel"),
        "submittedAt": review.get("submittedAt"),
        "reviewedAt": review.get("reviewedAt"),
        "pointCount": int(len(segment_points)),
        "artifact": os.path.relpath(segment_file, REPO_ROOT).replace("\\", "/"),
    }

    if review["correctionType"] == "reclassify" and review.get("requestedLabel") in LABEL_MAP:
        labels = np.full(len(segment_points), LABEL_MAP[review["requestedLabel"]], dtype=np.int32)
        np.savez_compressed(segment_file, points=segment_points.astype(np.float32), labels=labels)
        record["exportKind"] = "classification"
    else:
        np.savez_compressed(segment_file, points=segment_points.astype(np.float32))
        record["exportKind"] = "segmentation_feedback"
        record["relatedSegmentIds"] = review.get("relatedSegmentIds") or []

    return record


def main():
    print("=" * 60)
    print("Segment Review -> Training Export")
    print("=" * 60)

    output_dir = os.path.join(DATA_DIR, "segments")
    os.makedirs(output_dir, exist_ok=True)

    payload = fetch_training_reviews()
    reviews = payload.get("reviews", [])
    exported = []
    for review in reviews:
        record = export_review(review, output_dir)
        if record:
            exported.append(record)

    existing_dataset = load_dataset(TRAINING_DATASET) or {}
    merged_records = merge_records(existing_dataset.get("records", []), exported)

    dataset = {
        "version": "2.1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "totalReviews": payload.get("totalCount", payload.get("count", 0) + payload.get("exportedCount", 0)),
        "exportedSamples": len(merged_records),
        "records": merged_records,
    }

    with open(TRAINING_DATASET, "w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2)

    export_state = mark_reviews_exported([record["reviewId"] for record in exported])
    if export_state:
        print(
            f"Marked {export_state.get('count', 0)} reviews as exported at "
            f"{export_state.get('exportedAt')}"
        )

    print(
        f"Exported {len(exported)} new training samples "
        f"({len(merged_records)} total) to {TRAINING_DATASET}"
    )


if __name__ == "__main__":
    main()
