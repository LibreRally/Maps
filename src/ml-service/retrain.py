"""
Training dataset inspector for exported segment reviews.

This still does not run a full deep-learning retrain, but it now reads the
actual exported segment-review dataset rather than synthetic correction stubs.
"""
import json
import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAINING_DATASET = os.path.join(REPO_ROOT, "data", "training", "segment_reviews_dataset.json")
RETRAIN_THRESHOLD = 25


def load_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        return None

    with open(dataset_path, encoding="utf-8") as handle:
        return json.load(handle)


def summarize(dataset):
    by_kind = {}
    by_label = {}
    by_source = {}
    missing_provenance = 0
    for record in dataset.get("records", []):
        kind = record.get("exportKind", "unknown")
        by_kind[kind] = by_kind.get(kind, 0) + 1

        label = record.get("requestedLabel") or record.get("predictedLabel") or "unknown"
        by_label[label] = by_label.get(label, 0) + 1

        source_import_ids = record.get("sourceImportIds") or []
        if source_import_ids:
            source_key = ", ".join(str(import_id)[:8] for import_id in source_import_ids)
            by_source[source_key] = by_source.get(source_key, 0) + 1
        else:
            missing_provenance += 1

    return by_kind, by_label, by_source, missing_provenance


def retrain_model(dataset_path):
    dataset = load_dataset(dataset_path)
    if dataset is None:
        print(f"No exported dataset found at {dataset_path}. Run correction_pipeline.py first.")
        return

    total = max(dataset.get("exportedSamples", 0), len(dataset.get("records", [])))
    by_kind, by_label, by_source, missing_provenance = summarize(dataset)

    print("=" * 60)
    print("Segment Review Retraining Summary")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Exported samples: {total}")

    print("\nBy export kind:")
    for kind, count in sorted(by_kind.items()):
        print(f"  {kind:24s} {count:>5}")

    print("\nBy target label:")
    for label, count in sorted(by_label.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {label:24s} {count:>5}")

    if by_source:
        print("\nBy source provenance:")
        for source, count in sorted(by_source.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {source:24.24s} {count:>5}")

    if missing_provenance:
        print(f"\nWARNING: {missing_provenance} exported records are missing source provenance.")

    if total < RETRAIN_THRESHOLD:
        print(f"\nOnly {total} exported samples are available. Need {RETRAIN_THRESHOLD} to justify retraining.")
        return

    print(f"\nThreshold met ({total} >= {RETRAIN_THRESHOLD}).")
    print("Next manual step: point the training job at the exported .npz samples in data/training/segments/.")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=TRAINING_DATASET)
    args = parser.parse_args()

    retrain_model(args.dataset)


if __name__ == "__main__":
    main()
