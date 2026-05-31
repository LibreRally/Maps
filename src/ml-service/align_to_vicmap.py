#!/usr/bin/env python3
"""
Align Scaniverse LiDAR scan building segments to Vicmap property boundaries.

Fetches Vicmap property polygons from the ArcGIS FeatureServer, matches them
to building segment centroids from segments.json, and computes the optimal rigid transform
(rotation + translation) to align the scan data to cadastral ground truth.

Usage:
    python align_to_vicmap.py --segments-json data/tiles/14.14764.10065/segments.json

Output:
    JSON with alignment values ready for POST /api/imports/{id}/alignment
"""

import argparse
import json
import math
import sys
import time
import urllib.parse
import urllib.error
import urllib.request

import numpy as np

VICMAP_FEATURESERVER_URL = (
    "https://services-ap1.arcgis.com/P744lA0wf4LlBZ84/arcgis/rest/services/"
    "Vicmap_Property/FeatureServer/0/query"
)
QUERY_BUFFER_METRES = 80  # Expand tile bounds for FeatureServer query
MAX_MATCH_DISTANCE = 40  # Metres — max distance to match building to property
VICMAP_REQUEST_TIMEOUT_SECONDS = 60
VICMAP_REQUEST_MAX_ATTEMPTS = 2


def fetch_vicmap_properties(min_x: float, min_y: float, max_x: float, max_y: float) -> dict:
    """Query the Vicmap ArcGIS FeatureServer for property polygons in EPSG:28355."""
    params = {
        "where": "1=1",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "28355",
        "outSR": "28355",
        "f": "pjson",
        "returnGeometry": "true",
        "outFields": "prop_pfi,propv_pfi,propv_base_pfi",
        "resultRecordCount": "200",
        "geometry": json.dumps({
            "xmin": min_x,
            "ymin": min_y,
            "xmax": max_x,
            "ymax": max_y,
        }),
    }
    url = VICMAP_FEATURESERVER_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"Accept": "application/json"})

    last_error: Exception | None = None
    for attempt in range(1, VICMAP_REQUEST_MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=VICMAP_REQUEST_TIMEOUT_SECONDS) as response:
                return json.loads(response.read())
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt >= VICMAP_REQUEST_MAX_ATTEMPTS:
                break

            print(
                f"Vicmap FeatureServer request attempt {attempt} failed: {exc}. Retrying...",
                file=sys.stderr,
            )
            time.sleep(attempt * 2)

    assert last_error is not None
    raise last_error


def polygon_centroid(geom: dict) -> tuple[float, float] | None:
    """Compute the centroid of an ArcGIS polygon geometry in EPSG:28355."""
    rings = geom.get("rings")
    if not rings:
        return None

    coords = rings[0]
    xs = [p[0] for p in coords if len(p) >= 2]
    ys = [p[1] for p in coords if len(p) >= 2]
    if not xs or not ys:
        return None

    return (sum(xs) / len(xs), sum(ys) / len(ys))


def main() -> None:
    parser = argparse.ArgumentParser(description="Align scan to Vicmap cadastral data")
    parser.add_argument("--segments-json", required=True, help="Path to segments.json for the tile")
    parser.add_argument("--output", help="Optional output JSON path")
    parser.add_argument("--max-distance", type=float, default=MAX_MATCH_DISTANCE,
                        help=f"Max centroid distance for matching (default: {MAX_MATCH_DISTANCE}m)")
    args = parser.parse_args()

    # ── Read building segments ──────────────────────────────────────
    with open(args.segments_json, encoding="utf-8") as handle:
        tile_data = json.load(handle)

    buildings = [seg for seg in tile_data.get("segments", [])
                 if seg.get("featureType") == "building"]
    if not buildings:
        print("ERROR: No building segments found in segments.json", file=sys.stderr)
        sys.exit(1)

    scene_origin = tile_data.get("sceneOrigin")
    if scene_origin:
        print(f"sceneOrigin: {scene_origin}", file=sys.stderr)

    # ── Compute query bounds ────────────────────────────────────────
    all_x = [p for s in buildings for p in (s["boundsMin"][0], s["boundsMax"][0])]
    all_y = [p for s in buildings for p in (s["boundsMin"][1], s["boundsMax"][1])]
    qx0, qx1 = min(all_x) - QUERY_BUFFER_METRES, max(all_x) + QUERY_BUFFER_METRES
    qy0, qy1 = min(all_y) - QUERY_BUFFER_METRES, max(all_y) + QUERY_BUFFER_METRES

    # ── Fetch Vicmap properties ─────────────────────────────────────
    print(f"Fetching Vicmap FeatureServer: bbox=({qx0:.0f},{qy0:.0f},{qx1:.0f},{qy1:.0f})", file=sys.stderr)
    vicmap = fetch_vicmap_properties(qx0, qy0, qx1, qy1)
    features = vicmap.get("features", [])
    print(f"Returned {len(features)} Vicmap properties", file=sys.stderr)
    total_matched = vicmap.get("count") or len(features)
    print(f"  (total available in bbox: {total_matched})", file=sys.stderr)

    # ── Compute property centroids ──────────────────────────────────
    prop_entries: list[dict] = []
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        centroid = polygon_centroid(geom)
        if centroid is None:
            continue
        prop_entries.append({
            "centroid": centroid,
            "pfi": feat.get("attributes", {}).get("prop_pfi")
                or feat.get("attributes", {}).get("propv_pfi")
                or feat.get("attributes", {}).get("propv_base_pfi"),
            "geom": geom,
        })

    # ── Greedy 1:1 matching ─────────────────────────────────────────
    building_centroids = [(s["centroid"][0], s["centroid"][1]) for s in buildings]

    used_props: set[int] = set()
    matches: list[dict] = []

    for bi, bc in enumerate(building_centroids):
        best_idx = -1
        best_dist = float("inf")
        for pi, pc in enumerate(prop_entries):
            if pi in used_props:
                continue
            dx = bc[0] - pc["centroid"][0]
            dy = bc[1] - pc["centroid"][1]
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best_idx = pi

        if best_idx >= 0 and best_dist <= args.max_distance:
            pc = prop_entries[best_idx]
            matches.append({
                "building_index": bi,
                "building_centroid": bc,
                "property_centroid": pc["centroid"],
                "property_pfi": pc["pfi"],
                "centroid_distance": round(best_dist, 3),
                "offset_x": round(pc["centroid"][0] - bc[0], 3),
                "offset_y": round(pc["centroid"][1] - bc[1], 3),
            })
            used_props.add(best_idx)

    if not matches:
        print("ERROR: No building-property matches within distance threshold.", file=sys.stderr)
        print("Try increasing --max-distance or expanding the query area.", file=sys.stderr)
        sys.exit(1)

    # ── Compute translation offset ──────────────────────────────────
    offsets_x = np.array([m["offset_x"] for m in matches])
    offsets_y = np.array([m["offset_y"] for m in matches])
    avg_offset_x = float(np.mean(offsets_x))
    avg_offset_y = float(np.mean(offsets_y))

    # ── Compute rotation (pairwise vector comparison) ───────────────
    rotations: list[float] = []
    for i in range(len(matches)):
        for j in range(i + 1, len(matches)):
            b_vec = np.array(matches[j]["building_centroid"]) - np.array(matches[i]["building_centroid"])
            p_vec = np.array(matches[j]["property_centroid"]) - np.array(matches[i]["property_centroid"])
            b_len = np.linalg.norm(b_vec)
            p_len = np.linalg.norm(p_vec)
            if b_len < 0.5 or p_len < 0.5:
                continue  # skip degenerate pairs
            b_angle = math.atan2(b_vec[1], b_vec[0])
            p_angle = math.atan2(p_vec[1], p_vec[0])
            rot_degrees = math.degrees(p_angle - b_angle)
            # Normalize to [-180, 180]
            while rot_degrees > 180:
                rot_degrees -= 360
            while rot_degrees < -180:
                rot_degrees += 360
            rotations.append(rot_degrees)

    avg_yaw = float(np.mean(rotations)) if rotations else 0.0
    yaw_stdev = float(np.std(rotations)) if rotations else 0.0

    # Translation std dev for confidence
    offset_x_stdev = float(np.std(offsets_x))
    offset_y_stdev = float(np.std(offsets_y))

    # ── Output ──────────────────────────────────────────────────────
    result = {
        "matched_pairs": len(matches),
        "total_buildings": len(buildings),
        "total_vicmap_properties": total_matched,
        "alignment": {
            "yaw_degrees": round(avg_yaw, 3),
            "yaw_stdev": round(yaw_stdev, 3),
            "offset_x": round(avg_offset_x, 3),
            "offset_x_stdev": round(offset_x_stdev, 3),
            "offset_y": round(avg_offset_y, 3),
            "offset_y_stdev": round(offset_y_stdev, 3),
        },
        "scene_origin": scene_origin,
        "matches": [{
            "pfi": m["property_pfi"],
            "building_centroid": [round(m["building_centroid"][0], 3), round(m["building_centroid"][1], 3)],
            "property_centroid": [round(m["property_centroid"][0], 3), round(m["property_centroid"][1], 3)],
            "offset_x": m["offset_x"],
            "offset_y": m["offset_y"],
            "distance": m["centroid_distance"],
        } for m in matches],
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))

    # ── Human-readable summary ──────────────────────────────────────
    print(f"\n{'─'*50}", file=sys.stderr)
    print(f"ALIGNMENT RESULT  ({result['matched_pairs']} building↔property pairs)", file=sys.stderr)
    print(f"{'─'*50}", file=sys.stderr)
    print(f"  Yaw:       {avg_yaw:+.3f}°  (σ={yaw_stdev:.3f}°)", file=sys.stderr)
    print(f"  Offset X:  {avg_offset_x:+.3f}m  (σ={offset_x_stdev:.3f}m)", file=sys.stderr)
    print(f"  Offset Y:  {avg_offset_y:+.3f}m  (σ={offset_y_stdev:.3f}m)", file=sys.stderr)
    print(f"\nApply via API:", file=sys.stderr)
    print(f'  POST /api/imports/{{id}}/alignment', file=sys.stderr)
    print(f'  {{"clear":false,"yawDegrees":{avg_yaw:.3f},"offsetX":{avg_offset_x:.3f},"offsetY":{avg_offset_y:.3f},"source":"vicmap_auto"}}', file=sys.stderr)
    print(f"\nThen re-process the tile.", file=sys.stderr)


if __name__ == "__main__":
    main()
