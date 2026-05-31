#!/usr/bin/env python3
"""
Apply Vicmap-computed alignment to a tile's source import and trigger re-processing.

Usage:
    python apply_vicmap_alignment.py                           \
        --segments-json data/tiles/14.14764.10065/segments.json \
        --tileserver http://192.168.1.38:5250                    \
        [--dry-run]

Steps:
    1. Compute alignment offset from Vicmap property boundaries
    2. Find the source import for the tile via the API
    3. POST the alignment values to the import
    4. (Optionally) trigger re-processing
"""

import argparse
import json
import subprocess
import sys
import urllib.request
import urllib.error

ALIGN_SCRIPT = "src/ml-service/align_to_vicmap.py"


def api_get(base_url: str, path: str) -> dict | list:
    """GET from the TileServer API, return parsed JSON."""
    url = f"{base_url.rstrip('/')}{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def api_post(base_url: str, path: str, body: dict) -> dict:
    """POST to the TileServer API, return parsed JSON."""
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def find_import_for_tile(base_url: str, tile_x: int, tile_y: int, zoom: int) -> dict | None:
    """Find the active import that sources this tile."""
    tiles = api_get(base_url, "/api/tiles/catalog")
    if isinstance(tiles, list):
        for t in tiles:
            tx = t.get("tileX") or t.get("tile_x")
            ty = t.get("tileY") or t.get("tile_y")
            tz = t.get("zoomLevel") or t.get("zoom") or t.get("zoom_level")
            if tx == tile_x and ty == tile_y and tz == zoom:
                return t
    elif isinstance(tiles, dict) and "tiles" in tiles:
        for t in tiles["tiles"]:
            tx = t.get("tileX") or t.get("tile_x")
            ty = t.get("tileY") or t.get("tile_y")
            tz = t.get("zoomLevel") or t.get("zoom") or t.get("zoom_level")
            if tx == tile_x and ty == tile_y and tz == zoom:
                return t
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Vicmap alignment to a tile")
    parser.add_argument("--segments-json", required=True, help="Path to segments.json")
    parser.add_argument("--tileserver", required=True, help="TileServer base URL (e.g. http://192.168.1.38:5250)")
    parser.add_argument("--dry-run", action="store_true", help="Compute alignment but don't apply")
    parser.add_argument("--yaw", type=float, default=-90, help="Yaw degrees (default: -90)")
    args = parser.parse_args()

    # ── Step 1: Compute alignment ──────────────────────────────────
    print("▸ Computing alignment from Vicmap…")
    result = subprocess.run(
        ["python", ALIGN_SCRIPT, "--segments-json", args.segments_json],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"Alignment script failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    alignment = json.loads(result.stdout)
    offset_x = alignment["alignment"]["offset_x"]
    offset_y = alignment["alignment"]["offset_y"]
    print(f"  → Offset: X={offset_x:+.3f}m, Y={offset_y:+.3f}m")
    print(f"  → Matched {alignment['matched_pairs']} of {alignment['total_buildings']} buildings\n")

    # ── Step 2: Parse tile coordinates from segments.json path ─────
    import re
    m = re.search(r'(\d+)\.(\d+)\.(\d+)', args.segments_json)
    if not m:
        print("ERROR: Could not parse tile coords from segments.json path", file=sys.stderr)
        sys.exit(1)
    zoom, tile_x, tile_y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    print(f"▸ Tile: zoom={zoom}, x={tile_x}, y={tile_y}")

    # ── Step 3: Find matching import ───────────────────────────────
    try:
        tile_info = find_import_for_tile(args.tileserver, tile_x, tile_y, zoom)
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot reach TileServer at {args.tileserver}: {e}", file=sys.stderr)
        sys.exit(1)

    if tile_info is None:
        print("ERROR: No matching tile found on server. Has it been registered?", file=sys.stderr)
        sys.exit(1)

    source_ids = tile_info.get("sourceImportIds") or tile_info.get("source_import_ids") or []
    if not source_ids:
        print("ERROR: Tile has no source import IDs. Re-process the tile first.", file=sys.stderr)
        sys.exit(1)

    import_id = source_ids[0]
    print(f"  → Found tile {tile_info.get('id')} sourced by import {import_id}")

    # ── Step 4: POST alignment ─────────────────────────────────────
    payload = {
        "clear": False,
        "yawDegrees": args.yaw,
        "offsetX": offset_x,
        "offsetY": offset_y,
        "source": "vicmap_auto",
    }

    if args.dry_run:
        print(f"\n▸ DRY RUN — would POST to /api/imports/{import_id}/alignment:")
        print(f"  {json.dumps(payload, indent=2)}")
        print("\nThen re-process with:")
        print(f'  POST /api/tiles/process/imports {{"importIds":["{import_id}"]}}')
        return

    print(f"\n▸ Applying alignment to import {import_id}…")
    resp = api_post(args.tileserver, f"/api/imports/{import_id}/alignment", payload)
    print(f"  → {json.dumps(resp, indent=2)}")

    # ── Step 5: Trigger re-processing ──────────────────────────────
    print(f"\n▸ Triggering re-process for import {import_id}…")
    try:
        process_result = api_post(args.tileserver, "/api/tiles/process/imports",
                                  {"importIds": [import_id]})
        print(f"  → {json.dumps(process_result, indent=2)}")
        print("\n✓ Alignment applied and re-processing started.")
        print("  Monitor progress at GET /api/import/status or the web UI.")
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else str(e)
        print(f"  Re-process trigger failed ({e.code}): {body}", file=sys.stderr)
        print("\n  Alignment was saved. Manually re-process the tile to apply it.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
