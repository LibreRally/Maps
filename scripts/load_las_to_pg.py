"""
Load a sample of LAS points into pgpointcloud to verify the pipeline.
Full bulk loading should use PDAL's writers.pgpointcloud.
"""
import sys, os, time
import numpy as np
import laspy
import psycopg2
from pyproj import Transformer

DB_CONFIG = {
    'host': 'localhost', 'port': 5432,
    'dbname': 'mapsdb', 'user': 'mapsuser', 'password': 'mapspass',
}

PC_SCHEMA_XML = """<?xml version="1.0" encoding="UTF-8"?>
<pc:PointCloudSchema xmlns:pc="http://pointcloud.org/schemas/PC/2.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <pc:dimension><pc:position>1</pc:position><pc:name>X</pc:name><pc:size>4</pc:size><pc:description>X</pc:description></pc:dimension>
  <pc:dimension><pc:position>2</pc:position><pc:name>Y</pc:name><pc:size>4</pc:size><pc:description>Y</pc:description></pc:dimension>
  <pc:dimension><pc:position>3</pc:position><pc:name>Z</pc:name><pc:size>4</pc:size><pc:description>Z</pc:description></pc:dimension>
  <pc:dimension><pc:position>4</pc:position><pc:name>Intensity</pc:name><pc:size>2</pc:size><pc:description>Intensity</pc:description></pc:dimension>
  <pc:dimension><pc:position>5</pc:position><pc:name>ReturnNumber</pc:name><pc:size>1</pc:size><pc:description>ReturnNumber</pc:description></pc:dimension>
  <pc:dimension><pc:position>6</pc:position><pc:name>NumberOfReturns</pc:name><pc:size>1</pc:size><pc:description>NumberOfReturns</pc:description></pc:dimension>
  <pc:dimension><pc:position>7</pc:position><pc:name>Classification</pc:name><pc:size>1</pc:size><pc:description>Classification</pc:description></pc:dimension>
  <pc:dimension><pc:position>8</pc:position><pc:name>Red</pc:name><pc:size>2</pc:size><pc:description>Red</pc:description></pc:dimension>
  <pc:dimension><pc:position>9</pc:position><pc:name>Green</pc:name><pc:size>2</pc:size><pc:description>Green</pc:description></pc:dimension>
  <pc:dimension><pc:position>10</pc:position><pc:name>Blue</pc:name><pc:size>2</pc:size><pc:description>Blue</pc:description></pc:dimension>
</pc:PointCloudSchema>
"""

def main():
    las_dir = os.getenv("LAS_DATA_DIR", "")
    if not las_dir:
        print("LAS_DATA_DIR environment variable not set. Point it to your LAS data directory.", file=sys.stderr)
        return 1
    tile_file = 'Tile_+014_+011.las'
    las_path = os.path.join(las_dir, tile_file)
    
    print(f"Reading {tile_file}...")
    las = laspy.read(las_path)
    n_total = len(las.x)
    
    sample_n = min(100, n_total)
    step = max(1, n_total // sample_n)
    
    x = np.array(las.x[::step][:sample_n], dtype=np.float64)
    y = np.array(las.y[::step][:sample_n], dtype=np.float64)
    z = np.array(las.z[::step][:sample_n], dtype=np.float64)
    intensity = np.array(las.intensity[::step][:sample_n], dtype=np.uint16)
    classification = np.array(las.classification[::step][:sample_n], dtype=np.uint8)
    red = np.array(las.red[::step][:sample_n], dtype=np.uint16)
    green = np.array(las.green[::step][:sample_n], dtype=np.uint16)
    blue = np.array(las.blue[::step][:sample_n], dtype=np.uint16)
    
    print(f"Sampled {sample_n} points from {n_total:,} total")
    
    transformer = Transformer.from_crs('EPSG:28355', 'EPSG:4326', always_xy=True)
    min_lon, min_lat = transformer.transform(x.min(), y.min())
    max_lon, max_lat = transformer.transform(x.max(), y.max())
    bounds_wkt = f'SRID=4326;POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))'
    
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            # Create schema if needed
            cur.execute("SELECT count(*) FROM pointcloud_formats WHERE pcid = 1")
            if cur.fetchone()[0] == 0:
                cur.execute("INSERT INTO pointcloud_formats (pcid, srid, schema) VALUES (%s, %s, %s)",
                           (1, 28355, PC_SCHEMA_XML))
                conn.commit()
                print("Created pointcloud schema (pcid=1)")
            else:
                print("Schema pcid=1 already exists")
            
            t0 = time.time()
            point_calls = []
            for i in range(sample_n):
                # PC_MakePoint(1, ARRAY[x, y, z, intensity, return_num, num_returns, class, red, green, blue])
                vals = f"ARRAY[{x[i]}::float8, {y[i]}::float8, {z[i]}::float8, " \
                       f"{int(intensity[i])}::float8, 1::float8, 1::float8, " \
                       f"{int(classification[i])}::float8, " \
                       f"{int(red[i])}::float8, {int(green[i])}::float8, {int(blue[i])}::float8]"
                point_calls.append(f"PC_MakePoint(1, {vals})")
            
            sql = f"""
                INSERT INTO pointcloud.tiles (tile_x, tile_y, source, crs, bounds, point_count, pc_patch, metadata)
                VALUES (14, 11, 'melbourne_2018', 28355,
                    ST_GeomFromText('{bounds_wkt}'),
                    {sample_n},
                    PC_Patch(ARRAY[{','.join(point_calls)}]),
                    '{{"sample": true, "original_points": {n_total}}}'::jsonb
                )
                RETURNING id;
            """
            
            cur.execute(sql)
            tile_id = cur.fetchone()[0]
            conn.commit()
            elapsed = time.time() - t0
            
            print(f"Inserted {sample_n} points as patch (tile_id={tile_id}) in {elapsed:.1f}s")
            
            # Verify
            cur.execute("""
                SELECT id, tile_x, tile_y, point_count, 
                       PC_NumPoints(pc_patch) as stored_points,
                       ST_AsText(bounds) as bounds_wkt,
                       PC_PatchAvg(pc_patch, 'Z') as avg_elevation
                FROM pointcloud.tiles WHERE id = %s
            """, (tile_id,))
            row = cur.fetchone()
            print(f"\nVerification:")
            print(f"  id={row[0]}, tile=({row[1]},{row[2]}), points={row[3]}")
            print(f"  stored points in patch: {row[4]}")
            print(f"  avg elevation: {row[6]:.1f}m")
            
    finally:
        conn.close()
    
    print("\nDone. pgpointcloud is working.")

if __name__ == '__main__':
    main()
