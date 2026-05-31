"""Extract and inspect baked textures from a GLB file."""
import json
import struct
import sys
import zlib

if len(sys.argv) < 2:
    print("Usage: python check_textures.py <path/to/file.glb>")
    sys.exit(1)
path = sys.argv[1]

with open(path, "rb") as f:
    data = f.read()

# Parse GLB header
offset = 12
chunk_length = struct.unpack_from("<I", data, offset)[0]
json_start = offset + 8
json_data = json.loads(data[json_start : json_start + chunk_length].decode("utf-8"))

buffer_views = json_data.get("bufferViews", [])
images = json_data.get("images", [])
materials = json_data.get("materials", [])

# BIN chunk starts after JSON chunk
bin_start = json_start + chunk_length
bin_chunk_len = struct.unpack_from("<I", data, bin_start)[0]
bin_data_start = bin_start + 8

# Check materials to identify which image is which
for mi, mat in enumerate(materials[:3]):
    pbr = mat.get("pbrMetallicRoughness", {})
    bct = pbr.get("baseColorTexture", {})
    nt = mat.get("normalTexture", {})
    bc_idx = bct.get("index", -1) if bct else -1
    n_idx = nt.get("index", -1) if nt else -1
    print(f"Material {mi}: baseColorTexture index={bc_idx}, normalTexture index={n_idx}")

# Inspect first 3 base colour textures
for i in range(min(3, len(images))):
    bv_idx = images[i].get("bufferView")
    if bv_idx is None:
        print(f"Image {i}: no bufferView, skipping")
        continue
    
    bv = buffer_views[bv_idx]
    byte_offset = bv.get("byteOffset", 0)
    byte_length = bv.get("byteLength", 0)
    
    tex = data[bin_data_start + byte_offset : bin_data_start + byte_offset + byte_length]
    
    print(f"\nImage {i}: bufferView={bv_idx}, length={byte_length}")
    png_magic = b"\x89PNG"
    print(f"  PNG header valid: {tex[:4] == png_magic}")
    
    # Parse PNG chunks - concat all IDATs before decompressing
    pos = 8
    idat_chunks = []
    chunk_types = []
    ihdr_data = None
    while pos < len(tex) - 4:
        clen = struct.unpack_from(">I", tex, pos)[0]
        ctype = tex[pos + 4 : pos + 8]
        ctype_str = ctype.decode("ascii", "replace")
        chunk_types.append(ctype_str)
        if ctype == b"IHDR":
            ihdr_data = tex[pos + 8 : pos + 8 + clen]
        elif ctype == b"IDAT":
            idat_chunks.append(tex[pos + 8 : pos + 8 + clen])
        elif ctype == b"IEND":
            break
        pos += 12 + clen

    # Parse IHDR for dimensions
    if ihdr_data and len(ihdr_data) >= 8:
        width = struct.unpack_from(">I", ihdr_data, 0)[0]
        height = struct.unpack_from(">I", ihdr_data, 4)[0]
        bit_depth = ihdr_data[8]
        color_type = ihdr_data[9]
        print(f"  Dimensions: {width}x{height}, bit_depth={bit_depth}, color_type={color_type}")

    all_raw = bytearray()
    if idat_chunks:
        combined = b"".join(idat_chunks)
        try:
            raw = zlib.decompress(combined)
            all_raw.extend(raw)
        except Exception as e:
            print(f"  zlib decompress failed: {e}")

    print(f"  Chunks: {chunk_types}")
    print(f"  Decompressed: {len(all_raw)} bytes")
    
    if len(all_raw) > 0:
        # Sample the decompressed data
        # PNG raw data has filter bytes per row; for RGBA 1024x1024:
        # each row = 1 filter byte + 1024*4 = 4097 bytes
        unique_bytes = len(set(all_raw[: min(len(all_raw), 262144)]))
        print(f"  Unique byte values (first 256K): {unique_bytes}")
        
        if unique_bytes < 200:
            print("  >>> SOLID/NEAR-SOLID COLOUR - TEXTURE IS WRONG <<<")
        elif unique_bytes < 1000:
            print("  >> LOW VARIATION - mostly flat colour")
        else:
            print(f"  OK: Good colour variation")

print("\n--- Mesh attribute summary ---")
meshes = json_data.get("meshes", [])
for mi, mesh in enumerate(meshes[:3]):
    for pi, prim in enumerate(mesh.get("primitives", [])):
        attrs = prim.get("attributes", {})
        has_color = "COLOR_0" in attrs
        has_uv = "TEXCOORD_0" in attrs
        mat_idx = prim.get("material", -1)
        print(f"  Mesh[{mi}].prim[{pi}]: COLOR_0={has_color}, TEXCOORD_0={has_uv}, material={mat_idx}")
