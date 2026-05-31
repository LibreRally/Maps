"""Inspect a GLB file for colour/texture/material data."""
import json
import struct
import sys

if len(sys.argv) < 2:
    print("Usage: python inspect_glb.py <path/to/file.glb>")
    sys.exit(1)
path = sys.argv[1]

with open(path, "rb") as f:
    data = f.read()

magic = struct.unpack_from("<I", data, 0)[0]
version = struct.unpack_from("<I", data, 4)[0]
length = struct.unpack_from("<I", data, 8)[0]
print(f"GLB: magic={hex(magic)}, version={version}, total_length={length}")

# Parse GLB chunks
offset = 12
json_data = None
while offset < len(data):
    chunk_length = struct.unpack_from("<I", data, offset)[0]
    chunk_type = struct.unpack_from("<I", data, offset + 4)[0]
    chunk_start = offset + 8
    chunk_type_str = "JSON" if chunk_type == 0x4E4F534A else ("BIN\x00" if chunk_type == 0x004E4942 else hex(chunk_type))
    print(f"  Chunk at {offset}: type={chunk_type_str}, length={chunk_length}")
    if chunk_type == 0x4E4F534A:  # JSON
        json_data = json.loads(data[chunk_start:chunk_start + chunk_length].decode("utf-8", errors="replace"))
    offset = chunk_start + chunk_length

if json_data is None:
    print("ERROR: No JSON chunk found")
    sys.exit(1)

materials = json_data.get("materials", [])
print(f"\nMaterials: {len(materials)}")
for i, mat in enumerate(materials):
    pbr = mat.get("pbrMetallicRoughness", {})
    has_bct = "baseColorTexture" in pbr
    bcf = pbr.get("baseColorFactor", [1, 1, 1, 1])
    has_nt = "normalTexture" in mat
    has_emissive = "emissiveTexture" in mat
    print(f"  Mat[{i}]: baseColorTexture={has_bct}, baseColorFactor={bcf}, normalTexture={has_nt}, emissiveTexture={has_emissive}")

meshes = json_data.get("meshes", [])
print(f"\nMeshes: {len(meshes)}")
has_any_color = False
has_any_uv = False
for mi, mesh in enumerate(meshes):
    for pi, prim in enumerate(mesh.get("primitives", [])):
        attrs = prim.get("attributes", {})
        color0 = "COLOR_0" in attrs
        uv0 = "TEXCOORD_0" in attrs
        mat_idx = prim.get("material", -1)
        mode = prim.get("mode", 4)
        if color0:
            has_any_color = True
        if uv0:
            has_any_uv = True
        print(f"  Mesh[{mi}].prim[{pi}]: COLOR_0={color0}, TEXCOORD_0={uv0}, material={mat_idx}, mode={mode}")

print(f"\nAny COLOR_0: {has_any_color}")
print(f"Any TEXCOORD_0: {has_any_uv}")

images = json_data.get("images", [])
print(f"\nImages: {len(images)}")
for ii, img in enumerate(images):
    print(f"  Image[{ii}]: mimeType={img.get('mimeType', 'N/A')}, bufferView={img.get('bufferView', 'N/A')}")

textures = json_data.get("textures", [])
print(f"Textures: {len(textures)}")
for ti, tex in enumerate(textures):
    print(f"  Texture[{ti}]: source={tex.get('source', -1)}, sampler={tex.get('sampler', -1)}")

# Check accessors for COLOR data
accessors = json_data.get("accessors", [])
color_accessors = []
for ai, acc in enumerate(accessors):
    name = acc.get("name", "")
    atype = acc.get("type", "")
    if "COLOR" in name.upper() or atype == "VEC3":
        bv = acc.get("bufferView", -1)
        count = acc.get("count", 0)
        ct = acc.get("componentType", 0)
        color_accessors.append((ai, name, count, ct, bv))

print(f"\nAccessor summary:")
for ca in color_accessors:
    print(f"  Accessor[{ca[0]}]: name={ca[1]}, count={ca[2]}, componentType={ca[3]}, bufferView={ca[4]}")

if not has_any_color and not has_any_uv:
    print("\n*** NO COLOUR DATA IN GLB ***")
    print("The GLB contains only geometry (positions + normals), no colours or textures.")
    print("This means the pipeline didn't produce colour data, or the GLB export didn't include it.")
