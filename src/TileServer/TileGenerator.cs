using SharpGLTF.Schema2;
using SharpGLTF.Scenes;
using SharpGLTF.Geometry;
using SharpGLTF.Geometry.VertexTypes;
using SharpGLTF.Materials;
using System.Numerics;

namespace LibreRally.Maps.TileServer;

/// <summary>
/// Generates 3D Tiles (glTF + tileset.json) from classified point cloud meshes.
/// Uses SharpGLTF for glTF authoring with 3D Tiles extensions.
/// </summary>
public class TileGenerator
{
    /// <summary>
    /// Build a glTF model from classified mesh data with feature metadata.
    /// Uses 3D Tiles quadtree coordinates (zoom, x, y).
    /// </summary>
    public async Task<string> GenerateTileAsync(
        List<ClassifiedCluster> clusters,
        int zoom, int x, int y,
        string outputDir,
        CancellationToken ct = default)
    {
        // Validate quadtree coordinates — (0,0,0) means "whole world" which is invalid for a single tile
        if (zoom <= 0)
            throw new ArgumentException($"Invalid quadtree zoom: {zoom}. Must be >= 1.");

        // Resolve to absolute path from repo root to avoid bin-relative confusion
        if (!Path.IsPathRooted(outputDir))
            outputDir = RepoPaths.FromRoot(outputDir);

        Directory.CreateDirectory(outputDir);

        // Create scene builder
        var scene = new SceneBuilder();

        // Add each classified cluster as a separate mesh with metadata
        int featureId = 0;
        foreach (var cluster in clusters)
        {
            if (cluster.Vertices.Count < 3 || cluster.Faces.Count < 1)
                continue;

            var material = GetMaterialForFeatureType(cluster.FeatureType);

            var mesh = CreateMeshBuilder(cluster, material);
            scene.AddRigidMesh(mesh, Matrix4x4.Identity);

            featureId++;
        }

        // Build glTF
        var model = scene.ToGltf2();
        // 3D Tiles quadtree naming: {zoom}.{x}.{y}.glb
        var glbPath = Path.Combine(outputDir, $"{zoom}.{x}.{y}.glb");
        model.SaveGLB(glbPath);
        return glbPath;
    }

    /// <summary>
    /// Generate tileset.json for a collection of tiles.
    /// </summary>
    public string GenerateTilesetJson(
        List<TileInfo> tiles,
        double minLon, double minLat, double maxLon, double maxLat)
    {
        // Build a simple tileset.json
        var tileset = new TilesetRoot
        {
            asset = new TilesetAsset { version = "1.0" },
            geometricError = 500.0,
            root = new TilesetTile
            {
                boundingVolume = new TilesetBoundingVolume
                {
                    region = new double[]
                    {
                        DegToRad(minLon), DegToRad(minLat),  // west, south
                        DegToRad(maxLon), DegToRad(maxLat),  // east, north
                        -50, 300  // min/max height (meters)
                    }
                },
                geometricError = 200.0,
                refine = "ADD",
                content = new TilesetContent
                {
                    uri = $"{tiles[0].Zoom}.{tiles[0].X}.{tiles[0].Y}.glb"
                },
                children = tiles.Skip(1).Select(t => new TilesetTile
                {
                    boundingVolume = new TilesetBoundingVolume
                    {
                        region = new double[]
                        {
                            DegToRad(t.MinLon), DegToRad(t.MinLat),
                            DegToRad(t.MaxLon), DegToRad(t.MaxLat),
                            -50, 300
                        }
                    },
                    geometricError = 50.0,
                    content = new TilesetContent { uri = $"{t.Zoom}.{t.X}.{t.Y}.glb" }
                }).ToList()
            }
        };

        return System.Text.Json.JsonSerializer.Serialize(tileset,
            new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
    }

    // ── Helpers ──────────────────────────────────────────────────

    internal static MaterialBuilder GetMaterialForFeatureType(string featureType)
    {
        return featureType switch
        {
            "building" => new MaterialBuilder()
                .WithDoubleSide(true)
                .WithMetallicRoughnessShader()
                .WithChannelParam(KnownChannel.BaseColor, KnownProperty.RGBA, new Vector4(0.8f, 0.75f, 0.65f, 1)),
            "road" => new MaterialBuilder()
                .WithDoubleSide(true)
                .WithMetallicRoughnessShader()
                .WithChannelParam(KnownChannel.BaseColor, KnownProperty.RGBA, new Vector4(0.3f, 0.3f, 0.35f, 1)),
            "high_vegetation" => new MaterialBuilder()
                .WithDoubleSide(true)
                .WithMetallicRoughnessShader()
                .WithChannelParam(KnownChannel.BaseColor, KnownProperty.RGBA, new Vector4(0.2f, 0.55f, 0.2f, 1)),
            "low_vegetation" => new MaterialBuilder()
                .WithDoubleSide(true)
                .WithMetallicRoughnessShader()
                .WithChannelParam(KnownChannel.BaseColor, KnownProperty.RGBA, new Vector4(0.4f, 0.7f, 0.3f, 1)),
            "ground" => new MaterialBuilder()
                .WithDoubleSide(true)
                .WithMetallicRoughnessShader()
                .WithChannelParam(KnownChannel.BaseColor, KnownProperty.RGBA, new Vector4(0.45f, 0.4f, 0.35f, 1)),
            _ => new MaterialBuilder()
                .WithDoubleSide(true)
                .WithMetallicRoughnessShader()
                .WithChannelParam(KnownChannel.BaseColor, KnownProperty.RGBA, new Vector4(0.7f, 0.7f, 0.7f, 1)),
        };
    }

    private static IMeshBuilder<MaterialBuilder> CreateMeshBuilder(
        ClassifiedCluster cluster, MaterialBuilder material)
    {
        var mesh = new MeshBuilder<VertexPosition, VertexEmpty, VertexEmpty>("cluster");
        var prim = mesh.UsePrimitive(material);

        foreach (var face in cluster.Faces)
        {
            if (face.Length != 3) continue;
            var v0 = cluster.Vertices[face[0]];
            var v1 = cluster.Vertices[face[1]];
            var v2 = cluster.Vertices[face[2]];
            
            prim.AddTriangle(
                new VertexBuilder<VertexPosition, VertexEmpty, VertexEmpty>(
                    new VertexPosition((float)v0[0], (float)v0[1], (float)v0[2])),
                new VertexBuilder<VertexPosition, VertexEmpty, VertexEmpty>(
                    new VertexPosition((float)v1[0], (float)v1[1], (float)v1[2])),
                new VertexBuilder<VertexPosition, VertexEmpty, VertexEmpty>(
                    new VertexPosition((float)v2[0], (float)v2[1], (float)v2[2])));
        }
        return mesh;
    }

    private static double DegToRad(double deg) => deg * Math.PI / 180.0;
}

// ── Data Models ──────────────────────────────────────────────────

public class ClassifiedCluster
{
    public string FeatureType { get; set; } = "unknown";
    public int AsprsClass { get; set; }
    public List<double[]> Vertices { get; set; } = [];
    public List<int[]> Faces { get; set; } = [];
    public string? OsmName { get; set; }
    public string? OsmTags { get; set; }
}

public class TileInfo
{
    public int Zoom { get; set; }
    public int X { get; set; }
    public int Y { get; set; }
    public double MinLon { get; set; }
    public double MinLat { get; set; }
    public double MaxLon { get; set; }
    public double MaxLat { get; set; }
    public int FeatureCount { get; set; }
}

// ── 3D Tiles JSON Models ─────────────────────────────────────────

public class TilesetRoot
{
    public TilesetAsset asset { get; set; } = new();
    public double geometricError { get; set; }
    public TilesetTile root { get; set; } = new();
}

public class TilesetAsset
{
    public string version { get; set; } = "1.0";
}

public class TilesetTile
{
    public TilesetBoundingVolume boundingVolume { get; set; } = new();
    public double geometricError { get; set; }
    public string? refine { get; set; }
    public TilesetContent? content { get; set; }
    public List<TilesetTile>? children { get; set; }
}

public class TilesetBoundingVolume
{
    public double[]? region { get; set; }
    public double[]? box { get; set; }
}

public class TilesetContent
{
    public string uri { get; set; } = "";
}
