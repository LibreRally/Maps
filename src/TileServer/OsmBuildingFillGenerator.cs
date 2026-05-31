using System.Numerics;
using System.Text.Json;
using Microsoft.EntityFrameworkCore;
using Npgsql;
using SharpGLTF.Geometry;
using SharpGLTF.Geometry.VertexTypes;
using SharpGLTF.Materials;
using SharpGLTF.Schema2;
using SharpGLTF.Scenes;

namespace LibreRally.Maps.TileServer;

internal sealed class OsmBuildingFillGenerator
{
    private const double MinimumBuildingHeight = 2.4;
    private const double SegmentCoverageSkipRatio = 0.75;
    private const double RoofOnlyCoverageRatio = 0.35;
    private const double OversizedSegmentCoverageRatio = 2.0;
    private const float LowConfidenceCoverageMatchScore = 0.12f;
    private const double NearbyStandaloneFootprintBuffer = 10.0;
    private const double NearbyStandaloneSegmentBuffer = 8.0;
    private const double DefaultStandaloneBuildingHeight = 6.0;

    public async Task<OsmFillApplicationResult> ApplyAsync(
        MapsDbContext db,
        string glbPath,
        global::PipelineResultPayload pipelineResult,
        IReadOnlyDictionary<int, global::SegmentOsmMatchPayload> segmentMatches,
        CancellationToken cancellationToken = default)
    {
        if (pipelineResult.Segments is null || pipelineResult.Segments.Length == 0)
            return OsmFillApplicationResult.Empty;

        if (pipelineResult.SceneOrigin.Length < 3 || !File.Exists(glbPath))
            return OsmFillApplicationResult.Empty;

        SceneBuilder[]? sceneBuilders = null;
        var filledSegmentIds = new HashSet<int>();
        var addedOsmIds = new HashSet<long>();
        var addedMeshCount = 0;
        long addedVertexCount = 0;
        long addedFaceCount = 0;

        foreach (var segment in pipelineResult.Segments)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var match = segmentMatches.GetValueOrDefault(segment.LocalId);
            if (!ShouldGenerateFill(segment, match))
                continue;

            var footprint = await LoadFootprintAsync(db, match!.OsmId!.Value, cancellationToken);
            if (footprint is null)
                continue;

            var fillMesh = CreateFillMesh(segment, match, footprint, pipelineResult.SceneOrigin);
            if (fillMesh is null)
                continue;

            sceneBuilders ??= SceneBuilder.CreateFrom(ModelRoot.Load(glbPath));
            if (sceneBuilders.Length == 0)
                sceneBuilders = [new SceneBuilder()];

            var node = new NodeBuilder($"segment-{segment.LocalId:0000}-building-osmfill");
            sceneBuilders[0].AddRigidMesh(fillMesh.Mesh, node);
            filledSegmentIds.Add(segment.LocalId);
            addedOsmIds.Add(footprint.OsmId);
            addedMeshCount++;
            addedVertexCount += fillMesh.VertexCount;
            addedFaceCount += fillMesh.FaceCount;
        }

        var scanExtent = TryGetScanExtent(pipelineResult.Segments);
        if (scanExtent is not null)
        {
            var standaloneFootprints = await LoadNearbyFootprintsAsync(
                db,
                scanExtent.Value,
                addedOsmIds,
                cancellationToken);

            foreach (var footprint in standaloneFootprints)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var extrusion = TryCreateStandaloneExtrusion(pipelineResult.Segments, footprint);
                if (extrusion is null)
                    continue;

                var fillMesh = CreateFillMesh(
                    meshName: $"osm-{footprint.OsmId}-building-osmfill",
                    footprint,
                    pipelineResult.SceneOrigin,
                    extrusion.Value.BaseZ,
                    extrusion.Value.TopZ,
                    coverageRatio: 0d);
                if (fillMesh is null)
                    continue;

                sceneBuilders ??= SceneBuilder.CreateFrom(ModelRoot.Load(glbPath));
                if (sceneBuilders.Length == 0)
                    sceneBuilders = [new SceneBuilder()];

                var node = new NodeBuilder($"osm-building-{footprint.OsmId}");
                sceneBuilders[0].AddRigidMesh(fillMesh.Mesh, node);
                addedOsmIds.Add(footprint.OsmId);
                addedMeshCount++;
                addedVertexCount += fillMesh.VertexCount;
                addedFaceCount += fillMesh.FaceCount;
            }
        }

        if (sceneBuilders is null || addedMeshCount == 0)
            return OsmFillApplicationResult.Empty;

        var mergedModel = SceneBuilder.ToGltf2(sceneBuilders, new SceneBuilderSchema2Settings());
        mergedModel.SaveGLB(glbPath);

        return new OsmFillApplicationResult(filledSegmentIds, addedMeshCount, addedVertexCount, addedFaceCount);
    }

    private static bool ShouldGenerateFill(global::PipelineSegmentPayload segment, global::SegmentOsmMatchPayload? match)
    {
        if (match?.OsmId is null)
            return false;

        if (!string.Equals(match.MatchStatus, "matched", StringComparison.OrdinalIgnoreCase))
            return false;

        return BuildingSegmentHeuristics.CanPromoteToBuilding(segment, match);
    }

    private static async Task<OsmBuildingFootprint?> LoadFootprintAsync(
        MapsDbContext db,
        long osmId,
        CancellationToken cancellationToken)
    {
        var connection = db.Database.GetDbConnection();
        if (connection.State != System.Data.ConnectionState.Open)
            await connection.OpenAsync(cancellationToken);

        await using var command = connection.CreateCommand();
        command.CommandText = """
            SELECT osm_id,
                   COALESCE(name, '') AS name,
                   COALESCE(NULLIF(building, ''), NULLIF(amenity, ''), NULLIF(shop, ''), 'building') AS feature_type,
                   ST_Area(ST_Transform(way, 28355)) AS area_m2,
                   ST_AsGeoJSON(ST_GeometryN(ST_Multi(ST_Transform(way, 28355)), 1)) AS geometry_json
            FROM osm.planet_osm_polygon
            WHERE osm_id = $1
              AND building IS NOT NULL
            LIMIT 1
            """;
        command.Parameters.Add(new NpgsqlParameter<long> { Value = osmId });

        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        if (!await reader.ReadAsync(cancellationToken))
            return null;

        var area = reader.IsDBNull(3) ? 0d : reader.GetDouble(3);
        var geometryJson = reader.IsDBNull(4) ? null : reader.GetString(4);
        if (string.IsNullOrWhiteSpace(geometryJson))
            return null;

        return ParseFootprintGeometry(
            reader.GetInt64(0),
            reader.IsDBNull(1) ? null : reader.GetString(1),
            reader.IsDBNull(2) ? null : reader.GetString(2),
            area,
            geometryJson);
    }

    private static OsmBuildingFootprint? ParseFootprintGeometry(
        long osmId,
        string? name,
        string? featureType,
        double area,
        string geometryJson)
    {
        using var document = JsonDocument.Parse(geometryJson);
        var root = document.RootElement;
        if (!root.TryGetProperty("type", out var typeElement) || !root.TryGetProperty("coordinates", out var coordinatesElement))
            return null;

        JsonElement ringElement;
        var type = typeElement.GetString();
        if (string.Equals(type, "Polygon", StringComparison.OrdinalIgnoreCase))
        {
            if (coordinatesElement.GetArrayLength() == 0)
                return null;

            ringElement = coordinatesElement[0];
        }
        else if (string.Equals(type, "MultiPolygon", StringComparison.OrdinalIgnoreCase))
        {
            if (coordinatesElement.GetArrayLength() == 0 || coordinatesElement[0].GetArrayLength() == 0)
                return null;

            ringElement = coordinatesElement[0][0];
        }
        else
        {
            return null;
        }

        var ring = new List<Vector2>();
        foreach (var pointElement in ringElement.EnumerateArray())
        {
            if (pointElement.GetArrayLength() < 2)
                continue;

            ring.Add(new Vector2(
                (float)pointElement[0].GetDouble(),
                (float)pointElement[1].GetDouble()));
        }

        if (ring.Count >= 2 && Vector2.DistanceSquared(ring[0], ring[^1]) < 1e-6f)
            ring.RemoveAt(ring.Count - 1);

        if (ring.Count < 3)
            return null;

        return new OsmBuildingFootprint(osmId, name, featureType, area, ring);
    }

    private static OsmFillMesh? CreateFillMesh(
        global::PipelineSegmentPayload segment,
        global::SegmentOsmMatchPayload match,
        OsmBuildingFootprint footprint,
        double[] sceneOrigin)
    {
        var footprintArea = footprint.Area;
        if (footprintArea < 1.0)
            return null;

        var boundsWidth = segment.BoundsMax.ElementAtOrDefault(0) - segment.BoundsMin.ElementAtOrDefault(0);
        var boundsLength = segment.BoundsMax.ElementAtOrDefault(1) - segment.BoundsMin.ElementAtOrDefault(1);
        var segmentFootprintArea = Math.Max(boundsWidth, 0d) * Math.Max(boundsLength, 0d);
        var rawCoverageRatio = segmentFootprintArea > 0d ? segmentFootprintArea / footprintArea : 0d;
        var isOversizedMergedSegment = rawCoverageRatio >= OversizedSegmentCoverageRatio;
        var hasLowConfidenceCoverage = (match.MatchScore ?? 0f) < LowConfidenceCoverageMatchScore;
        var coverageRatio = (isOversizedMergedSegment || hasLowConfidenceCoverage)
            ? 0d
            : Math.Clamp(rawCoverageRatio, 0d, 1d);
        if (coverageRatio >= SegmentCoverageSkipRatio)
            return null;

        var baseZ = segment.BoundsMin.ElementAtOrDefault(2);
        var topZ = Math.Max(segment.BoundsMax.ElementAtOrDefault(2), baseZ + MinimumBuildingHeight);
        return CreateFillMesh(
            meshName: $"segment-{segment.LocalId:0000}-building-osmfill",
            footprint,
            sceneOrigin,
            baseZ,
            topZ,
            coverageRatio);
    }

    private static OsmFillMesh? CreateFillMesh(
        string meshName,
        OsmBuildingFootprint footprint,
        double[] sceneOrigin,
        double baseZ,
        double topZ,
        double coverageRatio)
    {
        var ring = footprint.ExteriorRing;
        if (ring.Count < 3)
            return null;

        if (topZ - baseZ < 0.5)
            return null;

        var localRing = ring
            .Select(point => new Vector2(
                point.X - (float)sceneOrigin[0],
                point.Y - (float)sceneOrigin[1]))
            .ToArray();
        var localTopZ = (float)(topZ - sceneOrigin[2]);
        var localBaseZ = (float)(baseZ - sceneOrigin[2]);
        var localTop = localRing
            .Select(point => new Vector3(point.X, point.Y, localTopZ))
            .ToArray();
        var localBottom = localRing
            .Select(point => new Vector3(point.X, point.Y, localBaseZ))
            .ToArray();

        var centroid = ComputeCentroid(localRing);
        var localRoofCenter = new Vector3(centroid.X, centroid.Y, localTopZ);

        var material = TileGenerator.GetMaterialForFeatureType("building");
        var mesh = new MeshBuilder<VertexPosition, VertexEmpty, VertexEmpty>(meshName);
        var primitive = mesh.UsePrimitive(material);

        var faceCount = 0;
        var includeWalls = coverageRatio < RoofOnlyCoverageRatio;
        if (includeWalls)
        {
            for (var index = 0; index < ring.Count; index++)
            {
                var next = (index + 1) % ring.Count;
                AddTriangle(primitive, localBottom[index], localBottom[next], localTop[next]);
                AddTriangle(primitive, localBottom[index], localTop[next], localTop[index]);
                faceCount += 2;
            }
        }

        var isCounterClockwise = ComputeSignedArea(localRing) >= 0d;
        for (var index = 0; index < ring.Count; index++)
        {
            var next = (index + 1) % ring.Count;
            if (isCounterClockwise)
            {
                AddTriangle(primitive, localRoofCenter, localTop[index], localTop[next]);
            }
            else
            {
                AddTriangle(primitive, localRoofCenter, localTop[next], localTop[index]);
            }

            faceCount++;
        }

        return new OsmFillMesh(mesh, ring.Count * 2 + 1, faceCount);
    }

    private static ScanExtent? TryGetScanExtent(IReadOnlyCollection<global::PipelineSegmentPayload> segments)
    {
        var usableSegments = segments
            .Where(segment =>
                segment.BoundsMin.Length >= 2 &&
                segment.BoundsMax.Length >= 2 &&
                segment.BoundsMax[0] > segment.BoundsMin[0] &&
                segment.BoundsMax[1] > segment.BoundsMin[1])
            .ToArray();

        if (usableSegments.Length == 0)
            return null;

        return new ScanExtent(
            usableSegments.Min(segment => segment.BoundsMin[0]),
            usableSegments.Min(segment => segment.BoundsMin[1]),
            usableSegments.Max(segment => segment.BoundsMax[0]),
            usableSegments.Max(segment => segment.BoundsMax[1]));
    }

    private static async Task<List<OsmBuildingFootprint>> LoadNearbyFootprintsAsync(
        MapsDbContext db,
        ScanExtent extent,
        IReadOnlySet<long> excludedOsmIds,
        CancellationToken cancellationToken)
    {
        var results = new List<OsmBuildingFootprint>();
        var connection = db.Database.GetDbConnection();
        if (connection.State != System.Data.ConnectionState.Open)
            await connection.OpenAsync(cancellationToken);

        await using var command = connection.CreateCommand();
        command.CommandText = """
            SELECT osm_id,
                   COALESCE(name, '') AS name,
                   COALESCE(NULLIF(building, ''), NULLIF(amenity, ''), NULLIF(shop, ''), 'building') AS feature_type,
                   ST_Area(ST_Transform(way, 28355)) AS area_m2,
                   ST_AsGeoJSON(ST_GeometryN(ST_Multi(ST_Transform(way, 28355)), 1)) AS geometry_json
            FROM osm.planet_osm_polygon
            WHERE building IS NOT NULL
              AND way && ST_Transform(ST_Buffer(ST_MakeEnvelope($1, $2, $3, $4, 28355), $5), 3857)
            ORDER BY area_m2 DESC
            LIMIT 250
            """;
        command.Parameters.Add(new NpgsqlParameter<double> { Value = extent.MinX });
        command.Parameters.Add(new NpgsqlParameter<double> { Value = extent.MinY });
        command.Parameters.Add(new NpgsqlParameter<double> { Value = extent.MaxX });
        command.Parameters.Add(new NpgsqlParameter<double> { Value = extent.MaxY });
        command.Parameters.Add(new NpgsqlParameter<double> { Value = NearbyStandaloneFootprintBuffer });

        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        while (await reader.ReadAsync(cancellationToken))
        {
            var osmId = reader.GetInt64(0);
            if (excludedOsmIds.Contains(osmId))
                continue;

            var area = reader.IsDBNull(3) ? 0d : reader.GetDouble(3);
            var geometryJson = reader.IsDBNull(4) ? null : reader.GetString(4);
            if (string.IsNullOrWhiteSpace(geometryJson))
                continue;

            var footprint = ParseFootprintGeometry(
                osmId,
                reader.IsDBNull(1) ? null : reader.GetString(1),
                reader.IsDBNull(2) ? null : reader.GetString(2),
                area,
                geometryJson);
            if (footprint is not null)
                results.Add(footprint);
        }

        return results;
    }

    private static BuildingExtrusion? TryCreateStandaloneExtrusion(
        IReadOnlyCollection<global::PipelineSegmentPayload> segments,
        OsmBuildingFootprint footprint)
    {
        var footprintBounds = GetBounds(footprint.ExteriorRing);
        var candidates = segments
            .Where(segment => !string.Equals(segment.FeatureType, "noise", StringComparison.OrdinalIgnoreCase))
            .Where(segment => SegmentIntersectsBounds(segment, footprintBounds, NearbyStandaloneSegmentBuffer))
            .ToArray();

        if (candidates.Length == 0)
            return null;

        var baseCandidates = candidates
            .Where(segment =>
                string.Equals(segment.FeatureType, "ground", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(segment.FeatureType, "road", StringComparison.OrdinalIgnoreCase))
            .Select(segment => segment.BoundsMin.ElementAtOrDefault(2))
            .ToArray();
        var baseZ = baseCandidates.Length > 0
            ? baseCandidates.Min()
            : candidates.Min(segment => segment.BoundsMin.ElementAtOrDefault(2));

        var topCandidates = candidates
            .Where(segment => !string.Equals(segment.FeatureType, "ground", StringComparison.OrdinalIgnoreCase))
            .Where(segment => !string.Equals(segment.FeatureType, "road", StringComparison.OrdinalIgnoreCase))
            .Select(segment => segment.BoundsMax.ElementAtOrDefault(2))
            .ToArray();
        var topZ = topCandidates.Length > 0
            ? Math.Max(topCandidates.Max(), baseZ + DefaultStandaloneBuildingHeight)
            : baseZ + DefaultStandaloneBuildingHeight;

        return new BuildingExtrusion(baseZ, topZ);
    }

    private static ScanExtent GetBounds(IReadOnlyList<Vector2> ring)
    {
        return new ScanExtent(
            ring.Min(point => point.X),
            ring.Min(point => point.Y),
            ring.Max(point => point.X),
            ring.Max(point => point.Y));
    }

    private static bool SegmentIntersectsBounds(global::PipelineSegmentPayload segment, ScanExtent bounds, double buffer)
    {
        if (segment.BoundsMin.Length < 2 || segment.BoundsMax.Length < 2)
            return false;

        return segment.BoundsMax[0] >= bounds.MinX - buffer &&
               segment.BoundsMin[0] <= bounds.MaxX + buffer &&
               segment.BoundsMax[1] >= bounds.MinY - buffer &&
               segment.BoundsMin[1] <= bounds.MaxY + buffer;
    }

    private static void AddTriangle(
        PrimitiveBuilder<MaterialBuilder, VertexPosition, VertexEmpty, VertexEmpty> primitive,
        Vector3 v0,
        Vector3 v1,
        Vector3 v2)
    {
        primitive.AddTriangle(
            new VertexBuilder<VertexPosition, VertexEmpty, VertexEmpty>(new VertexPosition(v0)),
            new VertexBuilder<VertexPosition, VertexEmpty, VertexEmpty>(new VertexPosition(v1)),
            new VertexBuilder<VertexPosition, VertexEmpty, VertexEmpty>(new VertexPosition(v2)));
    }

    private static double ComputeSignedArea(IReadOnlyList<Vector2> ring)
    {
        double area = 0;
        for (var index = 0; index < ring.Count; index++)
        {
            var next = (index + 1) % ring.Count;
            area += (ring[index].X * ring[next].Y) - (ring[next].X * ring[index].Y);
        }

        return area * 0.5d;
    }

    private static Vector2 ComputeCentroid(IReadOnlyList<Vector2> ring)
    {
        var signedArea = ComputeSignedArea(ring);
        if (Math.Abs(signedArea) < 1e-6d)
        {
            var averageX = ring.Average(point => point.X);
            var averageY = ring.Average(point => point.Y);
            return new Vector2((float)averageX, (float)averageY);
        }

        double cx = 0;
        double cy = 0;
        for (var index = 0; index < ring.Count; index++)
        {
            var next = (index + 1) % ring.Count;
            var factor = (ring[index].X * ring[next].Y) - (ring[next].X * ring[index].Y);
            cx += (ring[index].X + ring[next].X) * factor;
            cy += (ring[index].Y + ring[next].Y) * factor;
        }

        var scale = 1d / (6d * signedArea);
        return new Vector2((float)(cx * scale), (float)(cy * scale));
    }

    private sealed record OsmBuildingFootprint(
        long OsmId,
        string? Name,
        string? FeatureType,
        double Area,
        IReadOnlyList<Vector2> ExteriorRing);

    private sealed record OsmFillMesh(
        IMeshBuilder<MaterialBuilder> Mesh,
        int VertexCount,
        int FaceCount);

    private readonly record struct ScanExtent(double MinX, double MinY, double MaxX, double MaxY);

    private readonly record struct BuildingExtrusion(double BaseZ, double TopZ);
}

internal sealed record OsmFillApplicationResult(
    IReadOnlySet<int> FilledSegmentIds,
    int AddedMeshCount,
    long AddedVertexCount,
    long AddedFaceCount)
{
    public static OsmFillApplicationResult Empty { get; } = new(new HashSet<int>(), 0, 0, 0);
}
