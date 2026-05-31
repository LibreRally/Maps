using System.Collections.Concurrent;
using System.Data;
using System.Globalization;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Hosting;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage;
using Npgsql;
using NpgsqlTypes;
using LibreRally.Maps.TileServer;

var builder = WebApplication.CreateBuilder(args);

// Aspire service defaults (health checks, telemetry, resilience)
builder.AddServiceDefaults();

// Allow large file uploads (LAS tiles can be 100MB+)
builder.WebHost.ConfigureKestrel(o => o.Limits.MaxRequestBodySize = 500_000_000);

// PostgreSQL via Aspire — enable snake_case naming convention
builder.AddNpgsqlDbContext<MapsDbContext>("mapsdb",
    configureDbContextOptions: options => options.UseSnakeCaseNamingConvention());
#pragma warning disable EXTEXP0001
builder.Services.AddHttpClient("mlservice", client =>
{
    client.BaseAddress = new Uri("http://mlservice");
    client.Timeout = TimeSpan.FromHours(2);
})
    .RemoveAllResilienceHandlers();
#pragma warning restore EXTEXP0001

var app = builder.Build();
var httpClientFactory = app.Services.GetRequiredService<IHttpClientFactory>();
var osmBuildingFillGenerator = new OsmBuildingFillGenerator();

// Auto-initialize database schema (embedded SQL, safe to run multiple times)
await DbInitializer.InitializeAsync(app);

// Serve generated tile files from data/tiles/
var tilesDir = RepoPaths.TilesData;
if (Directory.Exists(tilesDir))
{
    app.UseStaticFiles(new StaticFileOptions
    {
        FileProvider = new Microsoft.Extensions.FileProviders.PhysicalFileProvider(
            Path.GetFullPath(tilesDir)),
        RequestPath = "/data/tiles",
        ServeUnknownFileTypes = true,
    });
}

app.MapDefaultEndpoints();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

// ── Health ──────────────────────────────────────────────────────
app.MapGet("/", () => Results.Ok(new { service = "LibreRally.Maps.TileServer", status = "healthy" }));

var pipelineJobs = new ConcurrentDictionary<Guid, PipelineJob>();
var importJobs = new ConcurrentDictionary<Guid, ImportJob>();
var pipelineJsonOptions = new JsonSerializerOptions(JsonSerializerDefaults.Web);
const double ScaniverseDefaultYawDegrees = -90d;
const string VicmapFeatureServerQueryUrl = "https://services-ap1.arcgis.com/P744lA0wf4LlBZ84/arcgis/rest/services/Vicmap_Property/FeatureServer/0/query";
const double VicmapQueryBufferMeters = 80d;
const double VicmapYawConfidenceThresholdDegrees = 10d;
const int VicmapQueryFeatureLimit = 200;
const int VicmapOverlayFetchTimeoutSeconds = 15;
Directory.CreateDirectory(RepoPaths.RawData);
Directory.CreateDirectory(RepoPaths.TilesData);
Directory.CreateDirectory(RepoPaths.PipelineInputData);
Directory.CreateDirectory(RepoPaths.ModelsData);

// ── Tile Catalog ────────────────────────────────────────────────
app.MapGet("/api/tiles/catalog", async (MapsDbContext db) =>
{
    var tiles = await db.TileCatalog
        .OrderByDescending(t => t.CreatedAt)
        .Take(50)
        .Select(t => new
        {
            t.Id,
            t.TileX,
            t.TileY,
            t.ZoomLevel,
            t.FeatureTypes,
            t.MeshCount,
            t.SegmentCount,
            t.TilesetPath,
            t.FileSizeBytes,
            t.VertexCount,
            t.SourceTileIds,
            t.SourceImportIds,
            t.CreatedAt,
            t.UpdatedAt
        })
        .ToListAsync();
    return Results.Ok(tiles);
});

app.MapGet("/api/tiles/{id:guid}/pointcloud", async (Guid id, MapsDbContext db) =>
{
    var tile = await db.TileCatalog.FirstOrDefaultAsync(t => t.Id == id);
    if (tile is null)
        return Results.NotFound(new { error = "Tile not found." });

    var previewScript = Path.Combine(RepoPaths.MlService, "pointcloud_preview.py");
    if (!File.Exists(previewScript))
        return Results.Problem("Point cloud preview script is missing.");

    // Resolve container name for docker exec
    var containerName = await GetMlServiceContainerNameAsync();
    if (string.IsNullOrEmpty(containerName))
        return Results.Problem("ML service container is not running.");

    string arguments;
    string? csvPath = null;
    if (!string.IsNullOrWhiteSpace(tile.SegmentsPath))
    {
        var segmentPath = Path.IsPathRooted(tile.SegmentsPath)
            ? tile.SegmentsPath
            : RepoPaths.FromRoot(tile.SegmentsPath);

        if (File.Exists(segmentPath))
        {
            var containerSegPath = RepoPaths.ToContainerPath(segmentPath);
            arguments = $"--segments \"{containerSegPath}\" --max-points 100000";
            goto RunPreview;
        }
    }

    var sourceImportIds = tile.SourceImportIds;
    if (sourceImportIds is null || sourceImportIds.Length == 0)
        return Results.NotFound(new { error = "No source imports are linked to this generated tile yet. Reprocess the active imports to regenerate the tile with point-bucket linkage." });

    csvPath = await BuildPointBucketCsvAsync(sourceImportIds, app.Services.GetRequiredService<IServiceScopeFactory>());
    if (csvPath is null)
        return Results.NotFound(new { error = "No point data is available for the imports linked to this generated tile." });

    var containerCsvPath = RepoPaths.ToContainerPath(csvPath);
    arguments = $"--csv \"{containerCsvPath}\" --max-points 100000";

RunPreview:
    // Run inside Docker container (has numpy, laspy, etc.)
    var psi = new System.Diagnostics.ProcessStartInfo
    {
        FileName = "docker",
        Arguments = $"exec -i {containerName} python \"{previewScript}\" {arguments}",
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true,
    };

    using var process = System.Diagnostics.Process.Start(psi);
    if (process is null)
        return Results.Problem("Failed to start point cloud preview process.");

    var stdoutTask = process.StandardOutput.ReadToEndAsync();
    var stderrTask = process.StandardError.ReadToEndAsync();
    await process.WaitForExitAsync();

    var stdout = await stdoutTask;
    var stderr = await stderrTask;

    if (csvPath is not null && File.Exists(csvPath))
        File.Delete(csvPath);

    if (process.ExitCode != 0)
        return Results.Problem($"Point cloud preview failed: {stderr}".Trim());

    var preview = JsonSerializer.Deserialize<PointCloudPreviewPayload>(stdout, pipelineJsonOptions);
    if (preview is null)
        return Results.Problem("Point cloud preview returned no data.");

    return Results.Ok(preview);
});

app.MapGet("/api/tiles/{id:guid}/alignment", async (Guid id, MapsDbContext db) =>
{
    var tile = await db.TileCatalog.FirstOrDefaultAsync(t => t.Id == id);
    if (tile is null)
        return Results.NotFound(new { error = "Tile not found." });

    if (!tile.TileX.HasValue || !tile.TileY.HasValue || !tile.ZoomLevel.HasValue)
        return Results.Problem("Tile is missing quadtree coordinates.");

    var (centerLat, centerLon) = GetTileCenter(tile.ZoomLevel.Value, tile.TileX.Value, tile.TileY.Value);
    var (west, south, east, north) = GetTileBounds(tile.ZoomLevel.Value, tile.TileX.Value, tile.TileY.Value);
    var tileBoundsGeoJson = CreateTileBoundsGeoJson(tile.ZoomLevel.Value, tile.TileX.Value, tile.TileY.Value);
    var alignment = await ResolveTileAlignmentMetadataAsync(db, tile);
    var alignmentOsmSearchBufferMeters = Math.Clamp(
        Math.Sqrt(Math.Pow(alignment?.OffsetX ?? 0d, 2) + Math.Pow(alignment?.OffsetY ?? 0d, 2)) + 60d,
        60d,
        250d);

    var segmentPayloads = new List<TileAlignmentSegmentPayload>();
    double? scanMinX = null;
    double? scanMinY = null;
    double? scanMaxX = null;
    double? scanMaxY = null;
    var connection = db.Database.GetDbConnection();
    if (connection.State != ConnectionState.Open)
        await connection.OpenAsync();

    await using (var segmentCommand = connection.CreateCommand())
    {
        segmentCommand.CommandText = """
            SELECT local_segment_id,
                   predicted_label,
                   geometry_source,
                   has_osm_fill,
                   osm_identifier,
                   osm_match_status,
                   CASE
                       WHEN centroid_x IS NOT NULL AND centroid_y IS NOT NULL
                       THEN ST_Y(ST_Transform(ST_SetSRID(ST_MakePoint(centroid_x, centroid_y), 28355), 4326))
                       ELSE NULL
                   END AS centroid_lat,
                   CASE
                       WHEN centroid_x IS NOT NULL AND centroid_y IS NOT NULL
                       THEN ST_X(ST_Transform(ST_SetSRID(ST_MakePoint(centroid_x, centroid_y), 28355), 4326))
                       ELSE NULL
                   END AS centroid_lon,
                   CASE
                       WHEN bounds_min_x IS NOT NULL AND bounds_min_y IS NOT NULL AND bounds_max_x IS NOT NULL AND bounds_max_y IS NOT NULL
                       THEN ST_AsGeoJSON(
                           ST_Transform(
                               ST_SetSRID(ST_MakeEnvelope(bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y), 28355),
                               4326))
                       ELSE NULL
                   END AS bounds_geojson,
                  bounds_min_x,
                  bounds_min_y,
                  bounds_max_x,
                  bounds_max_y
            FROM tiles.segments
            WHERE tile_id = $1
            ORDER BY local_segment_id
            """;
        segmentCommand.Parameters.Add(new NpgsqlParameter<Guid> { Value = tile.Id });

        await using var reader = await segmentCommand.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            segmentPayloads.Add(new TileAlignmentSegmentPayload
            {
                LocalSegmentId = reader.GetInt32(0),
                Label = reader.IsDBNull(1) ? string.Empty : reader.GetString(1),
                GeometrySource = reader.IsDBNull(2) ? "scan" : reader.GetString(2),
                HasOsmFill = !reader.IsDBNull(3) && reader.GetBoolean(3),
                OsmIdentifier = reader.IsDBNull(4) ? null : reader.GetInt64(4),
                OsmMatchStatus = reader.IsDBNull(5) ? "unmatched" : reader.GetString(5),
                CentroidLat = reader.IsDBNull(6) ? null : reader.GetDouble(6),
                CentroidLon = reader.IsDBNull(7) ? null : reader.GetDouble(7),
                BoundsGeoJson = reader.IsDBNull(8) ? null : reader.GetString(8),
            });

            if (!reader.IsDBNull(9) && !reader.IsDBNull(10) && !reader.IsDBNull(11) && !reader.IsDBNull(12))
            {
                var boundsMinX = reader.GetDouble(9);
                var boundsMinY = reader.GetDouble(10);
                var boundsMaxX = reader.GetDouble(11);
                var boundsMaxY = reader.GetDouble(12);
                scanMinX = !scanMinX.HasValue ? boundsMinX : Math.Min(scanMinX.Value, boundsMinX);
                scanMinY = !scanMinY.HasValue ? boundsMinY : Math.Min(scanMinY.Value, boundsMinY);
                scanMaxX = !scanMaxX.HasValue ? boundsMaxX : Math.Max(scanMaxX.Value, boundsMaxX);
                scanMaxY = !scanMaxY.HasValue ? boundsMaxY : Math.Max(scanMaxY.Value, boundsMaxY);
            }
        }
    }

    var segmentsByOsmId = segmentPayloads
        .Where(segment => segment.OsmIdentifier.HasValue)
        .GroupBy(segment => segment.OsmIdentifier!.Value)
        .ToDictionary(group => group.Key, group => group.Select(segment => segment.LocalSegmentId).ToArray());

    var matchedOsmIds = segmentsByOsmId.Keys.ToArray();
    var osmPayloads = new List<TileAlignmentOsmPayload>();
    await using (var osmCommand = connection.CreateCommand())
    {
        if (scanMinX.HasValue && scanMinY.HasValue && scanMaxX.HasValue && scanMaxY.HasValue)
        {
            osmCommand.CommandText = """
                SELECT osm_id,
                       COALESCE(name, '') AS name,
                       COALESCE(NULLIF(building, ''), NULLIF(amenity, ''), NULLIF(shop, ''), 'building') AS feature_type,
                       ST_AsGeoJSON(ST_Transform(way, 4326)) AS geometry_json
                FROM osm.planet_osm_polygon
                WHERE building IS NOT NULL
                  AND way && ST_Transform(ST_Buffer(ST_MakeEnvelope($1, $2, $3, $4, 28355), $5), 3857)
                ORDER BY CASE WHEN osm_id = ANY($6) THEN 0 ELSE 1 END,
                         COALESCE(ST_Area(way), 0) DESC
                LIMIT 500
                """;
            osmCommand.Parameters.Add(new NpgsqlParameter<double> { Value = scanMinX.Value });
            osmCommand.Parameters.Add(new NpgsqlParameter<double> { Value = scanMinY.Value });
            osmCommand.Parameters.Add(new NpgsqlParameter<double> { Value = scanMaxX.Value });
            osmCommand.Parameters.Add(new NpgsqlParameter<double> { Value = scanMaxY.Value });
            osmCommand.Parameters.Add(new NpgsqlParameter<double> { Value = alignmentOsmSearchBufferMeters });
            osmCommand.Parameters.Add(new NpgsqlParameter<long[]> { Value = matchedOsmIds });
        }
        else
        {
            osmCommand.CommandText = """
                SELECT osm_id,
                       COALESCE(name, '') AS name,
                       COALESCE(NULLIF(building, ''), NULLIF(amenity, ''), NULLIF(shop, ''), 'building') AS feature_type,
                       ST_AsGeoJSON(ST_Transform(way, 4326)) AS geometry_json
                FROM osm.planet_osm_polygon
                WHERE building IS NOT NULL
                  AND way && ST_Transform(ST_MakeEnvelope($1, $2, $3, $4, 4326), 3857)
                ORDER BY CASE WHEN osm_id = ANY($5) THEN 0 ELSE 1 END,
                         COALESCE(ST_Area(way), 0) DESC
                LIMIT 500
                """;
            osmCommand.Parameters.Add(new NpgsqlParameter<double> { Value = west });
            osmCommand.Parameters.Add(new NpgsqlParameter<double> { Value = south });
            osmCommand.Parameters.Add(new NpgsqlParameter<double> { Value = east });
            osmCommand.Parameters.Add(new NpgsqlParameter<double> { Value = north });
            osmCommand.Parameters.Add(new NpgsqlParameter<long[]> { Value = matchedOsmIds });
        }

        await using var reader = await osmCommand.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            var osmId = reader.GetInt64(0);
            var segmentIds = segmentsByOsmId.GetValueOrDefault(osmId) ?? [];
            osmPayloads.Add(new TileAlignmentOsmPayload
            {
                OsmId = osmId,
                Name = reader.IsDBNull(1) ? null : reader.GetString(1),
                FeatureType = reader.IsDBNull(2) ? "building" : reader.GetString(2),
                GeometryJson = reader.IsDBNull(3) ? null : reader.GetString(3),
                SegmentIds = segmentIds,
                IsMatched = segmentIds.Length > 0,
            });
        }
    }

    List<TileAlignmentVicmapPropertyPayload> vicmapProperties = [];
    string? vicmapFetchError = null;
    if (scanMinX.HasValue && scanMinY.HasValue && scanMaxX.HasValue && scanMaxY.HasValue)
    {
        (vicmapProperties, vicmapFetchError) = await TryGetVicmapPropertyOverlayAsync(
            tile,
            scanMinX.Value,
            scanMinY.Value,
            scanMaxX.Value,
            scanMaxY.Value);
    }

    return Results.Ok(new TileAlignmentDebugPayload
    {
        TileId = tile.Id,
        TileX = tile.TileX.Value,
        TileY = tile.TileY.Value,
        ZoomLevel = tile.ZoomLevel.Value,
        CenterLat = centerLat,
        CenterLon = centerLon,
        Alignment = alignment,
        TileBoundsGeoJson = tileBoundsGeoJson,
        Segments = segmentPayloads.ToArray(),
        OsmBuildings = osmPayloads.ToArray(),
        VicmapProperties = vicmapProperties.ToArray(),
        VicmapFetchError = vicmapFetchError,
    });
});

app.MapGet("/api/segments", async (MapsDbContext db) =>
{
    var segments = await db.TileSegments
        .OrderByDescending(segment => segment.UpdatedAt)
        .Take(200)
        .Select(segment => new
        {
            segment.Id,
            segment.TileId,
            segment.SourceTileId,
            segment.SourceImportIds,
            segment.LocalSegmentId,
            segment.PredictedLabel,
            segment.ReviewedLabel,
            segment.PointCount,
            segment.Confidence,
            segment.OsmFeatureType,
            segment.OsmName,
            segment.OsmIdentifier,
            segment.OsmMatchStatus,
            segment.OsmMatchScore,
            segment.GeometrySource,
            segment.HasOsmFill,
            segment.CreatedAt,
            segment.UpdatedAt
        })
        .ToListAsync();

    return Results.Ok(segments);
});

app.MapGet("/api/tiles/{id:guid}/segments", async (Guid id, MapsDbContext db) =>
{
    var segments = await db.TileSegments
        .Where(segment => segment.TileId == id)
        .OrderBy(segment => segment.LocalSegmentId)
        .Select(segment => new
        {
            segment.Id,
            segment.TileId,
            segment.SourceTileId,
            segment.SourceImportIds,
            segment.LocalSegmentId,
            segment.PredictedLabel,
            segment.ReviewedLabel,
            segment.PointCount,
            segment.Confidence,
            segment.OsmFeatureType,
            segment.OsmName,
            segment.OsmIdentifier,
            segment.OsmMatchStatus,
            segment.OsmMatchScore,
            segment.GeometrySource,
            segment.HasOsmFill,
            segment.CreatedAt,
            segment.UpdatedAt
        })
        .ToListAsync();

    return Results.Ok(segments);
});

// ── OSM Features in Bounds ──────────────────────────────────────
app.MapGet("/api/osm/features", async (
    double west, double south, double east, double north,
    MapsDbContext db) =>
{
    var features = await db.Database
        .SqlQueryRaw<OsmBuilding>(
            @"SELECT building, COALESCE(name,'') as name, COALESCE(amenity,'') as amenity, COALESCE(shop,'') as shop,
              ST_X(ST_Centroid(ST_Transform(way, 4326))) as lon,
              ST_Y(ST_Centroid(ST_Transform(way, 4326))) as lat
            FROM osm.planet_osm_polygon
            WHERE building IS NOT NULL
              AND way && ST_Transform(ST_MakeEnvelope({0},{1},{2},{3},4326), 3857)
            LIMIT 100", west, south, east, north)
        .ToListAsync();
    return Results.Ok(features);
});

// ── OSM Stats ───────────────────────────────────────────────────
app.MapGet("/api/osm/stats", async (
    double west, double south, double east, double north,
    MapsDbContext db) =>
{
    var conn = db.Database.GetDbConnection();
    await conn.OpenAsync();
    await using var cmd = conn.CreateCommand();
    cmd.CommandText = @"
        SELECT COALESCE(building, 'other') as feature_type, count(*) as count
        FROM osm.planet_osm_polygon
        WHERE building IS NOT NULL
          AND way && ST_Transform(ST_MakeEnvelope($1,$2,$3,$4,4326), 3857)
        GROUP BY building ORDER BY count DESC LIMIT 20";
    cmd.Parameters.Add(new Npgsql.NpgsqlParameter<double> { Value = west });
    cmd.Parameters.Add(new Npgsql.NpgsqlParameter<double> { Value = south });
    cmd.Parameters.Add(new Npgsql.NpgsqlParameter<double> { Value = east });
    cmd.Parameters.Add(new Npgsql.NpgsqlParameter<double> { Value = north });

    var stats = new List<FeatureStat>();
    await using var reader = await cmd.ExecuteReaderAsync();
    while (await reader.ReadAsync())
        stats.Add(new FeatureStat(reader.GetString(0), reader.GetInt64(1)));

    return Results.Ok(stats);
});

// ── Generate 3D Tile ────────────────────────────────────────────
app.MapPost("/api/tiles/generate", async (TileGenerateRequest req, MapsDbContext db) =>
{
    // Validate coordinates
    if (req.Zoom <= 0 || req.MinLon == 0 && req.MinLat == 0)
        return Results.BadRequest(new { error = "Invalid tile coordinates. Ensure LAS data has valid WGS84 bounds." });

    var generator = new TileGenerator();
    var outputDir = RepoPaths.FromRoot("data", "tiles", $"{req.Zoom}.{req.X}.{req.Y}");
            
    var clusters = req.Clusters.Select(c => new ClassifiedCluster
    {
        FeatureType = c.FeatureType,
        AsprsClass = c.AsprsClass,
        Vertices = c.Vertices,
        Faces = c.Faces,
        OsmName = c.OsmName,
    }).ToList();

    try
    {
        var glbPath = await generator.GenerateTileAsync(clusters, req.Zoom, req.X, req.Y, outputDir);
        // ... rest of the endpoint
    }
    catch (ArgumentException ex)
    {
        return Results.BadRequest(new { error = ex.Message });
    }

    // Generate tileset.json
    var tilesetJson = generator.GenerateTilesetJson(
        [new TileInfo
        {
            Zoom = req.Zoom, X = req.X, Y = req.Y,
            MinLon = req.MinLon, MinLat = req.MinLat,
            MaxLon = req.MaxLon, MaxLat = req.MaxLat,
            FeatureCount = clusters.Count,
        }],
        req.MinLon, req.MinLat, req.MaxLon, req.MaxLat);

    var tilesetPath = Path.Combine(outputDir, "tileset.json");
    await File.WriteAllTextAsync(tilesetPath, tilesetJson);

    // Register in catalog with quadtree coordinates
    var entry = new TileCatalogEntry
    {
        Id = Guid.NewGuid(),
        TilesetPath = RepoPaths.RelativeToRoot(tilesetPath).Replace('\\', '/'),
        TileX = req.X,
        TileY = req.Y,
        ZoomLevel = req.Zoom,
        FeatureTypes = clusters.Select(c => c.FeatureType).Distinct().ToArray(),
        MeshCount = clusters.Count,
        CreatedAt = DateTime.UtcNow,
        UpdatedAt = DateTime.UtcNow,
    };
    db.TileCatalog.Add(entry);
    await db.SaveChangesAsync();

    return Results.Ok(new { entry.Id, tilesetPath, clusters = clusters.Count });
});

app.MapPost("/api/tiles/process", async (IServiceScopeFactory scopeFactory) =>
{
    var activeImportIds = await GetActiveImportIdsAsync(scopeFactory);
    if (activeImportIds.Length == 0)
        return Results.BadRequest(new { error = "No active imports are available to process." });

    var job = await StartPipelineJobAsync(
        scopeFactory,
        activeImportIds,
        $"{activeImportIds.Length} active import(s)",
        importJob: null);

    return Results.Ok(new { jobId = job.Id, status = job.Status, fileName = job.FileName, message = job.Message });
});

app.MapPost("/api/tiles/process/imports", async (TileProcessImportsRequest req, IServiceScopeFactory scopeFactory) =>
{
    if (req.ImportIds is null || req.ImportIds.Length == 0)
        return Results.BadRequest(new { error = "At least one import id is required." });

    await using var scope = scopeFactory.CreateAsyncScope();
    var db = scope.ServiceProvider.GetRequiredService<MapsDbContext>();
    var importIds = await db.PointCloudImports
        .Where(importRecord =>
            req.ImportIds.Contains(importRecord.Id) &&
            importRecord.DeletedAt == null &&
            importRecord.Status == "active")
        .OrderBy(importRecord => importRecord.ImportedAt)
        .Select(importRecord => importRecord.Id)
        .ToArrayAsync();

    if (importIds.Length == 0)
        return Results.BadRequest(new { error = "None of the requested imports are active." });

    var job = await StartPipelineJobAsync(
        scopeFactory,
        importIds,
        $"{importIds.Length} selected import(s)",
        importJob: null);

    return Results.Ok(new { jobId = job.Id, status = job.Status, fileName = job.FileName, message = job.Message });
});

app.MapGet("/api/tiles/process/{jobId}", (Guid jobId) =>
{
    if (pipelineJobs.TryGetValue(jobId, out var job))
        return Results.Ok(job);
    return Results.NotFound();
});

app.MapGet("/api/tiles/process", () =>
    Results.Ok(pipelineJobs.Values.OrderByDescending(j => j.StartedAt).Take(20)));

// ── Segment Review Workflow ────────────────────────────────────────

app.MapGet("/api/segments/reviews", async (MapsDbContext db, bool? reviewed) =>
{
    var query = db.SegmentReviews.AsQueryable();
    if (reviewed.HasValue)
        query = query.Where(review => review.Reviewed == reviewed.Value);

    var reviews = await query
        .OrderByDescending(review => review.SubmittedAt)
        .Take(200)
        .Join(
            db.TileSegments,
            review => review.SegmentId,
            segment => segment.Id,
            (review, segment) => new
            {
                review.Id,
                review.SegmentId,
                segment.TileId,
                segment.LocalSegmentId,
                segment.PredictedLabel,
                segment.ReviewedLabel,
                review.CorrectionType,
                review.PreviousLabel,
                review.RequestedLabel,
                review.RelatedSegmentIds,
                review.Notes,
                review.SubmittedBy,
                review.SubmittedAt,
                review.Reviewed,
                review.Approved,
                review.ReviewedAt,
                review.ReviewerNotes,
                review.ExportedAt
            })
        .ToListAsync();

    return Results.Ok(reviews);
});

app.MapPost("/api/segments/reviews", async (SegmentReviewRequest req, MapsDbContext db) =>
{
    var segment = await db.TileSegments.FirstOrDefaultAsync(candidate => candidate.Id == req.SegmentId);
    if (segment is null)
        return Results.NotFound(new { error = "Segment not found." });

    var review = new SegmentReview
    {
        Id = Guid.NewGuid(),
        SegmentId = req.SegmentId,
        CorrectionType = req.CorrectionType,
        PreviousLabel = segment.ReviewedLabel ?? segment.PredictedLabel,
        RequestedLabel = req.RequestedLabel,
        RelatedSegmentIds = req.RelatedSegmentIds,
        Notes = req.Notes,
        SubmittedBy = string.IsNullOrWhiteSpace(req.SubmittedBy) ? "anonymous" : req.SubmittedBy,
        SubmittedAt = DateTime.UtcNow,
        Reviewed = false,
        Approved = false,
    };

    db.SegmentReviews.Add(review);
    await db.SaveChangesAsync();

    return Results.Ok(new { review.Id, status = "pending_review" });
});

app.MapPost("/api/segments/reviews/review", async (SegmentReviewDecision req, MapsDbContext db) =>
{
    var review = await db.SegmentReviews.FirstOrDefaultAsync(candidate => candidate.Id == req.ReviewId);
    if (review is null)
        return Results.NotFound(new { error = "Review not found." });

    review.Reviewed = true;
    review.Approved = req.Approved;
    review.ReviewedAt = DateTime.UtcNow;
    review.ReviewerNotes = req.Notes;

    if (req.Approved)
    {
        var segment = await db.TileSegments.FirstOrDefaultAsync(candidate => candidate.Id == review.SegmentId);
        if (segment is not null && !string.IsNullOrWhiteSpace(review.RequestedLabel) && review.CorrectionType == "reclassify")
            segment.ReviewedLabel = review.RequestedLabel;
    }

    await db.SaveChangesAsync();
    return Results.Ok(new { review.Id, review.Approved });
});

app.MapGet("/api/segments/reviews/training", async (MapsDbContext db) =>
{
    var approved = await (
        from review in db.SegmentReviews
        where review.Reviewed && review.Approved
        join segment in db.TileSegments on review.SegmentId equals segment.Id
        join tile in db.TileCatalog on segment.TileId equals tile.Id
        orderby review.ReviewedAt descending
        select new
        {
            review.Id,
            review.SegmentId,
            segment.TileId,
            SegmentSourceImportIds = segment.SourceImportIds,
            TileSourceImportIds = tile.SourceImportIds,
            segment.LocalSegmentId,
            segment.PredictedLabel,
            segment.ReviewedLabel,
            segment.PointCount,
            segment.ArtifactPath,
            review.CorrectionType,
            review.PreviousLabel,
            review.RequestedLabel,
            review.RelatedSegmentIds,
            review.SubmittedAt,
            review.ReviewedAt,
            review.ExportedAt
        })
        .ToListAsync();

    var pending = approved
        .Where(review => !review.ExportedAt.HasValue)
        .ToList();

    return Results.Ok(new
    {
        count = pending.Count,
        exportedCount = approved.Count(review => review.ExportedAt.HasValue),
        totalCount = approved.Count,
        reviews = pending
    });
});

app.MapPost("/api/segments/reviews/{reviewId:guid}/exported", async (Guid reviewId, MapsDbContext db) =>
{
    var review = await db.SegmentReviews.FirstOrDefaultAsync(candidate => candidate.Id == reviewId);
    if (review is null)
        return Results.NotFound(new { error = "Review not found." });

    if (!review.Reviewed || !review.Approved)
        return Results.BadRequest(new { error = "Only approved reviews can be marked as exported." });

    review.ExportedAt ??= DateTime.UtcNow;
    await db.SaveChangesAsync();
    return Results.Ok(new { review.Id, review.ExportedAt });
});

app.MapPost("/api/segments/reviews/exported", async (SegmentReviewExportRequest req, MapsDbContext db) =>
{
    if (req.ReviewIds is null || req.ReviewIds.Length == 0)
        return Results.BadRequest(new { error = "At least one review id is required." });

    var reviews = await db.SegmentReviews
        .Where(review => req.ReviewIds.Contains(review.Id))
        .ToListAsync();

    if (reviews.Count != req.ReviewIds.Length)
        return Results.BadRequest(new { error = "One or more reviews could not be found." });

    if (reviews.Any(review => !review.Reviewed || !review.Approved))
        return Results.BadRequest(new { error = "Only approved reviews can be marked as exported." });

    var exportedAt = DateTime.UtcNow;
    foreach (var review in reviews)
        review.ExportedAt ??= exportedAt;

    await db.SaveChangesAsync();

    return Results.Ok(new
    {
        count = reviews.Count,
        exportedAt,
        reviewIds = reviews.Select(review => review.Id).ToArray()
    });
});

// ── Corrections / Feedback ───────────────────────────────────────

// Submit a correction
app.MapPost("/api/corrections", async (CorrectionRequest req, MapsDbContext db) =>
{
    var correction = new TileCorrection
    {
        Id = Guid.NewGuid(),
        TileId = req.TileId,
        FeatureIndex = req.FeatureIndex,
        CorrectionType = req.CorrectionType,
        OldLabel = req.OldLabel,
        NewLabel = req.NewLabel,
        SubmittedBy = req.SubmittedBy ?? "anonymous",
        SubmittedAt = DateTime.UtcNow,
        Reviewed = false,
    };
    db.Corrections.Add(correction);
    await db.SaveChangesAsync();
    
    return Results.Ok(new { correction.Id, status = "pending_review" });
});

// List pending corrections
app.MapGet("/api/corrections", async (MapsDbContext db, bool? reviewed) =>
{
    var query = db.Corrections.AsQueryable();
    if (reviewed.HasValue)
        query = query.Where(c => c.Reviewed == reviewed.Value);
    
    var corrections = await query
        .OrderByDescending(c => c.SubmittedAt)
        .Take(100)
        .Select(c => new {
            c.Id, c.TileId, c.FeatureIndex,
            c.CorrectionType, c.OldLabel, c.NewLabel,
            c.SubmittedBy, c.SubmittedAt, c.Reviewed, c.Approved
        })
        .ToListAsync();
    
    return Results.Ok(corrections);
});

// Review a correction (approve/reject)
app.MapPost("/api/corrections/review", async (CorrectionReview review, MapsDbContext db) =>
{
    var correction = await db.Corrections.FindAsync(review.CorrectionId);
    if (correction is null)
        return Results.NotFound();
    
    correction.Reviewed = true;
    correction.Approved = review.Approved;
    correction.ReviewedAt = DateTime.UtcNow;
    correction.ReviewerNotes = review.Notes;
    await db.SaveChangesAsync();
    
    return Results.Ok(new { correction.Id, approved = review.Approved });
});

// Get training-ready corrections (approved, not yet exported)
app.MapGet("/api/corrections/training", async (MapsDbContext db) =>
{
    var approved = await db.Corrections
        .Where(c => c.Reviewed && c.Approved)
        .OrderByDescending(c => c.ReviewedAt)
        .Select(c => new {
            c.Id, c.TileId, c.CorrectionType,
            c.OldLabel, c.NewLabel, c.SubmittedAt
        })
        .ToListAsync();
    
    return Results.Ok(new { count = approved.Count, corrections = approved });
});

// Upload LAS file
app.MapGet("/api/imports", async (MapsDbContext db, bool includeDeleted = false) =>
{
    var query = db.PointCloudImports.AsQueryable();
    if (!includeDeleted)
        query = query.Where(importRecord => importRecord.DeletedAt == null);

    var imports = await query
        .OrderByDescending(importRecord => importRecord.ImportedAt)
        .Select(importRecord => new
        {
            importRecord.Id,
            importRecord.OriginalFileName,
            importRecord.Source,
            importRecord.TotalPointCount,
            importRecord.UniquePointCount,
            importRecord.NewPointCount,
            importRecord.DuplicatePointCount,
            importRecord.DuplicateWithinImportCount,
            importRecord.FileSizeBytes,
            importRecord.Status,
            importRecord.RejectionReason,
            importRecord.AlignmentYawDegrees,
            importRecord.AlignmentOffsetX,
            importRecord.AlignmentOffsetY,
            importRecord.AlignmentSource,
            importRecord.AlignmentUpdatedAt,
            importRecord.ImportedAt,
            importRecord.DeletedAt
        })
        .ToListAsync();

    return Results.Ok(imports);
});

app.MapPost("/api/imports/{id:guid}/alignment", async (Guid id, ImportAlignmentUpdateRequest req, MapsDbContext db) =>
{
    var importRecord = await db.PointCloudImports.FirstOrDefaultAsync(candidate => candidate.Id == id && candidate.DeletedAt == null);
    if (importRecord is null)
        return Results.NotFound(new { error = "Import not found." });

    if (req.Clear)
    {
        importRecord.AlignmentYawDegrees = null;
        importRecord.AlignmentOffsetX = null;
        importRecord.AlignmentOffsetY = null;
        importRecord.AlignmentSource = null;
        importRecord.AlignmentUpdatedAt = null;
    }
    else
    {
        importRecord.AlignmentYawDegrees = req.YawDegrees;
        importRecord.AlignmentOffsetX = req.OffsetX;
        importRecord.AlignmentOffsetY = req.OffsetY;
        importRecord.AlignmentSource = string.IsNullOrWhiteSpace(req.Source) ? "manual" : req.Source.Trim();
        importRecord.AlignmentUpdatedAt = DateTime.UtcNow;
    }

    await db.SaveChangesAsync();

    return Results.Ok(new
    {
        importRecord.Id,
        importRecord.AlignmentYawDegrees,
        importRecord.AlignmentOffsetX,
        importRecord.AlignmentOffsetY,
        importRecord.AlignmentSource,
        importRecord.AlignmentUpdatedAt
    });
});

app.MapPost("/api/imports/{id:guid}/alignment/vicmap", async (Guid id, MapsDbContext db, IServiceScopeFactory scopeFactory, bool? reprocess) =>
{
    var importRecord = await db.PointCloudImports.FirstOrDefaultAsync(candidate => candidate.Id == id && candidate.DeletedAt == null);
    if (importRecord is null)
        return Results.NotFound(new { error = "Import not found." });

    if (!importRecord.Status.Equals("active", StringComparison.OrdinalIgnoreCase))
        return Results.BadRequest(new { error = "Only active imports can be re-aligned." });

    var tile = await db.TileCatalog
        .Where(candidate => candidate.SourceImportIds != null && candidate.SourceImportIds.Contains(id))
        .OrderByDescending(candidate => candidate.UpdatedAt)
        .FirstOrDefaultAsync();

    if (tile is null)
    {
        return Results.BadRequest(new
        {
            error = "No generated tile was found for this import. Process the import once before requesting Vicmap alignment."
        });
    }

    var segmentMetadataPath = ResolveTileSegmentMetadataPath(tile);
    if (string.IsNullOrWhiteSpace(segmentMetadataPath) || !File.Exists(segmentMetadataPath))
    {
        return Results.BadRequest(new
        {
            error = "The current tile is missing segment metadata needed for Vicmap alignment.",
            tileId = tile.Id
        });
    }

    await EnsureDefaultImportAlignmentAsync(db, [id]);
    await db.Entry(importRecord).ReloadAsync();

    VicmapAlignmentResult result;
    try
    {
        result = await RunVicmapAlignmentAsync(segmentMetadataPath);
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new
        {
            error = $"Vicmap alignment failed: {ex.Message}",
            tileId = tile.Id,
            tileX = tile.TileX,
            tileY = tile.TileY
        });
    }

    if (result.Alignment is null || result.MatchedPairs < 1)
    {
        return Results.BadRequest(new
        {
            error = "Vicmap alignment did not return a usable correction.",
            tileId = tile.Id,
            tileX = tile.TileX,
            tileY = tile.TileY,
            matchedPairs = result.MatchedPairs
        });
    }

    var resolvedAlignment = BuildResolvedVicmapAlignment(importRecord, result);
    resolvedAlignment.Source = ShouldUseVicmapYaw(result.Alignment)
        ? $"vicmap_groundtruth_pairs={result.MatchedPairs}"
        : $"vicmap_groundtruth_xy_pairs={result.MatchedPairs}";
    resolvedAlignment.UpdatedAt = DateTime.UtcNow;
    ApplyResolvedAlignment([importRecord], resolvedAlignment);
    await db.SaveChangesAsync();

    var shouldReprocess = reprocess ?? true;
    PipelineJob? job = null;
    if (shouldReprocess)
    {
        job = await StartPipelineJobAsync(
            scopeFactory,
            [importRecord.Id],
            importRecord.OriginalFileName,
            importJob: null);
    }

    return Results.Ok(new
    {
        importId = importRecord.Id,
        importRecord.OriginalFileName,
        tileId = tile.Id,
        tile.TileX,
        tile.TileY,
        segmentMetadataPath = RepoPaths.RelativeToRoot(segmentMetadataPath).Replace('\\', '/'),
        matchedPairs = result.MatchedPairs,
        usedVicmapYaw = ShouldUseVicmapYaw(result.Alignment),
        yawConfidenceThresholdDegrees = VicmapYawConfidenceThresholdDegrees,
        scriptAlignment = result.Alignment,
        appliedAlignment = new
        {
            resolvedAlignment.YawDegrees,
            resolvedAlignment.OffsetX,
            resolvedAlignment.OffsetY,
            resolvedAlignment.Source,
            resolvedAlignment.UpdatedAt
        },
        reprocessStarted = shouldReprocess,
        pipelineJob = job is null
            ? null
            : new
            {
                job.Id,
                job.Status,
                job.Message,
                job.FileName
            }
    });
});

app.MapDelete("/api/imports/{id:guid}", async (Guid id, MapsDbContext db, IServiceScopeFactory scopeFactory) =>
{
    var importRecord = await db.PointCloudImports.FirstOrDefaultAsync(candidate => candidate.Id == id);
    if (importRecord is null || importRecord.DeletedAt.HasValue)
        return Results.NotFound(new { error = "Import not found." });

    Guid? promotedImportId = null;

    var strategy = db.Database.CreateExecutionStrategy();
    await strategy.ExecuteAsync(async () =>
    {
        await using var transaction = await db.Database.BeginTransactionAsync();

        var trackedImport = await db.PointCloudImports.FirstOrDefaultAsync(candidate => candidate.Id == id);
        if (trackedImport is null || trackedImport.DeletedAt.HasValue)
            return;

        var promotionCandidate = trackedImport.Status == "active" && !string.IsNullOrWhiteSpace(trackedImport.PointSignature)
            ? await db.PointCloudImports
                .Where(candidate =>
                    candidate.Id != id &&
                    candidate.DeletedAt == null &&
                    candidate.Status == "rejected" &&
                    candidate.PointSignature == trackedImport.PointSignature)
                .OrderBy(candidate => candidate.ImportedAt)
                .FirstOrDefaultAsync()
            : null;

        var connection = (NpgsqlConnection)db.Database.GetDbConnection();
        if (connection.State != ConnectionState.Open)
            await connection.OpenAsync();

        var dbTransaction = (NpgsqlTransaction)transaction.GetDbTransaction();

        if (promotionCandidate is not null)
        {
            await using var promoteLinks = new NpgsqlCommand("""
                INSERT INTO pointcloud.import_points (import_id, point_id)
                SELECT @promotedImportId, import_point.point_id
                FROM pointcloud.import_points import_point
                WHERE import_point.import_id = @deletedImportId
                ON CONFLICT DO NOTHING;
                """,
                connection,
                dbTransaction);
            promoteLinks.CommandTimeout = 0;
            promoteLinks.Parameters.Add(new NpgsqlParameter("promotedImportId", promotionCandidate.Id));
            promoteLinks.Parameters.Add(new NpgsqlParameter("deletedImportId", id));
            await promoteLinks.ExecuteNonQueryAsync();

            promotionCandidate.Status = "active";
            promotionCandidate.RejectionReason = null;
            promotedImportId = promotionCandidate.Id;
        }

        await using (var deleteLinks = new NpgsqlCommand(
            "DELETE FROM pointcloud.import_points WHERE import_id = @importId",
            connection,
            dbTransaction))
        {
            deleteLinks.CommandTimeout = 0;
            deleteLinks.Parameters.Add(new NpgsqlParameter("importId", id));
            await deleteLinks.ExecuteNonQueryAsync();
        }

        while (true)
        {
            await using var deleteOrphans = new NpgsqlCommand("""
                WITH orphaned AS (
                    SELECT point.id
                    FROM pointcloud.points point
                    LEFT JOIN pointcloud.import_points import_point ON import_point.point_id = point.id
                    WHERE import_point.point_id IS NULL
                    LIMIT 50000
                )
                DELETE FROM pointcloud.points point
                USING orphaned
                WHERE point.id = orphaned.id;
                """,
                connection,
                dbTransaction);
            deleteOrphans.CommandTimeout = 0;
            var deletedRows = await deleteOrphans.ExecuteNonQueryAsync();
            if (deletedRows == 0)
                break;
        }

        trackedImport.Status = "deleted";
        trackedImport.DeletedAt = DateTime.UtcNow;
        await db.SaveChangesAsync();
        await transaction.CommitAsync();
    });

    if (!string.IsNullOrWhiteSpace(importRecord.StoredFilePath))
    {
        var storedPath = Path.IsPathRooted(importRecord.StoredFilePath)
            ? importRecord.StoredFilePath
            : RepoPaths.FromRoot(importRecord.StoredFilePath);
        if (File.Exists(storedPath))
            File.Delete(storedPath);
    }

    var activeImportIds = await GetActiveImportIdsAsync(scopeFactory);
    if (activeImportIds.Length > 0)
    {
        await StartPipelineJobAsync(
            scopeFactory,
            activeImportIds,
            $"{activeImportIds.Length} active import(s)",
            importJob: null);
    }
    else
    {
        await ClearGeneratedOutputsAsync(scopeFactory);
    }

    return Results.Ok(new { importRecord.Id, importRecord.Status, importRecord.DeletedAt, promotedImportId });
});

app.MapPost("/api/import/las", async (HttpRequest request, IServiceScopeFactory scopeFactory) =>
{
    if (!request.HasFormContentType)
        return Results.BadRequest("Expected multipart/form-data");

    var form = await request.ReadFormAsync();
    var file = form.Files.GetFile("file");
    if (file is null || file.Length == 0)
        return Results.BadRequest("No file uploaded");

    var source = form["source"].FirstOrDefault() ?? "user_upload";

    var uploadDir = RepoPaths.RawData;
    Directory.CreateDirectory(uploadDir);
    var savePath = GetAvailableFilePath(uploadDir, file.FileName);

    var job = new ImportJob
    {
        Id = Guid.NewGuid(),
        Type = "las",
        FileName = file.FileName,
        Status = "uploading",
        Message = "Receiving upload...",
        StartedAt = DateTime.UtcNow,
    };
    importJobs[job.Id] = job;
    ApplyImportStage(job);

    UpdateImportState(job, "saving", "Saving LAS file to data\\raw...");
    await using (var stream = new FileStream(savePath, FileMode.Create))
        await file.CopyToAsync(stream);

    job.FileSizeBytes = new FileInfo(savePath).Length;
    UpdateImportState(job, "queued", "Upload received. Import analysis is queued on the server.");

    QueueLasImportProcessing(scopeFactory, savePath, file.FileName, source, job);

    return Results.Ok(new
    {
        job.Id,
        importId = job.ImportId,
        job.Status,
        job.FileName,
        message = job.Message
    });
});

// Get import job status
app.MapGet("/api/import/status", (Guid? id) =>
{
    if (id.HasValue && importJobs.TryGetValue(id.Value, out var job))
        return Results.Ok(job);
    return Results.Ok(importJobs.Values.OrderByDescending(j => j.StartedAt).Take(10));
});

// List recent imports
app.MapGet("/api/import/jobs", () =>
    Results.Ok(importJobs.Values.OrderByDescending(j => j.StartedAt).Take(20)));

// Trigger OSM import
app.MapPost("/api/import/osm", async (HttpRequest request, MapsDbContext db) =>
{
    var form = await request.ReadFormAsync();
    var pbfPath = form["path"].FirstOrDefault() ?? "data/raw/victoria-latest.osm.pbf";

    var job = new ImportJob
    {
        Id = Guid.NewGuid(),
        Type = "osm",
        FileName = Path.GetFileName(pbfPath),
        Status = "starting",
        Message = "Preparing OSM import...",
        StartedAt = DateTime.UtcNow,
    };
    importJobs[job.Id] = job;
    ApplyImportStage(job);

    // Clear any prior osm2pgsql output before re-importing into the same schema.
    UpdateImportState(job, "resetting schema", "Resetting osm schema for a clean import...");
    await db.Database.ExecuteSqlRawAsync("""
        DROP SCHEMA IF EXISTS osm CASCADE;
        CREATE SCHEMA osm;
        """);

    // Run osm2pgsql in Docker (uses host.docker.internal for Windows compatibility)
    UpdateImportState(job, "loading", "Starting osm2pgsql in Docker...");
    
    // Find the Postgres container name so we can attach to its network
    var pgContainerProcess = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
    {
        FileName = "docker",
        Arguments = "ps -q --filter ancestor=postgis/postgis:17-3.5",
        RedirectStandardOutput = true,
        UseShellExecute = false
    });
    var pgContainerId = (pgContainerProcess != null ? await pgContainerProcess.StandardOutput.ReadToEndAsync() : "").Trim().Split('\n').FirstOrDefault()?.Trim();
    
    var pgPassword = Environment.GetEnvironmentVariable("MAPSDB_PASSWORD") ?? "mapspass";
    
    // Fix relative path resolving against TileServer working dir instead of repo root
    var absolutePbfPath = Path.IsPathRooted(pbfPath)
        ? pbfPath
        : RepoPaths.FromRoot(pbfPath);
    
    string networkArg = string.IsNullOrEmpty(pgContainerId) 
        ? $"-H host.docker.internal -P {Environment.GetEnvironmentVariable("MAPSDB_PORT") ?? "5432"}"
        : $"--network container:{pgContainerId} -H localhost -P 5432";

    var args = $"run --rm " +
        $"-e PGPASSWORD=\"{pgPassword}\" " +
        (!string.IsNullOrEmpty(pgContainerId) ? $"--network container:{pgContainerId} " : "") +
        $"-v \"{absolutePbfPath.Replace("\\", "/")}\":/data/osm.pbf:ro " +
        $"iboates/osm2pgsql:latest " +
        $"--create -d mapsdb -U postgres " +
        (string.IsNullOrEmpty(pgContainerId) ? $"-H host.docker.internal -P {Environment.GetEnvironmentVariable("MAPSDB_PORT") ?? "5432"} " : "-H localhost -P 5432 ") +
        $"--schema osm --slim --drop /data/osm.pbf";
    
    job.Message = "Loading OSM data into PostGIS. This can take a while for large PBF files.";
    
    var process = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
    {
        FileName = "docker",
        Arguments = args,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
    });

    if (process is null)
    {
        UpdateImportState(job, "error", "OSM import failed.");
        job.Error = "Failed to start osm2pgsql — is Docker Desktop running?";
        return Results.Ok(new { job.Id, job.Status, job.FileName, job.Error });
    }

    _ = Task.Run(async () =>
    {
        // Read stdout and stderr concurrently to prevent pipe buffer deadlocks
        var stdoutTask = process.StandardOutput.ReadToEndAsync();
        var stderrTask = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();
        
        var stdout = await stdoutTask;
        var stderr = await stderrTask;
        
        if (process.ExitCode == 0)
        {
            UpdateImportState(job, "complete", "OSM import complete.");
            job.CompletedAt = DateTime.UtcNow;
            job.Output = stdout.Length > 500 ? stdout[^500..] : stdout;
        }
        else
        {
            UpdateImportState(job, "error", "OSM import failed.");
            // Show last 300 chars of both streams for diagnosis
            var lastStderr = stderr.Length > 0 ? stderr[^Math.Min(stderr.Length, 300)..] : "";
            var lastStdout = stdout.Length > 0 ? stdout[^Math.Min(stdout.Length, 200)..] : "";
            job.Error = $"[exit {process.ExitCode}] {lastStderr}\n{lastStdout}".Trim();
        }
    });

    return Results.Ok(new { job.Id, job.Status, job.FileName });
});

async Task<PipelineJob> StartPipelineJobAsync(
    IServiceScopeFactory scopeFactory,
    Guid[] sourceImportIds,
    string fileName,
    ImportJob? importJob)
{
    var job = new PipelineJob
    {
        Id = Guid.NewGuid(),
        FileName = fileName,
        SourceImportIds = sourceImportIds,
        Status = "queued",
        Message = "Queued to rebuild the active point bucket.",
        ImportJobId = importJob?.Id,
        StartedAt = DateTime.UtcNow,
    };

    pipelineJobs[job.Id] = job;
    ApplyPipelineStage(job);

    if (importJob is not null)
    {
        importJob.PipelineJobId = job.Id;
        importJob.Status = job.Status;
        importJob.Message = job.Message;
        importJob.TotalBatches = job.TotalBatches;
        importJob.CurrentBatchIndex = job.CurrentBatchIndex;
        ApplyImportStage(importJob);
    }

    _ = Task.Run(async () =>
    {
        try
        {
            var batches = await BuildPipelineInputBatchesAsync(scopeFactory, sourceImportIds);
            if (batches.Length == 0)
                throw new InvalidOperationException("No active point bucket could be built for processing.");

            job.TotalBatches = batches.Length;
            if (importJob is not null)
                importJob.TotalBatches = batches.Length;

            for (var batchIndex = 0; batchIndex < batches.Length; batchIndex++)
            {
                var batch = batches[batchIndex];
                var batchLabel = batches.Length == 1
                    ? batch.FileName
                    : $"import {batchIndex + 1}/{batches.Length} ({batch.FileName})";
                PipelineResultPayload? pipelineResult = null;
                string? inputPath = null;
                job.CurrentBatchIndex = batchIndex + 1;
                if (importJob is not null)
                    importJob.CurrentBatchIndex = job.CurrentBatchIndex;

                try
                {
                    UpdatePipelineState(job, importJob, "assembling_input", $"Building input for {batchLabel}...");
                    inputPath = await BuildPointBucketCsvAsync(batch.SourceImportIds, scopeFactory);
                    if (string.IsNullOrWhiteSpace(inputPath) || !File.Exists(inputPath))
                        throw new InvalidOperationException($"No point bucket could be built for {batch.FileName}.");

                    UpdatePipelineState(job, importJob, "starting", $"Submitting {batchLabel} to mlservice...");
                    pipelineResult = await ProcessPointBucketViaMlServiceAsync(
                        inputPath,
                        batch.Crs,
                        status => UpdatePipelineState(job, importJob, status.Status, status.Message, status.CompletedSteps, status.TotalSteps));
                    job.ZoomLevel = pipelineResult.Zoom;
                    job.TileX = pipelineResult.X;
                    job.TileY = pipelineResult.Y;
                    job.OutputPath = pipelineResult.GlbPath;

                    if (importJob is not null && importJob.ImportId.HasValue && batch.SourceImportIds.Contains(importJob.ImportId.Value))
                    {
                        importJob.ZoomLevel = pipelineResult.Zoom;
                        importJob.TileX = pipelineResult.X;
                        importJob.TileY = pipelineResult.Y;
                        importJob.OutputPath = pipelineResult.GlbPath;
                    }

                    UpdatePipelineState(job, importJob, "registering_tile", $"Registering generated mesh for {batchLabel}...");
                    await RegisterPipelineResultAsync(scopeFactory, pipelineResult, job, batch.SourceImportIds);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Processing '{batch.FileName}' failed. {ex.Message}", ex);
                }
                finally
                {
                    if (!string.IsNullOrWhiteSpace(inputPath) && File.Exists(inputPath))
                        File.Delete(inputPath);
                }
            }

            var completionMessage = batches.Length == 1
                ? "Mesh generation complete."
                : $"Mesh generation complete for {batches.Length} imports.";
            UpdatePipelineState(job, importJob, "complete", completionMessage);
            job.CompletedAt = DateTime.UtcNow;

            if (importJob is not null)
            {
                importJob.Message = importJob.TileX.HasValue && importJob.TileY.HasValue
                    ? $"{job.Message} Tile ({importJob.TileX},{importJob.TileY}) is ready in the catalog."
                    : job.Message;
                importJob.OutputPath = job.OutputPath;
                importJob.CompletedAt = job.CompletedAt;
            }
        }
        catch (Exception ex)
        {
            UpdatePipelineState(job, importJob, "error", "Mesh generation failed.");
            job.Error = AppendTail(job.Error, ex.Message, 8);
            job.CompletedAt = DateTime.UtcNow;

            if (importJob is not null)
            {
                importJob.Error = job.Error;
                importJob.CompletedAt = job.CompletedAt;
            }
        }
    });

    return job;
}

async Task<PipelineInputBatch[]> BuildPipelineInputBatchesAsync(IServiceScopeFactory scopeFactory, Guid[] sourceImportIds)
{
    await using var scope = scopeFactory.CreateAsyncScope();
    var db = scope.ServiceProvider.GetRequiredService<MapsDbContext>();

    var imports = await db.PointCloudImports
        .Where(importRecord =>
            sourceImportIds.Contains(importRecord.Id) &&
            importRecord.DeletedAt == null &&
            importRecord.Status == "active")
        .OrderBy(importRecord => importRecord.ImportedAt)
        .Select(importRecord => new
        {
            importRecord.Id,
            importRecord.OriginalFileName,
            Crs = importRecord.Crs
        })
        .ToListAsync();

    if (imports.Count == 0)
        return [];

    if (imports.Count == 1)
    {
        var importRecord = imports[0];
        return [new PipelineInputBatch([importRecord.Id], importRecord.OriginalFileName, $"EPSG:{importRecord.Crs}")];
    }

    return imports
        .Select(importRecord => new PipelineInputBatch([importRecord.Id], importRecord.OriginalFileName, $"EPSG:{importRecord.Crs}"))
        .ToArray();
}

async Task<ResolvedAlignmentMetadata?> TryApplyVicmapAlignmentAsync(
    MapsDbContext db,
    PipelineResultPayload pipelineResult,
    Guid[] sourceImportIds,
    ResolvedAlignmentMetadata? currentAlignment)
{
    // Don't override manually-set alignments
    if (currentAlignment?.Source is { } source &&
        !source.StartsWith("scaniverse_default", StringComparison.OrdinalIgnoreCase) &&
        !source.StartsWith("vicmap_auto", StringComparison.OrdinalIgnoreCase))
        return null;

    var imports = await db.PointCloudImports
        .Where(importRecord => sourceImportIds.Contains(importRecord.Id) &&
            importRecord.OriginalFileName.ToLower().Contains("scaniverse"))
        .ToListAsync();

    if (imports.Count == 0)
        return null;

    var segmentMetadataPath = pipelineResult.SegmentMetadataPath;
    if (string.IsNullOrWhiteSpace(segmentMetadataPath))
        return null;

    var absoluteMetadataPath = Path.IsPathRooted(segmentMetadataPath)
        ? segmentMetadataPath
        : RepoPaths.FromRoot(segmentMetadataPath);

    if (!File.Exists(absoluteMetadataPath))
        return null;

    try
    {
        var result = await RunVicmapAlignmentAsync(absoluteMetadataPath);
        if (result.Alignment is null || result.MatchedPairs < 1)
            return null;

        var resolvedAlignment = BuildResolvedVicmapAlignment(imports[0], result);
        ApplyResolvedAlignment(imports, resolvedAlignment);
        await db.SaveChangesAsync();
        return resolvedAlignment;
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"Vicmap alignment failed: {ex.Message}");
        return null;
    }
}

async Task<VicmapAlignmentResult> RunVicmapAlignmentAsync(string absoluteMetadataPath)
{
    var alignmentScript = Path.Combine(RepoPaths.MlService, "align_to_vicmap.py");
    if (!File.Exists(alignmentScript))
        throw new FileNotFoundException("Vicmap alignment script is missing.", alignmentScript);

    var psi = new System.Diagnostics.ProcessStartInfo
    {
        FileName = "python",
        Arguments = $"\"{alignmentScript}\" --segments-json \"{absoluteMetadataPath}\"",
        WorkingDirectory = RepoPaths.MlService,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true,
    };

    using var process = System.Diagnostics.Process.Start(psi);
    if (process is null)
        throw new InvalidOperationException("Failed to start the Vicmap alignment script.");

    var stdoutTask = process.StandardOutput.ReadToEndAsync();
    var stderrTask = process.StandardError.ReadToEndAsync();
    await process.WaitForExitAsync();

    var stdout = await stdoutTask;
    var stderr = await stderrTask;
    if (process.ExitCode != 0)
        throw new InvalidOperationException($"Vicmap alignment exited with code {process.ExitCode}: {stderr}".Trim());

    var result = JsonSerializer.Deserialize<VicmapAlignmentResult>(stdout, pipelineJsonOptions);
    if (result is null)
        throw new InvalidOperationException("Vicmap alignment returned no usable JSON result.");

    return result;
}

bool ShouldUseVicmapYaw(VicmapAlignmentPayload alignment)
    => alignment.YawStdev < VicmapYawConfidenceThresholdDegrees;

double ResolveVicmapFallbackYaw(PointCloudImport importRecord)
{
    if (importRecord.AlignmentYawDegrees.HasValue)
        return importRecord.AlignmentYawDegrees.Value;

    return importRecord.OriginalFileName.Contains("scaniverse", StringComparison.OrdinalIgnoreCase)
        ? ScaniverseDefaultYawDegrees
        : 0d;
}

ResolvedAlignmentMetadata BuildResolvedVicmapAlignment(PointCloudImport importRecord, VicmapAlignmentResult result)
{
    if (result.Alignment is null)
        throw new InvalidOperationException("Vicmap alignment payload is missing.");

    var useVicmapYaw = ShouldUseVicmapYaw(result.Alignment);
    var appliedAt = DateTime.UtcNow;

    return new ResolvedAlignmentMetadata
    {
        YawDegrees = useVicmapYaw
            ? result.Alignment.YawDegrees
            : ResolveVicmapFallbackYaw(importRecord),
        OffsetX = result.Alignment.OffsetX,
        OffsetY = result.Alignment.OffsetY,
        Source = useVicmapYaw
            ? $"vicmap_auto_pairs={result.MatchedPairs}"
            : $"vicmap_auto_xy_pairs={result.MatchedPairs}",
        UpdatedAt = appliedAt
    };
}

void ApplyResolvedAlignment(IEnumerable<PointCloudImport> imports, ResolvedAlignmentMetadata alignment)
{
    var appliedAt = alignment.UpdatedAt ?? DateTime.UtcNow;

    foreach (var importRecord in imports)
    {
        importRecord.AlignmentYawDegrees = alignment.YawDegrees;
        importRecord.AlignmentOffsetX = alignment.OffsetX;
        importRecord.AlignmentOffsetY = alignment.OffsetY;
        importRecord.AlignmentSource = alignment.Source;
        importRecord.AlignmentUpdatedAt = appliedAt;
    }
}

async Task RegisterPipelineResultAsync(
    IServiceScopeFactory scopeFactory,
    PipelineResultPayload pipelineResult,
    PipelineJob job,
    Guid[] sourceImportIds)
{
    var absoluteGlbPath = Path.IsPathRooted(pipelineResult.GlbPath)
        ? pipelineResult.GlbPath
        : RepoPaths.FromRoot(pipelineResult.GlbPath);

    if (!File.Exists(absoluteGlbPath))
        throw new FileNotFoundException("Pipeline reported a GLB that does not exist.", absoluteGlbPath);

    await using var scope = scopeFactory.CreateAsyncScope();
    var db = scope.ServiceProvider.GetRequiredService<MapsDbContext>();

    var relativePath = RepoPaths.RelativeToRoot(absoluteGlbPath).Replace('\\', '/');
    var relativeSegmentPath = string.IsNullOrWhiteSpace(pipelineResult.SegmentDataPath)
        ? null
        : RepoPaths.RelativeToRoot(
            Path.IsPathRooted(pipelineResult.SegmentDataPath)
                ? pipelineResult.SegmentDataPath
                : RepoPaths.FromRoot(pipelineResult.SegmentDataPath)).Replace('\\', '/');
    var alignment = await ResolveSourceImportAlignmentAsync(db, sourceImportIds, persistDefaults: true);

    // ── Vicmap auto-alignment (overrides defaults for Scaniverse imports) ──
    alignment = await TryApplyVicmapAlignmentAsync(db, pipelineResult, sourceImportIds, alignment) ?? alignment;

    await TryApplyStoredAlignmentCorrectionAsync(pipelineResult, absoluteGlbPath, alignment);
    var segmentMatches = await MatchSegmentsToOsmAsync(db, pipelineResult.Segments);
    var fillResult = await osmBuildingFillGenerator.ApplyAsync(db, absoluteGlbPath, pipelineResult, segmentMatches);
    pipelineResult.MeshCount += fillResult.AddedMeshCount;
    pipelineResult.VertexCount += fillResult.AddedVertexCount;
    pipelineResult.FaceCount += fillResult.AddedFaceCount;
    var fileInfo = new FileInfo(absoluteGlbPath);
    var featureTypes = (pipelineResult.FeatureTypes?
        .Where(feature => !string.IsNullOrWhiteSpace(feature))
        .Distinct(StringComparer.OrdinalIgnoreCase)
        .ToArray())
        ?? pipelineResult.Segments?
            .Select(segment => segment.FeatureType)
            .Where(feature => !string.IsNullOrWhiteSpace(feature))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    if (fillResult.AddedMeshCount > 0)
    {
        featureTypes = (featureTypes ?? [])
            .Append("building")
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    var existing = await db.TileCatalog
        .FirstOrDefaultAsync(t =>
            t.TileX == pipelineResult.X &&
            t.TileY == pipelineResult.Y &&
            t.ZoomLevel == pipelineResult.Zoom);

    var tileEntry = existing ?? new TileCatalogEntry
    {
        Id = Guid.NewGuid(),
        CreatedAt = DateTime.UtcNow,
    };

    tileEntry.TilesetPath = relativePath;
    tileEntry.TileX = pipelineResult.X;
    tileEntry.TileY = pipelineResult.Y;
    tileEntry.ZoomLevel = pipelineResult.Zoom;
    tileEntry.FeatureTypes = featureTypes;
    tileEntry.MeshCount = pipelineResult.MeshCount;
    tileEntry.SegmentCount = pipelineResult.SegmentCount > 0
        ? pipelineResult.SegmentCount
        : pipelineResult.Segments?.Length;
    tileEntry.VertexCount = pipelineResult.VertexCount;
    tileEntry.FileSizeBytes = fileInfo.Length;
    tileEntry.SegmentsPath = relativeSegmentPath;
    tileEntry.SourceImportIds = sourceImportIds.Length > 0
        ? sourceImportIds
        : tileEntry.SourceImportIds;
    tileEntry.AlignmentYawDegrees = alignment?.YawDegrees;
    tileEntry.AlignmentOffsetX = alignment?.OffsetX;
    tileEntry.AlignmentOffsetY = alignment?.OffsetY;
    tileEntry.AlignmentSource = alignment?.Source;
    tileEntry.AlignmentUpdatedAt = alignment?.UpdatedAt;
    tileEntry.UpdatedAt = DateTime.UtcNow;

    if (existing is null)
    {
        db.TileCatalog.Add(tileEntry);
    }

    await db.SaveChangesAsync();
    await UpsertTileSegmentsAsync(
        db,
        tileEntry,
        pipelineResult,
        tileEntry.SourceImportIds,
        relativeSegmentPath,
        segmentMatches,
        fillResult.FilledSegmentIds);
    await db.SaveChangesAsync();

    job.OutputPath = relativePath;
}

async Task TryApplyStoredAlignmentCorrectionAsync(
    PipelineResultPayload pipelineResult,
    string absoluteGlbPath,
    ResolvedAlignmentMetadata? alignment)
{
    if (alignment is null || !HasAlignmentTransform(alignment.YawDegrees, alignment.OffsetX, alignment.OffsetY))
        return;

    // Skip correction for the default Scaniverse yaw — it's a no-op rotation
    if (Math.Abs(alignment.YawDegrees - ScaniverseDefaultYawDegrees) < 0.01 &&
        Math.Abs(alignment.OffsetX) < 0.01 &&
        Math.Abs(alignment.OffsetY) < 0.01)
        return;

    if (pipelineResult.SceneOrigin.Length < 2 ||
        string.IsNullOrWhiteSpace(pipelineResult.SegmentDataPath) ||
        string.IsNullOrWhiteSpace(pipelineResult.SegmentMetadataPath))
        return;

    var segmentDataPath = Path.IsPathRooted(pipelineResult.SegmentDataPath)
        ? pipelineResult.SegmentDataPath
        : RepoPaths.FromRoot(pipelineResult.SegmentDataPath);
    var segmentMetadataPath = Path.IsPathRooted(pipelineResult.SegmentMetadataPath)
        ? pipelineResult.SegmentMetadataPath
        : RepoPaths.FromRoot(pipelineResult.SegmentMetadataPath);

    if (!File.Exists(absoluteGlbPath) || !File.Exists(segmentDataPath) || !File.Exists(segmentMetadataPath))
        return;

    var correctionScript = Path.Combine(RepoPaths.MlService, "rotate_scan_alignment.py");
    if (!File.Exists(correctionScript))
        return; // Non-fatal — alignment correction script is optional

    try
    {
        var pivotX = pipelineResult.SceneOrigin[0].ToString(CultureInfo.InvariantCulture);
        var pivotY = pipelineResult.SceneOrigin[1].ToString(CultureInfo.InvariantCulture);
        var rotationDegrees = alignment.YawDegrees.ToString(CultureInfo.InvariantCulture);
        var translateX = alignment.OffsetX.ToString(CultureInfo.InvariantCulture);
        var translateY = alignment.OffsetY.ToString(CultureInfo.InvariantCulture);

    // Run inside the mlservice Docker container (has numpy/trimesh)
    var containerName = await GetMlServiceContainerNameAsync();
    if (string.IsNullOrEmpty(containerName))
        return; // No container — skip alignment

    // Convert host paths to container paths (bind mount: data/ → /data/)
    var containerGlbPath = RepoPaths.ToContainerPath(absoluteGlbPath);
    var containerSegmentPath = RepoPaths.ToContainerPath(segmentDataPath);
    var containerMetadataPath = RepoPaths.ToContainerPath(segmentMetadataPath);

    var psi = new System.Diagnostics.ProcessStartInfo
    {
        FileName = "docker",
        Arguments =
            $"exec -i {containerName} python \"{correctionScript}\" " +
            $"--glb \"{containerGlbPath}\" " +
            $"--segments \"{containerSegmentPath}\" " +
            $"--metadata \"{containerMetadataPath}\" " +
            $"--pivot-x \"{pivotX}\" " +
            $"--pivot-y \"{pivotY}\" " +
            $"--rotation-degrees \"{rotationDegrees}\" " +
            $"--translate-x \"{translateX}\" " +
            $"--translate-y \"{translateY}\"",
        WorkingDirectory = RepoPaths.MlService,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true,
    };

        using var process = System.Diagnostics.Process.Start(psi);
        if (process is null)
            return;

        var stdoutTask = process.StandardOutput.ReadToEndAsync();
        var stderrTask = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();

        var stdout = await stdoutTask;
        var stderr = await stderrTask;
        if (process.ExitCode != 0)
        {
            Console.WriteLine($"[TileServer] Alignment correction skipped (non-fatal): {stderr}".Trim());
            return;
        }

        var correction = JsonSerializer.Deserialize<ScanAlignmentCorrectionPayload>(stdout, pipelineJsonOptions);
        if (correction?.Segments is not { Length: > 0 })
            return;

        pipelineResult.Segments = correction.Segments;
        pipelineResult.SegmentCount = correction.Segments.Length;
        pipelineResult.FeatureTypes = correction.Segments
            .Select(segment => segment.FeatureType)
            .Where(featureType => !string.IsNullOrWhiteSpace(featureType))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }
    catch (Exception ex)
    {
        // Non-fatal — alignment correction is best-effort
        Console.WriteLine($"[TileServer] Alignment correction skipped: {ex.Message}");
    }
}

static async Task<string?> GetMlServiceContainerNameAsync()
{
    try
    {
        var psi = new System.Diagnostics.ProcessStartInfo
        {
            FileName = "docker",
            Arguments = "ps --filter ancestor=librerallymaps-mlservice:local --format \"{{.Names}}\" --latest",
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        using var process = System.Diagnostics.Process.Start(psi);
        if (process is null) return null;
        var output = await process.StandardOutput.ReadToEndAsync();
        await process.WaitForExitAsync();
        return output.Trim().Split('\n', StringSplitOptions.RemoveEmptyEntries).FirstOrDefault();
    }
    catch
    {
        return null;
    }
}

async Task EnsureDefaultImportAlignmentAsync(MapsDbContext db, Guid[] sourceImportIds)
{
    if (sourceImportIds.Length == 0)
        return;

    var imports = await db.PointCloudImports
        .Where(importRecord => sourceImportIds.Contains(importRecord.Id))
        .ToListAsync();

    var updated = false;
    foreach (var importRecord in imports)
    {
        if (HasPersistedAlignment(
                importRecord.AlignmentYawDegrees,
                importRecord.AlignmentOffsetX,
                importRecord.AlignmentOffsetY,
                importRecord.AlignmentSource,
                importRecord.AlignmentUpdatedAt) ||
            !importRecord.OriginalFileName.Contains("scaniverse", StringComparison.OrdinalIgnoreCase))
        {
            continue;
        }

        importRecord.AlignmentYawDegrees = ScaniverseDefaultYawDegrees;
        importRecord.AlignmentOffsetX = 0d;
        importRecord.AlignmentOffsetY = 0d;
        importRecord.AlignmentSource = "scaniverse_default";
        importRecord.AlignmentUpdatedAt = DateTime.UtcNow;
        updated = true;
    }

    if (updated)
        await db.SaveChangesAsync();
}

async Task<ResolvedAlignmentMetadata?> ResolveSourceImportAlignmentAsync(
    MapsDbContext db,
    Guid[] sourceImportIds,
    bool persistDefaults)
{
    if (sourceImportIds.Length == 0)
        return null;

    if (persistDefaults)
        await EnsureDefaultImportAlignmentAsync(db, sourceImportIds);

    var imports = await db.PointCloudImports
        .Where(importRecord => sourceImportIds.Contains(importRecord.Id))
        .Select(importRecord => new ResolvedAlignmentMetadata
        {
            YawDegrees = importRecord.AlignmentYawDegrees ?? 0d,
            OffsetX = importRecord.AlignmentOffsetX ?? 0d,
            OffsetY = importRecord.AlignmentOffsetY ?? 0d,
            Source = string.IsNullOrWhiteSpace(importRecord.AlignmentSource) ? null : importRecord.AlignmentSource,
            UpdatedAt = importRecord.AlignmentUpdatedAt
        })
        .ToListAsync();

    if (imports.Count == 0 || imports.Any(importAlignment => !HasPersistedAlignment(
            importAlignment.YawDegrees,
            importAlignment.OffsetX,
            importAlignment.OffsetY,
            importAlignment.Source,
            importAlignment.UpdatedAt)))
    {
        return null;
    }

    var first = imports[0];
    if (imports.Any(importAlignment => !AlignmentValuesMatch(first, importAlignment)))
        return null;

    return new ResolvedAlignmentMetadata
    {
        YawDegrees = first.YawDegrees,
        OffsetX = first.OffsetX,
        OffsetY = first.OffsetY,
        Source = imports.Select(importAlignment => importAlignment.Source)
            .Where(source => !string.IsNullOrWhiteSpace(source))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Count() == 1
            ? first.Source
            : "source_imports",
        UpdatedAt = imports.Max(importAlignment => importAlignment.UpdatedAt)
    };
}

async Task<TileAlignmentMetadataPayload?> ResolveTileAlignmentMetadataAsync(MapsDbContext db, TileCatalogEntry tile)
{
    if (HasPersistedAlignment(
            tile.AlignmentYawDegrees,
            tile.AlignmentOffsetX,
            tile.AlignmentOffsetY,
            tile.AlignmentSource,
            tile.AlignmentUpdatedAt))
    {
        return new TileAlignmentMetadataPayload
        {
            YawDegrees = tile.AlignmentYawDegrees ?? 0d,
            OffsetX = tile.AlignmentOffsetX ?? 0d,
            OffsetY = tile.AlignmentOffsetY ?? 0d,
            Source = tile.AlignmentSource,
            UpdatedAt = tile.AlignmentUpdatedAt
        };
    }

    if (tile.SourceImportIds is not { Length: > 0 })
        return null;

    var resolved = await ResolveSourceImportAlignmentAsync(db, tile.SourceImportIds, persistDefaults: false);
    if (resolved is null)
        return null;

    return new TileAlignmentMetadataPayload
    {
        YawDegrees = resolved.YawDegrees,
        OffsetX = resolved.OffsetX,
        OffsetY = resolved.OffsetY,
        Source = resolved.Source,
        UpdatedAt = resolved.UpdatedAt
    };
}

async Task<(List<TileAlignmentVicmapPropertyPayload> Properties, string? Error)> TryGetVicmapPropertyOverlayAsync(
    TileCatalogEntry tile,
    double minX,
    double minY,
    double maxX,
    double maxY)
{
    var sceneOrigin = TryReadSceneOrigin(tile);
    if (sceneOrigin is null)
        return ([], "Tile scene origin is unavailable, so Vicmaps boundaries cannot be projected into the preview.");

    try
    {
        using var client = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(VicmapOverlayFetchTimeoutSeconds)
        };
        var request = new HttpRequestMessage(HttpMethod.Get, BuildVicmapFeatureServerQueryUri(
            minX - VicmapQueryBufferMeters,
            minY - VicmapQueryBufferMeters,
            maxX + VicmapQueryBufferMeters,
            maxY + VicmapQueryBufferMeters));
        request.Headers.Accept.ParseAdd("application/json");

        using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync();
        using var document = await JsonDocument.ParseAsync(stream);
        if (!document.RootElement.TryGetProperty("features", out var featuresElement) ||
            featuresElement.ValueKind != JsonValueKind.Array)
        {
            return ([], null);
        }

        var properties = new List<TileAlignmentVicmapPropertyPayload>();
        foreach (var featureElement in featuresElement.EnumerateArray())
        {
            if (!featureElement.TryGetProperty("geometry", out var geometryElement) ||
                geometryElement.ValueKind != JsonValueKind.Object)
            {
                continue;
            }

            var localExteriorRings = ExtractLocalExteriorRingsFromArcGis(geometryElement, sceneOrigin.Value);
            if (localExteriorRings.Count == 0)
                continue;

            string? pfi = null;
            if (featureElement.TryGetProperty("attributes", out var propertiesElement) &&
                propertiesElement.ValueKind == JsonValueKind.Object &&
                TryGetArcGisPropertyIdentifier(propertiesElement) is { } pfiValue)
            {
                pfi = pfiValue;
            }

            properties.Add(new TileAlignmentVicmapPropertyPayload
            {
                Pfi = pfi,
                LocalExteriorRings = localExteriorRings
                    .Select(ring => ring.ToArray())
                    .ToArray(),
            });
        }

        return (properties, null);
    }
    catch (OperationCanceledException)
    {
        return ([], "Vicmaps property boundary fetch timed out.");
    }
    catch (HttpRequestException ex)
    {
        return ([], $"Vicmaps property boundary fetch failed: {ex.Message}");
    }
    catch (JsonException ex)
    {
        return ([], $"Vicmaps property boundary response was invalid JSON: {ex.Message}");
    }
}

static Uri BuildVicmapFeatureServerQueryUri(double minX, double minY, double maxX, double maxY)
{
    var query = new Dictionary<string, string>
    {
        ["where"] = "1=1",
        ["geometryType"] = "esriGeometryEnvelope",
        ["spatialRel"] = "esriSpatialRelIntersects",
        ["inSR"] = "28355",
        ["outSR"] = "28355",
        ["f"] = "pjson",
        ["returnGeometry"] = "true",
        ["outFields"] = "prop_pfi,propv_pfi,propv_base_pfi",
        ["resultRecordCount"] = VicmapQueryFeatureLimit.ToString(CultureInfo.InvariantCulture),
        ["geometry"] = string.Create(
            CultureInfo.InvariantCulture,
            $"{{\"xmin\":{minX},\"ymin\":{minY},\"xmax\":{maxX},\"ymax\":{maxY}}}"),
    };

    var queryString = string.Join("&", query.Select(pair =>
        $"{Uri.EscapeDataString(pair.Key)}={Uri.EscapeDataString(pair.Value)}"));

    return new Uri($"{VicmapFeatureServerQueryUrl}?{queryString}");
}

static (double X, double Y, double Z)? TryReadSceneOrigin(TileCatalogEntry tile)
{
    var metadataPath = ResolveTileSegmentMetadataPath(tile);
    if (metadataPath is null || !File.Exists(metadataPath))
        return null;

    try
    {
        using var stream = File.OpenRead(metadataPath);
        using var document = JsonDocument.Parse(stream);
        if (!document.RootElement.TryGetProperty("sceneOrigin", out var sceneOriginElement) ||
            sceneOriginElement.ValueKind != JsonValueKind.Array)
        {
            return null;
        }

        var ordinates = sceneOriginElement.EnumerateArray()
            .Where(value => value.ValueKind == JsonValueKind.Number)
            .Select(value => value.GetDouble())
            .Take(3)
            .ToArray();

        return ordinates.Length < 2
            ? null
            : (ordinates[0], ordinates[1], ordinates.Length > 2 ? ordinates[2] : 0d);
    }
    catch (JsonException)
    {
        return null;
    }
}

static string? ResolveTileSegmentMetadataPath(TileCatalogEntry tile)
{
    if (!string.IsNullOrWhiteSpace(tile.TilesetPath))
    {
        var absoluteTilesetPath = Path.IsPathRooted(tile.TilesetPath)
            ? tile.TilesetPath
            : RepoPaths.FromRoot(tile.TilesetPath);
        var siblingMetadataPath = Path.Combine(Path.GetDirectoryName(absoluteTilesetPath) ?? string.Empty, "segments.json");
        if (File.Exists(siblingMetadataPath))
            return siblingMetadataPath;
    }

    if (string.IsNullOrWhiteSpace(tile.SegmentsPath))
        return null;

    var absoluteSegmentsPath = Path.IsPathRooted(tile.SegmentsPath)
        ? tile.SegmentsPath
        : RepoPaths.FromRoot(tile.SegmentsPath);

    if (File.Exists(absoluteSegmentsPath) &&
        string.Equals(Path.GetExtension(absoluteSegmentsPath), ".json", StringComparison.OrdinalIgnoreCase))
    {
        return absoluteSegmentsPath;
    }

    var sameDirectoryMetadataPath = Path.Combine(Path.GetDirectoryName(absoluteSegmentsPath) ?? string.Empty, "segments.json");
    return File.Exists(sameDirectoryMetadataPath)
        ? sameDirectoryMetadataPath
        : null;
}

static string? TryGetArcGisPropertyIdentifier(JsonElement attributesElement)
{
    foreach (var propertyName in new[] { "prop_pfi", "propv_pfi", "propv_base_pfi" })
    {
        if (attributesElement.TryGetProperty(propertyName, out var valueElement) &&
            valueElement.ValueKind is JsonValueKind.String or JsonValueKind.Number)
        {
            var value = valueElement.ToString();
            if (!string.IsNullOrWhiteSpace(value))
                return value;
        }
    }

    return null;
}

static List<List<TileAlignmentLocalPointPayload>> ExtractLocalExteriorRingsFromArcGis(
    JsonElement geometryElement,
    (double X, double Y, double Z) sceneOrigin)
{
    if (!geometryElement.TryGetProperty("rings", out var ringsElement) ||
        ringsElement.ValueKind != JsonValueKind.Array)
        return [];

    return ringsElement.EnumerateArray()
        .Select(ringElement => TryReadLocalRingFromArcGis(ringElement, sceneOrigin))
        .Where(ring => ring.Count > 1)
        .ToList();
}

static List<TileAlignmentLocalPointPayload> TryReadLocalRingFromArcGis(
    JsonElement ringElement,
    (double X, double Y, double Z) sceneOrigin)
{
    if (ringElement.ValueKind != JsonValueKind.Array)
        return [];

    var ring = new List<TileAlignmentLocalPointPayload>();
    foreach (var pointElement in ringElement.EnumerateArray())
    {
        if (pointElement.ValueKind != JsonValueKind.Array)
            continue;

        var ordinates = pointElement.EnumerateArray()
            .Where(value => value.ValueKind == JsonValueKind.Number)
            .Take(2)
            .Select(value => value.GetDouble())
            .ToArray();

        if (ordinates.Length < 2)
            continue;

        ring.Add(new TileAlignmentLocalPointPayload
        {
            X = ordinates[0] - sceneOrigin.X,
            Y = ordinates[1] - sceneOrigin.Y,
        });
    }

    return ring;
}

static bool HasPersistedAlignment(double? yawDegrees, double? offsetX, double? offsetY, string? source, DateTime? updatedAt)
{
    return yawDegrees.HasValue ||
           offsetX.HasValue ||
           offsetY.HasValue ||
           !string.IsNullOrWhiteSpace(source) ||
           updatedAt.HasValue;
}

static bool HasAlignmentTransform(double yawDegrees, double offsetX, double offsetY)
{
    return Math.Abs(yawDegrees) > 0.0001d ||
           Math.Abs(offsetX) > 0.0001d ||
           Math.Abs(offsetY) > 0.0001d;
}

static bool AlignmentValuesMatch(ResolvedAlignmentMetadata left, ResolvedAlignmentMetadata right)
{
    return Math.Abs(left.YawDegrees - right.YawDegrees) < 0.0001d &&
           Math.Abs(left.OffsetX - right.OffsetX) < 0.0001d &&
           Math.Abs(left.OffsetY - right.OffsetY) < 0.0001d;
}

static (double lat, double lon) GetTileCenter(int zoom, int x, int y)
{
    var n = Math.Pow(2, zoom);
    var lon = ((x + 0.5) / n * 360.0) - 180.0;
    var latRadians = Math.Atan(Math.Sinh(Math.PI * (1 - (2 * (y + 0.5) / n))));
    var lat = latRadians * 180.0 / Math.PI;
    return (lat, lon);
}

static (double west, double south, double east, double north) GetTileBounds(int zoom, int x, int y)
{
    var n = Math.Pow(2, zoom);
    var west = (x / n * 360.0) - 180.0;
    var east = ((x + 1) / n * 360.0) - 180.0;
    var north = Math.Atan(Math.Sinh(Math.PI * (1 - (2d * y / n)))) * 180.0 / Math.PI;
    var south = Math.Atan(Math.Sinh(Math.PI * (1 - (2d * (y + 1) / n)))) * 180.0 / Math.PI;
    return (west, south, east, north);
}

static string CreateTileBoundsGeoJson(int zoom, int x, int y)
{
    var (west, south, east, north) = GetTileBounds(zoom, x, y);

    return JsonSerializer.Serialize(new
    {
        type = "Polygon",
        coordinates = new[]
        {
            new[]
            {
                new[] { west, south },
                new[] { west, north },
                new[] { east, north },
                new[] { east, south },
                new[] { west, south },
            }
        }
    });
}

async Task<Dictionary<int, SegmentOsmMatchPayload>> MatchSegmentsToOsmAsync(
    MapsDbContext db,
    PipelineSegmentPayload[]? segments)
{
    var matches = new Dictionary<int, SegmentOsmMatchPayload>();
    if (segments is null || segments.Length == 0)
        return matches;

    foreach (var segment in segments)
        matches[segment.LocalId] = await MatchSegmentToOsmAsync(db, segment);

    DeduplicateOsmMatches(segments, matches);
    return matches;
}

async Task UpsertTileSegmentsAsync(
    MapsDbContext db,
    TileCatalogEntry tileEntry,
    PipelineResultPayload pipelineResult,
    Guid[]? sourceImportIds,
    string? relativeSegmentPath,
    IReadOnlyDictionary<int, SegmentOsmMatchPayload> segmentMatches,
    IReadOnlySet<int> filledSegmentIds)
{
    var existingSegments = await db.TileSegments
        .Where(segment => segment.TileId == tileEntry.Id)
        .ToDictionaryAsync(segment => segment.LocalSegmentId);

    if (pipelineResult.Segments is null || pipelineResult.Segments.Length == 0)
    {
        if (existingSegments.Count > 0)
            db.TileSegments.RemoveRange(existingSegments.Values);
        return;
    }

    foreach (var segment in pipelineResult.Segments)
    {
        var match = segmentMatches.GetValueOrDefault(segment.LocalId) ?? SegmentOsmMatchPayload.Unmatched;

        if (!existingSegments.TryGetValue(segment.LocalId, out var segmentEntry))
        {
            segmentEntry = new TileSegment
            {
                Id = Guid.NewGuid(),
                TileId = tileEntry.Id,
                LocalSegmentId = segment.LocalId,
                CreatedAt = DateTime.UtcNow,
            };
            db.TileSegments.Add(segmentEntry);
        }
        else
        {
            existingSegments.Remove(segment.LocalId);
        }

        segmentEntry.TileId = tileEntry.Id;
        segmentEntry.SourceImportIds = sourceImportIds?.Length > 0 ? sourceImportIds : segmentEntry.SourceImportIds;
        segmentEntry.PredictedLabel = GetRefinedSegmentLabel(segment, match);
        segmentEntry.PointCount = segment.PointCount;
        segmentEntry.Confidence = segment.Confidence;
        segmentEntry.OsmFeatureType = match.FeatureType;
        segmentEntry.OsmName = match.Name;
        segmentEntry.OsmIdentifier = match.OsmId;
        segmentEntry.OsmMatchStatus = match.MatchStatus;
        segmentEntry.OsmMatchScore = match.MatchScore;
        segmentEntry.PreviewPath = relativeSegmentPath;
        segmentEntry.ArtifactPath = relativeSegmentPath;
        segmentEntry.GeometrySource = filledSegmentIds.Contains(segment.LocalId) ? "scan+osm_fill" : "scan";
        segmentEntry.HasOsmFill = filledSegmentIds.Contains(segment.LocalId);
        segmentEntry.BoundsMinX = segment.BoundsMin.ElementAtOrDefault(0);
        segmentEntry.BoundsMinY = segment.BoundsMin.ElementAtOrDefault(1);
        segmentEntry.BoundsMinZ = segment.BoundsMin.ElementAtOrDefault(2);
        segmentEntry.BoundsMaxX = segment.BoundsMax.ElementAtOrDefault(0);
        segmentEntry.BoundsMaxY = segment.BoundsMax.ElementAtOrDefault(1);
        segmentEntry.BoundsMaxZ = segment.BoundsMax.ElementAtOrDefault(2);
        segmentEntry.CentroidX = segment.Centroid.ElementAtOrDefault(0);
        segmentEntry.CentroidY = segment.Centroid.ElementAtOrDefault(1);
        segmentEntry.CentroidZ = segment.Centroid.ElementAtOrDefault(2);
        segmentEntry.UpdatedAt = DateTime.UtcNow;
    }

    if (existingSegments.Count > 0)
        db.TileSegments.RemoveRange(existingSegments.Values);
}

void DeduplicateOsmMatches(IReadOnlyCollection<PipelineSegmentPayload> segments, IDictionary<int, SegmentOsmMatchPayload> matches)
{
    var groups = segments
        .Select(segment =>
        {
            matches.TryGetValue(segment.LocalId, out var match);
            return new
            {
                Segment = segment,
                Match = match ?? SegmentOsmMatchPayload.Unmatched
            };
        })
        .Where(entry =>
            entry.Match.OsmId.HasValue &&
            !string.Equals(entry.Match.MatchStatus, "unmatched", StringComparison.OrdinalIgnoreCase))
        .GroupBy(entry => entry.Match.OsmId!.Value);

    foreach (var group in groups)
    {
        var keep = group
            .OrderBy(entry => GetMatchPriority(entry.Match.MatchStatus))
            .ThenByDescending(entry => entry.Match.MatchScore ?? 0f)
            .ThenByDescending(entry => entry.Segment.PointCount)
            .First();

        foreach (var duplicate in group)
        {
            if (duplicate.Segment.LocalId == keep.Segment.LocalId)
                continue;

            matches[duplicate.Segment.LocalId] = SegmentOsmMatchPayload.Unmatched;
        }
    }
}

int GetMatchPriority(string? matchStatus)
{
    if (string.Equals(matchStatus, "matched", StringComparison.OrdinalIgnoreCase))
        return 0;

    if (string.Equals(matchStatus, "nearby", StringComparison.OrdinalIgnoreCase))
        return 1;

    return 2;
}

async Task<SegmentOsmMatchPayload> MatchSegmentToOsmAsync(MapsDbContext db, PipelineSegmentPayload segment)
{
    if (!ShouldEvaluateBuildingOsmMatch(segment))
        return SegmentOsmMatchPayload.Unmatched;

    var hasCentroid = segment.Centroid.Length >= 2;
    var hasBounds =
        segment.BoundsMin.Length >= 2 &&
        segment.BoundsMax.Length >= 2 &&
        segment.BoundsMax[0] > segment.BoundsMin[0] &&
        segment.BoundsMax[1] > segment.BoundsMin[1];

    if (!hasCentroid)
        return SegmentOsmMatchPayload.Unmatched;

    try
    {
        var conn = db.Database.GetDbConnection();
        if (conn.State != System.Data.ConnectionState.Open)
            await conn.OpenAsync();

        await using var cmd = conn.CreateCommand();
        cmd.CommandText = """
            WITH segment_input AS (
                SELECT
                    ST_Transform(ST_SetSRID(ST_MakePoint($1, $2), 28355), 3857) AS centroid_geom,
                    CASE
                        WHEN $3 IS NOT NULL AND $4 IS NOT NULL AND $5 IS NOT NULL AND $6 IS NOT NULL
                        THEN ST_Transform(ST_SetSRID(ST_MakeEnvelope($3, $4, $5, $6), 28355), 3857)
                        ELSE NULL
                    END AS bounds_geom
            ),
            candidates AS (
                SELECT
                    osm_id,
                    COALESCE(name, '') AS name,
                    COALESCE(NULLIF(building, ''), NULLIF(amenity, ''), NULLIF(shop, ''), 'feature') AS feature_type,
                    CASE
                        WHEN segment_input.bounds_geom IS NOT NULL AND ST_Intersects(way, segment_input.bounds_geom) THEN 'matched'
                        WHEN ST_Contains(way, segment_input.centroid_geom) THEN 'matched'
                        ELSE 'nearby'
                    END AS match_status,
                    ST_Distance(way, segment_input.centroid_geom) AS centroid_distance,
                    CASE
                        WHEN segment_input.bounds_geom IS NOT NULL AND ST_Intersects(way, segment_input.bounds_geom)
                        THEN ST_Area(ST_Intersection(way, segment_input.bounds_geom))
                        ELSE 0
                    END AS overlap_area,
                    CASE
                        WHEN segment_input.bounds_geom IS NOT NULL
                        THEN NULLIF(ST_Area(segment_input.bounds_geom), 0)
                        ELSE NULL
                    END AS segment_area
                FROM osm.planet_osm_polygon, segment_input
                WHERE building IS NOT NULL
                  AND (
                      (segment_input.bounds_geom IS NOT NULL AND ST_DWithin(way, segment_input.bounds_geom, 20))
                      OR ST_DWithin(way, segment_input.centroid_geom, 35)
                  )
            )
            SELECT osm_id,
                   name,
                   feature_type,
                   match_status,
                   centroid_distance,
                   overlap_area,
                   segment_area
            FROM candidates
            ORDER BY CASE match_status WHEN 'matched' THEN 0 ELSE 1 END,
                     CASE
                         WHEN segment_area IS NOT NULL AND segment_area > 0 THEN overlap_area / segment_area
                         ELSE 0
                     END DESC,
                     centroid_distance
            LIMIT 1
            """;
        cmd.Parameters.Add(new NpgsqlParameter<double> { Value = segment.Centroid[0] });
        cmd.Parameters.Add(new NpgsqlParameter<double> { Value = segment.Centroid[1] });
        cmd.Parameters.Add(new NpgsqlParameter<double?> { Value = hasBounds ? segment.BoundsMin[0] : null });
        cmd.Parameters.Add(new NpgsqlParameter<double?> { Value = hasBounds ? segment.BoundsMin[1] : null });
        cmd.Parameters.Add(new NpgsqlParameter<double?> { Value = hasBounds ? segment.BoundsMax[0] : null });
        cmd.Parameters.Add(new NpgsqlParameter<double?> { Value = hasBounds ? segment.BoundsMax[1] : null });

        await using var reader = await cmd.ExecuteReaderAsync();
        if (!await reader.ReadAsync())
            return SegmentOsmMatchPayload.Unmatched;

        var distance = reader.IsDBNull(4) ? 35d : reader.GetDouble(4);
        var overlapArea = reader.IsDBNull(5) ? 0d : reader.GetDouble(5);
        var segmentArea = reader.IsDBNull(6) ? 0d : reader.GetDouble(6);
        var overlapRatio = segmentArea > 0d ? Math.Clamp(overlapArea / segmentArea, 0d, 1d) : 0d;
        var proximityScore = Math.Max(0d, 1d - Math.Min(distance, 35d) / 35d);
        var score = Math.Max(proximityScore, overlapRatio);
        return new SegmentOsmMatchPayload
        {
            OsmId = reader.IsDBNull(0) ? null : reader.GetInt64(0),
            Name = reader.IsDBNull(1) ? null : reader.GetString(1),
            FeatureType = reader.IsDBNull(2) ? null : reader.GetString(2),
            MatchStatus = reader.IsDBNull(3) ? "unmatched" : reader.GetString(3),
            MatchScore = (float)score,
        };
    }
    catch
    {
        return SegmentOsmMatchPayload.Unmatched;
    }
}

bool ShouldEvaluateBuildingOsmMatch(PipelineSegmentPayload segment)
{
    return BuildingSegmentHeuristics.ShouldEvaluateOsmMatch(segment);
}

string GetRefinedSegmentLabel(PipelineSegmentPayload segment, SegmentOsmMatchPayload match)
{
    if (BuildingSegmentHeuristics.CanPromoteToBuilding(segment, match))
        return "building";

    return segment.FeatureType;
}

async Task<PreparedImportData> AnalyzeImportAsync(string inputPath)
{
    var scriptPath = Path.Combine(RepoPaths.MlService, "extract_unique_points.py");
    if (!File.Exists(scriptPath))
        throw new FileNotFoundException("Import analysis script is missing.", scriptPath);

    var csvPath = Path.Combine(Path.GetTempPath(), $"maps-import-{Guid.NewGuid():N}.csv");
    var jsonPath = Path.Combine(Path.GetTempPath(), $"maps-import-{Guid.NewGuid():N}.json");
    var psi = new System.Diagnostics.ProcessStartInfo
    {
        FileName = "python",
        Arguments = $"\"{scriptPath}\" --input \"{inputPath}\" --output-csv \"{csvPath}\" --output-json \"{jsonPath}\"",
        WorkingDirectory = RepoPaths.MlService,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true,
    };

    using var process = System.Diagnostics.Process.Start(psi);
    if (process is null)
        throw new InvalidOperationException("Failed to start LAS import analysis.");

    var stdoutTask = process.StandardOutput.ReadToEndAsync();
    var stderrTask = process.StandardError.ReadToEndAsync();
    await process.WaitForExitAsync();

    if (process.ExitCode != 0)
        throw new InvalidOperationException($"Import analysis failed: {(await stderrTask).Trim()}".Trim());

    if (!File.Exists(csvPath) || !File.Exists(jsonPath))
        throw new InvalidOperationException("Import analysis did not produce the expected output files.");

    var metadata = JsonSerializer.Deserialize<PreparedImportMetadata>(
        await File.ReadAllTextAsync(jsonPath),
        pipelineJsonOptions);

    if (metadata is null)
        throw new InvalidOperationException("Import analysis metadata was empty.");

    return new PreparedImportData(
        csvPath,
        jsonPath,
        metadata.TotalPoints,
        metadata.UniquePoints,
        metadata.DuplicateWithinImportPoints,
        metadata.PointSignature);
}

async Task<ImportMergeResult> RegisterImportAsync(MapsDbContext db, PointCloudImport importRecord, PreparedImportData preparedImport)
{
    var strategy = db.Database.CreateExecutionStrategy();
    return await strategy.ExecuteAsync(async () =>
    {
        await using var transaction = await db.Database.BeginTransactionAsync();
        if (db.Entry(importRecord).State == EntityState.Detached)
            db.PointCloudImports.Add(importRecord);

        await db.SaveChangesAsync();

        var connection = (NpgsqlConnection)db.Database.GetDbConnection();
        if (connection.State != ConnectionState.Open)
            await connection.OpenAsync();

        var dbTransaction = (NpgsqlTransaction)transaction.GetDbTransaction();

        await using (var createStage = new NpgsqlCommand("""
            CREATE TEMP TABLE import_stage_points (
                x DOUBLE PRECISION NOT NULL,
                y DOUBLE PRECISION NOT NULL,
                z DOUBLE PRECISION NOT NULL,
                red INTEGER,
                green INTEGER,
                blue INTEGER
            ) ON COMMIT DROP;
            """, connection, dbTransaction))
        {
            createStage.CommandTimeout = 0;
            await createStage.ExecuteNonQueryAsync();
        }

        await using (var importer = connection.BeginTextImport("COPY import_stage_points (x, y, z, red, green, blue) FROM STDIN WITH (FORMAT csv)"))
        using (var reader = new StreamReader(preparedImport.CsvPath))
        {
            char[] buffer = new char[8192];
            int read;
            while ((read = await reader.ReadAsync(buffer, 0, buffer.Length)) > 0)
                await importer.WriteAsync(buffer.AsMemory(0, read));
        }

        long insertedPointCount;
        await using (var insertPoints = new NpgsqlCommand("""
            WITH inserted AS (
                INSERT INTO pointcloud.points (x, y, z, red, green, blue, created_at, updated_at)
                SELECT stage.x,
                       stage.y,
                       stage.z,
                       NULLIF(stage.red, -1),
                       NULLIF(stage.green, -1),
                       NULLIF(stage.blue, -1),
                       NOW(),
                       NOW()
                FROM import_stage_points stage
                ON CONFLICT (x, y, z) DO NOTHING
                RETURNING 1
            )
            SELECT COUNT(*) FROM inserted;
            """, connection, dbTransaction))
        {
            insertPoints.CommandTimeout = 0;
            insertedPointCount = Convert.ToInt64(await insertPoints.ExecuteScalarAsync());
        }

        importRecord.UniquePointCount = preparedImport.UniquePoints;
        importRecord.NewPointCount = insertedPointCount;
        importRecord.DuplicatePointCount = preparedImport.UniquePoints - insertedPointCount;
        importRecord.DuplicateWithinImportCount = preparedImport.DuplicateWithinImportPoints;
        importRecord.PointSignature = preparedImport.PointSignature;

        if (preparedImport.UniquePoints == 0 || insertedPointCount == 0)
        {
            importRecord.Status = "rejected";
            importRecord.RejectionReason = preparedImport.UniquePoints == 0
                ? "Import did not contain any unique XYZ points."
                : "Import contains only points that already exist in the global point bucket.";
            await db.SaveChangesAsync();
            await transaction.CommitAsync();
            return new ImportMergeResult(true, insertedPointCount, importRecord.DuplicatePointCount ?? 0);
        }

        await using (var linkPoints = new NpgsqlCommand("""
            INSERT INTO pointcloud.import_points (import_id, point_id)
            SELECT @importId,
                   point.id
            FROM import_stage_points stage
            JOIN pointcloud.points point
              ON point.x = stage.x
             AND point.y = stage.y
             AND point.z = stage.z
            ON CONFLICT DO NOTHING;
            """, connection, dbTransaction))
        {
            linkPoints.CommandTimeout = 0;
            linkPoints.Parameters.Add(new NpgsqlParameter("importId", importRecord.Id));
            await linkPoints.ExecuteNonQueryAsync();
        }

        importRecord.Status = "active";
        await db.SaveChangesAsync();
        await transaction.CommitAsync();
        return new ImportMergeResult(false, insertedPointCount, importRecord.DuplicatePointCount ?? 0);
    });
}

void QueueLasImportProcessing(
    IServiceScopeFactory scopeFactory,
    string savePath,
    string fileName,
    string source,
    ImportJob job)
{
    _ = Task.Run(async () =>
    {
        PreparedImportData? preparedImport = null;

        try
        {
            await using var scope = scopeFactory.CreateAsyncScope();
            var db = scope.ServiceProvider.GetRequiredService<MapsDbContext>();

            var storedRelativePath = RepoPaths.RelativeToRoot(savePath).Replace('\\', '/');

            UpdateImportState(job, "analyzing", "Extracting unique XYZ points for import dedupe...");
            preparedImport = await AnalyzeImportAsync(savePath);
            job.TotalSteps = preparedImport.TotalPoints;
            job.CompletedSteps = preparedImport.TotalPoints;

            var importRecord = new PointCloudImport
            {
                Id = Guid.NewGuid(),
                OriginalFileName = fileName,
                StoredFilePath = storedRelativePath,
                Source = source,
                  Crs = 4326,
                PointSignature = preparedImport.PointSignature,
                TotalPointCount = preparedImport.TotalPoints,
                UniquePointCount = preparedImport.UniquePoints,
                DuplicateWithinImportCount = preparedImport.DuplicateWithinImportPoints,
                FileSizeBytes = job.FileSizeBytes,
                Status = "pending",
                ImportedAt = DateTime.UtcNow,
            };

            job.ImportId = importRecord.Id;
            UpdateImportState(job, "registering", "Registering deduplicated points in the global point bucket...");

            var mergeResult = await RegisterImportAsync(db, importRecord, preparedImport);
            job.NewPointCount = mergeResult.NewPointCount;
            job.UniquePointCount = preparedImport.UniquePoints;
            job.DuplicatePointCount = mergeResult.DuplicatePointCount;
            job.DuplicateWithinImportCount = preparedImport.DuplicateWithinImportPoints;

            if (mergeResult.Rejected)
            {
                if (File.Exists(savePath))
                    File.Delete(savePath);

                importRecord.StoredFilePath = null;
                await db.SaveChangesAsync();

                UpdateImportState(job, "rejected", importRecord.RejectionReason ?? "Import rejected.");
                job.CompletedAt = DateTime.UtcNow;
                return;
            }

            UpdateImportState(job, "queued", $"Import registered with {mergeResult.NewPointCount:N0} new unique points. Rebuilding the active point bucket...");

            var activeImportIds = await GetActiveImportIdsAsync(scopeFactory);
            await StartPipelineJobAsync(
                scopeFactory,
                activeImportIds,
                $"{activeImportIds.Length} active import(s)",
                job);
        }
        catch (Exception ex)
        {
            UpdateImportState(job, "error", "Import processing failed.");
            job.Error = AppendTail(job.Error, ex.Message, 8);
            job.CompletedAt = DateTime.UtcNow;
        }
        finally
        {
            preparedImport?.Dispose();
        }
    });
}

async Task<Guid[]> GetActiveImportIdsAsync(IServiceScopeFactory scopeFactory)
{
    await using var scope = scopeFactory.CreateAsyncScope();
    var db = scope.ServiceProvider.GetRequiredService<MapsDbContext>();
    return await db.PointCloudImports
        .Where(importRecord => importRecord.DeletedAt == null && importRecord.Status == "active")
        .OrderBy(importRecord => importRecord.ImportedAt)
        .Select(importRecord => importRecord.Id)
        .ToArrayAsync();
}

async Task<string?> BuildPointBucketCsvAsync(Guid[] importIds, IServiceScopeFactory scopeFactory)
{
    if (importIds.Length == 0)
        return null;

    await using var scope = scopeFactory.CreateAsyncScope();
    var db = scope.ServiceProvider.GetRequiredService<MapsDbContext>();
    var connection = (NpgsqlConnection)db.Database.GetDbConnection();
    if (connection.State != ConnectionState.Open)
        await connection.OpenAsync();

    await using (var countCommand = new NpgsqlCommand("""
        SELECT COUNT(DISTINCT import_point.point_id)
        FROM pointcloud.import_points import_point
        JOIN pointcloud.imports import_record ON import_record.id = import_point.import_id
        WHERE import_record.deleted_at IS NULL
          AND import_record.status = 'active'
          AND import_point.import_id = ANY (@importIds);
        """, connection))
    {
        countCommand.CommandTimeout = 0;
        countCommand.Parameters.Add(new NpgsqlParameter("importIds", importIds)
        {
            NpgsqlDbType = NpgsqlDbType.Array | NpgsqlDbType.Uuid
        });

        var pointCount = Convert.ToInt64(await countCommand.ExecuteScalarAsync());
        if (pointCount == 0)
            return null;
    }

    Directory.CreateDirectory(RepoPaths.PipelineInputData);
    var csvPath = Path.Combine(RepoPaths.PipelineInputData, $"maps-point-bucket-{Guid.NewGuid():N}.csv");
    await using var exportCommand = new NpgsqlCommand("""
        SELECT DISTINCT ON (point.id)
               point.id,
               point.x,
               point.y,
               point.z,
               COALESCE(point.red, -1),
               COALESCE(point.green, -1),
               COALESCE(point.blue, -1)
        FROM pointcloud.import_points import_point
        JOIN pointcloud.imports import_record ON import_record.id = import_point.import_id
        JOIN pointcloud.points point ON point.id = import_point.point_id
        WHERE import_record.deleted_at IS NULL
          AND import_record.status = 'active'
          AND import_point.import_id = ANY (@importIds)
        ORDER BY point.id;
        """, connection);
    exportCommand.CommandTimeout = 0;
    exportCommand.Parameters.Add(new NpgsqlParameter("importIds", importIds)
    {
        NpgsqlDbType = NpgsqlDbType.Array | NpgsqlDbType.Uuid
    });

    await using var reader = await exportCommand.ExecuteReaderAsync(CommandBehavior.SequentialAccess);
    await using var writer = new StreamWriter(csvPath);

    while (await reader.ReadAsync())
    {
        var line = string.Join(',',
            reader.GetInt64(0).ToString(CultureInfo.InvariantCulture),
            reader.GetDouble(1).ToString("R", CultureInfo.InvariantCulture),
            reader.GetDouble(2).ToString("R", CultureInfo.InvariantCulture),
            reader.GetDouble(3).ToString("R", CultureInfo.InvariantCulture),
            (reader.IsDBNull(4) ? -1 : reader.GetInt32(4)).ToString(CultureInfo.InvariantCulture),
            (reader.IsDBNull(5) ? -1 : reader.GetInt32(5)).ToString(CultureInfo.InvariantCulture),
            (reader.IsDBNull(6) ? -1 : reader.GetInt32(6)).ToString(CultureInfo.InvariantCulture));

        await writer.WriteLineAsync(line);
    }

    return csvPath;
}

async Task<PipelineResultPayload> ProcessPointBucketViaMlServiceAsync(string inputPath, string inputCrs = "EPSG:28355", Action<PipelineStatusPayload>? onStatus = null)
{
    var client = httpClientFactory.CreateClient("mlservice");
    var relativeInputPath = RepoPaths.RelativeToRoot(inputPath).Replace('\\', '/');
    using var request = new HttpRequestMessage(HttpMethod.Post, "/process/stream")
    {
        Content = JsonContent.Create(new MlServiceProcessRequest
        {
            InputPath = relativeInputPath,
            Crs = inputCrs
        })
    };
    using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);

    if (!response.IsSuccessStatusCode)
    {
        var responseContent = await response.Content.ReadAsStringAsync();
        throw new InvalidOperationException($"ML service processing failed ({(int)response.StatusCode}): {responseContent}".Trim());
    }

    await using var responseStream = await response.Content.ReadAsStreamAsync();
    using var reader = new StreamReader(responseStream);

    PipelineResultPayload? pipelineResult = null;

    while (!reader.EndOfStream)
    {
        var line = await reader.ReadLineAsync();
        if (string.IsNullOrWhiteSpace(line))
            continue;

        var processEvent = JsonSerializer.Deserialize<MlServiceProcessEventPayload>(line, pipelineJsonOptions);
        if (processEvent is null)
            continue;

        switch (processEvent.Type)
        {
            case "status" when processEvent.Status is not null && processEvent.Message is not null:
                onStatus?.Invoke(new PipelineStatusPayload
                {
                    Status = processEvent.Status,
                    Message = processEvent.Message,
                    CompletedSteps = processEvent.CompletedSteps,
                    TotalSteps = processEvent.TotalSteps
                });
                break;
            case "result" when processEvent.Result is not null:
                pipelineResult = processEvent.Result;
                break;
            case "error":
                throw new InvalidOperationException($"ML service processing failed: {processEvent.Detail}".Trim());
        }
    }

    if (pipelineResult is null)
        throw new InvalidOperationException("ML service completed without returning pipeline output.");

    return pipelineResult;
}

async Task ClearGeneratedOutputsAsync(IServiceScopeFactory scopeFactory)
{
    await using var scope = scopeFactory.CreateAsyncScope();
    var db = scope.ServiceProvider.GetRequiredService<MapsDbContext>();

    var segments = await db.TileSegments.ToListAsync();
    if (segments.Count > 0)
        db.TileSegments.RemoveRange(segments);

    var tiles = await db.TileCatalog.ToListAsync();
    if (tiles.Count > 0)
        db.TileCatalog.RemoveRange(tiles);

    await db.SaveChangesAsync();

    if (Directory.Exists(RepoPaths.TilesData))
        Directory.Delete(RepoPaths.TilesData, recursive: true);

    Directory.CreateDirectory(RepoPaths.TilesData);
}

void UpdatePipelineState(PipelineJob job, ImportJob? importJob, string status, string message, long? completedSteps = null, long? totalSteps = null)
{
    job.Status = status;
    job.Message = message;
    if (completedSteps.HasValue)
        job.CompletedSteps = completedSteps.Value;
    if (totalSteps.HasValue)
        job.TotalSteps = totalSteps.Value;
    ApplyPipelineStage(job);

    if (importJob is null)
        return;

    importJob.Status = status;
    importJob.Message = message;
    if (completedSteps.HasValue)
        importJob.CompletedSteps = completedSteps.Value;
    if (totalSteps.HasValue)
        importJob.TotalSteps = totalSteps.Value;
    importJob.CurrentBatchIndex = job.CurrentBatchIndex;
    importJob.TotalBatches = job.TotalBatches;
    ApplyImportStage(importJob);
}

void UpdateImportState(ImportJob job, string status, string message)
{
    job.Status = status;
    job.Message = message;
    ApplyImportStage(job);
}

void ApplyPipelineStage(PipelineJob job)
{
    job.TotalStages = 5;
    job.CurrentStage = job.Status switch
    {
        "queued" => 1,
        "assembling_input" => 2,
        "starting" or "loading_las" or "classifying" or "segmenting" or "reconstructing" or "generating_glb" or "generated_glb" => 3,
        "registering_tile" => 4,
        "complete" => 5,
        "error" => Math.Clamp(job.CurrentStage == 0 ? 3 : job.CurrentStage, 1, job.TotalStages),
        _ => 1
    };
}

void ApplyImportStage(ImportJob job)
{
    if (string.Equals(job.Type, "osm", StringComparison.OrdinalIgnoreCase))
    {
        job.TotalStages = 4;
        job.CurrentStage = job.Status switch
        {
            "starting" => 1,
            "resetting schema" => 2,
            "loading" => 3,
            "complete" => 4,
            "error" => Math.Clamp(job.CurrentStage == 0 ? 3 : job.CurrentStage, 1, job.TotalStages),
            _ => 1
        };
        return;
    }

    job.TotalStages = 7;
    job.CurrentStage = job.Status switch
    {
        "uploading" or "saving" => 1,
        "analyzing" => 2,
        "registering" => 3,
        "queued" => 4,
        "assembling_input" => 5,
        "starting" or "loading_las" or "classifying" or "segmenting" or "reconstructing" or "generating_glb" or "generated_glb" => 6,
        "registering_tile" or "complete" => 7,
        "rejected" => 3,
        "error" => Math.Clamp(job.CurrentStage == 0 ? (job.PipelineJobId.HasValue ? 6 : 3) : job.CurrentStage, 1, job.TotalStages),
        _ => 1
    };
}

static string AppendTail(string? existing, string line, int maxLines)
{
    var lines = (existing ?? string.Empty)
        .Split('\n', StringSplitOptions.RemoveEmptyEntries)
        .ToList();
    lines.Add(line.TrimEnd());

    if (lines.Count > maxLines)
        lines = lines[^maxLines..];

    return string.Join(Environment.NewLine, lines);
}

static string GetAvailableFilePath(string directory, string fileName)
{
    var baseName = Path.GetFileNameWithoutExtension(fileName);
    var extension = Path.GetExtension(fileName);
    var candidate = Path.Combine(directory, fileName);
    var suffix = 1;

    while (File.Exists(candidate))
    {
        candidate = Path.Combine(directory, $"{baseName}-{suffix}{extension}");
        suffix++;
    }

    return candidate;
}

app.Run();

// ── Request Models ──────────────────────────────────────────────
record TileGenerateRequest(
    int Zoom, int X, int Y,
    double MinLon, double MinLat, double MaxLon, double MaxLat,
    List<ClusterData> Clusters
);

record ClusterData(
    string FeatureType, int AsprsClass,
    List<double[]> Vertices, List<int[]> Faces,
    string? OsmName
);

// ── Pipeline Job ────────────────────────────────────────────────
class PipelineJob
{
    public Guid Id { get; set; }
    public Guid? ImportJobId { get; set; }
    public Guid[]? SourceImportIds { get; set; }
    public string FileName { get; set; } = "";
    public int? TileX { get; set; }
    public int? TileY { get; set; }
    public int? ZoomLevel { get; set; }
    public string Status { get; set; } = "queued";
    public string? Message { get; set; }
    public string? Output { get; set; }
    public string? OutputPath { get; set; }
    public string? Error { get; set; }
    public long? TotalSteps { get; set; }
    public long CompletedSteps { get; set; }
    public int CurrentStage { get; set; }
    public int TotalStages { get; set; }
    public int? CurrentBatchIndex { get; set; }
    public int? TotalBatches { get; set; }
    public DateTime StartedAt { get; set; }
    public DateTime? CompletedAt { get; set; }
    public double ElapsedSeconds => CompletedAt.HasValue
        ? (CompletedAt.Value - StartedAt).TotalSeconds
        : (DateTime.UtcNow - StartedAt).TotalSeconds;
}

record OsmBuilding(string building, string name, string amenity, string shop, double lon, double lat);
record FeatureStat(string feature_type, long count);
record CorrectionRequest(Guid? TileId, int? FeatureIndex, string CorrectionType, string? OldLabel, string? NewLabel, string? SubmittedBy);
record CorrectionReview(Guid CorrectionId, bool Approved, string? Notes);
record TileProcessImportsRequest(Guid[] ImportIds);
record ImportAlignmentUpdateRequest(bool Clear, double YawDegrees, double OffsetX, double OffsetY, string? Source);
class ImportJob
{
    public Guid Id { get; set; }
    public Guid? ImportId { get; set; }
    public Guid? PipelineJobId { get; set; }
    public string Type { get; set; } = "";
    public string FileName { get; set; } = "";
    public int? TileX { get; set; }
    public int? TileY { get; set; }
    public int? ZoomLevel { get; set; }
    public string Status { get; set; } = "";
    public string? Message { get; set; }
    public long? TotalSteps { get; set; }
    public long CompletedSteps { get; set; }
    public int CurrentStage { get; set; }
    public int TotalStages { get; set; }
    public int? CurrentBatchIndex { get; set; }
    public int? TotalBatches { get; set; }
    public long FileSizeBytes { get; set; }
    public long? UniquePointCount { get; set; }
    public long? NewPointCount { get; set; }
    public long? DuplicatePointCount { get; set; }
    public long? DuplicateWithinImportCount { get; set; }
    public string? Error { get; set; }
    public string? Output { get; set; }
    public string? OutputPath { get; set; }
    public DateTime StartedAt { get; set; }
    public DateTime? CompletedAt { get; set; }
    public double ElapsedSeconds => CompletedAt.HasValue
        ? (CompletedAt.Value - StartedAt).TotalSeconds
        : (DateTime.UtcNow - StartedAt).TotalSeconds;
}

class PipelineStatusPayload
{
    public string Status { get; set; } = "";
    public string Message { get; set; } = "";
    public long? CompletedSteps { get; set; }
    public long? TotalSteps { get; set; }
}

class MlServiceProcessEventPayload
{
    public string Type { get; set; } = "";
    public string? Status { get; set; }
    public string? Message { get; set; }
    public long? CompletedSteps { get; set; }
    public long? TotalSteps { get; set; }
    public PipelineResultPayload? Result { get; set; }
    public string? Detail { get; set; }
}

class PipelineResultPayload
{
    public string GlbPath { get; set; } = "";
    public int Zoom { get; set; }
    public int X { get; set; }
    public int Y { get; set; }
    public int MeshCount { get; set; }
    public int SegmentCount { get; set; }
    public long VertexCount { get; set; }
    public long FaceCount { get; set; }
    public string[] FeatureTypes { get; set; } = [];
    public string? SegmentDataPath { get; set; }
    public string? SegmentMetadataPath { get; set; }
    public double[] SceneOrigin { get; set; } = [];
    public PipelineSegmentPayload[]? Segments { get; set; }
}

class PipelineSegmentPayload
{
    public int LocalId { get; set; }
    public int AsprsClass { get; set; }
    public string FeatureType { get; set; } = "";
    public int PointCount { get; set; }
    public float Confidence { get; set; }
    public double[] BoundsMin { get; set; } = [];
    public double[] BoundsMax { get; set; } = [];
    public double[] Centroid { get; set; } = [];
    public int VertexCount { get; set; }
    public int FaceCount { get; set; }
}

class SegmentOsmMatchPayload
{
    public static SegmentOsmMatchPayload Unmatched => new()
    {
        MatchStatus = "unmatched"
    };

    public long? OsmId { get; set; }
    public string? Name { get; set; }
    public string? FeatureType { get; set; }
    public string MatchStatus { get; set; } = "unmatched";
    public float? MatchScore { get; set; }
}

class PointCloudPreviewPayload
{
    public string FileName { get; set; } = "";
    public int TotalPoints { get; set; }
    public int SampledPoints { get; set; }
    public float[] Points { get; set; } = [];
    public float[]? Colors { get; set; }
    public int[]? SegmentIds { get; set; }
}

class TileAlignmentDebugPayload
{
    public Guid TileId { get; set; }
    public int TileX { get; set; }
    public int TileY { get; set; }
    public int ZoomLevel { get; set; }
    public double CenterLat { get; set; }
    public double CenterLon { get; set; }
    public TileAlignmentMetadataPayload? Alignment { get; set; }
    public string TileBoundsGeoJson { get; set; } = "";
    public TileAlignmentSegmentPayload[] Segments { get; set; } = [];
    public TileAlignmentOsmPayload[] OsmBuildings { get; set; } = [];
    public TileAlignmentVicmapPropertyPayload[] VicmapProperties { get; set; } = [];
    public string? VicmapFetchError { get; set; }
}

class TileAlignmentMetadataPayload
{
    public double YawDegrees { get; set; }
    public double OffsetX { get; set; }
    public double OffsetY { get; set; }
    public string? Source { get; set; }
    public DateTime? UpdatedAt { get; set; }
}

class TileAlignmentSegmentPayload
{
    public int LocalSegmentId { get; set; }
    public string Label { get; set; } = "";
    public string GeometrySource { get; set; } = "scan";
    public bool HasOsmFill { get; set; }
    public long? OsmIdentifier { get; set; }
    public string OsmMatchStatus { get; set; } = "unmatched";
    public double? CentroidLat { get; set; }
    public double? CentroidLon { get; set; }
    public string? BoundsGeoJson { get; set; }
}

class TileAlignmentOsmPayload
{
    public long OsmId { get; set; }
    public string? Name { get; set; }
    public string FeatureType { get; set; } = "building";
    public string? GeometryJson { get; set; }
    public int[] SegmentIds { get; set; } = [];
    public bool IsMatched { get; set; }
}

class TileAlignmentVicmapPropertyPayload
{
    public string? Pfi { get; set; }
    public TileAlignmentLocalPointPayload[][] LocalExteriorRings { get; set; } = [];
}

class TileAlignmentLocalPointPayload
{
    public double X { get; set; }
    public double Y { get; set; }
}

class ScanAlignmentCorrectionPayload
{
    public bool Rotated { get; set; }
    public double RotationDegrees { get; set; }
    public double TranslateX { get; set; }
    public double TranslateY { get; set; }
    public PipelineSegmentPayload[]? Segments { get; set; }
}

class ResolvedAlignmentMetadata
{
    public double YawDegrees { get; set; }
    public double OffsetX { get; set; }
    public double OffsetY { get; set; }
    public string? Source { get; set; }
    public DateTime? UpdatedAt { get; set; }
}

class PreparedImportMetadata
{
    public long TotalPoints { get; set; }
    public long UniquePoints { get; set; }
    public long DuplicateWithinImportPoints { get; set; }
    public string? PointSignature { get; set; }
}

class MlServiceProcessRequest
{
    [JsonPropertyName("input_path")]
    public string InputPath { get; set; } = "";

    [JsonPropertyName("max_points")]
    public int MaxPoints { get; set; }

    [JsonPropertyName("crs")]
    public string Crs { get; set; } = "EPSG:28355";
}

sealed class PreparedImportData(
    string csvPath,
    string metadataPath,
    long totalPoints,
    long uniquePoints,
    long duplicateWithinImportPoints,
    string? pointSignature) : IDisposable
{
    public string CsvPath { get; } = csvPath;
    public string MetadataPath { get; } = metadataPath;
    public long TotalPoints { get; } = totalPoints;
    public long UniquePoints { get; } = uniquePoints;
    public long DuplicateWithinImportPoints { get; } = duplicateWithinImportPoints;
    public string? PointSignature { get; } = pointSignature;

    public void Dispose()
    {
        if (File.Exists(CsvPath))
            File.Delete(CsvPath);
        if (File.Exists(MetadataPath))
            File.Delete(MetadataPath);
    }
}

record ImportMergeResult(bool Rejected, long NewPointCount, long DuplicatePointCount);
sealed record PipelineInputBatch(Guid[] SourceImportIds, string FileName, string Crs);

record SegmentReviewRequest(
    Guid SegmentId,
    string CorrectionType,
    string? RequestedLabel,
    string[]? RelatedSegmentIds,
    string? Notes,
    string? SubmittedBy);

record SegmentReviewDecision(
    Guid ReviewId,
    bool Approved,
    string? Notes);

// ── Vicmap alignment DTOs ────────────────────────────────────────
class VicmapAlignmentResult
{
    [JsonPropertyName("matched_pairs")]
    public int MatchedPairs { get; set; }

    [JsonPropertyName("alignment")]
    public VicmapAlignmentPayload? Alignment { get; set; }
}

class VicmapAlignmentPayload
{
    [JsonPropertyName("yaw_degrees")]
    public double YawDegrees { get; set; }

    [JsonPropertyName("yaw_stdev")]
    public double YawStdev { get; set; }

    [JsonPropertyName("offset_x")]
    public double OffsetX { get; set; }

    [JsonPropertyName("offset_x_stdev")]
    public double OffsetXStdev { get; set; }

    [JsonPropertyName("offset_y")]
    public double OffsetY { get; set; }

    [JsonPropertyName("offset_y_stdev")]
    public double OffsetYStdev { get; set; }
}

record SegmentReviewExportRequest(Guid[] ReviewIds);
