using System.Net.Http.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Components.Forms;

namespace LibreRally.Maps.WebUI;

/// <summary>
/// HTTP client for the TileServer API.
/// </summary>
public class TileApiClient(HttpClient http)
{
    // ── Health ───────────────────────────────────────────────────
    public async Task<bool> IsHealthyAsync() =>
        (await http.GetAsync("/"))?.IsSuccessStatusCode ?? false;

    // ── Tiles ────────────────────────────────────────────────────
    public async Task<List<TileEntry>> GetTileCatalogAsync()
    {
        var result = await http.GetFromJsonAsync<List<TileEntry>>("/api/tiles/catalog");
        return result ?? [];
    }

    public async Task<TileGenerateResult?> GenerateTileAsync(TileGenerateRequest req)
    {
        var response = await http.PostAsJsonAsync("/api/tiles/generate", req);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<TileGenerateResult>();
    }

    // ── OSM ──────────────────────────────────────────────────────
    public async Task<List<OsmFeatureItem>> GetFeaturesAsync(
        double west, double south, double east, double north)
    {
        var result = await http.GetFromJsonAsync<List<OsmFeatureItem>>(
            $"/api/osm/features?west={west}&south={south}&east={east}&north={north}");
        return result ?? [];
    }

    public async Task<List<OsmStatItem>> GetStatsAsync(
        double west, double south, double east, double north)
    {
        var result = await http.GetFromJsonAsync<List<OsmStatItem>>(
            $"/api/osm/stats?west={west}&south={south}&east={east}&north={north}");
        return result ?? [];
    }

    // ── Corrections ──────────────────────────────────────────────
    public async Task<List<CorrectionItem>> GetCorrectionsAsync(bool? reviewed = null)
    {
        var url = "/api/corrections";
        if (reviewed.HasValue) url += $"?reviewed={reviewed.Value.ToString().ToLower()}";
        var result = await http.GetFromJsonAsync<List<CorrectionItem>>(url);
        return result ?? [];
    }

    public async Task SubmitCorrectionAsync(CorrectionRequest req)
    {
        var response = await http.PostAsJsonAsync("/api/corrections", req);
        response.EnsureSuccessStatusCode();
    }

    public async Task ReviewCorrectionAsync(Guid id, bool approved, string? notes = null)
    {
        var response = await http.PostAsJsonAsync("/api/corrections/review",
            new { correctionId = id, approved, notes });
        response.EnsureSuccessStatusCode();
    }

    public async Task<TrainingDataSummary?> GetTrainingDataAsync()
    {
        return await http.GetFromJsonAsync<TrainingDataSummary>("/api/corrections/training");
    }

    // ── Import ────────────────────────────────────────────────────
    public async Task<List<ImportJobDisplay>> GetImportJobsAsync()
    {
        return await http.GetFromJsonAsync<List<ImportJobDisplay>>("/api/import/jobs") ?? [];
    }

    public async Task<List<ImportRecordItem>> GetImportsAsync(bool includeDeleted = false)
    {
        var suffix = includeDeleted ? "?includeDeleted=true" : string.Empty;
        return await http.GetFromJsonAsync<List<ImportRecordItem>>($"/api/imports{suffix}") ?? [];
    }

    public async Task DeleteImportAsync(Guid importId)
    {
        var response = await http.DeleteAsync($"/api/imports/{importId}");
        response.EnsureSuccessStatusCode();
    }

    public async Task UpdateImportAlignmentAsync(Guid importId, ImportAlignmentUpdateRequest req)
    {
        var response = await http.PostAsJsonAsync($"/api/imports/{importId}/alignment", req);
        response.EnsureSuccessStatusCode();
    }

    public async Task<ImportStartResult> UploadLasAsync(IBrowserFile file)
    {
        // Copy to MemoryStream so the stream can be re-read on retry
        using var ms = new MemoryStream();
        await using var fs = file.OpenReadStream(500_000_000);
        await fs.CopyToAsync(ms);
        ms.Position = 0;

        using var content = new MultipartFormDataContent();
        var streamContent = new StreamContent(ms);
        streamContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/octet-stream");
        content.Add(streamContent, "file", file.Name);
        var response = await http.PostAsync("/api/import/las", content);
        response.EnsureSuccessStatusCode();
        var result = await response.Content.ReadFromJsonAsync<ImportStartResult>();
        return result ?? new ImportStartResult();
    }

    public async Task ImportOsmAsync(string path)
    {
        var content = new MultipartFormDataContent();
        content.Add(new StringContent(path), "path");
        var response = await http.PostAsync("/api/import/osm", content);
        response.EnsureSuccessStatusCode();
    }

    // ── Pipeline / Processing ─────────────────────────────────────
    public async Task<ProcessStartResult> StartPipelineAsync()
    {
        var response = await http.PostAsync("/api/tiles/process", null);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<ProcessStartResult>() ?? new ProcessStartResult();
    }

    public async Task<ProcessStartResult> StartPipelineForImportsAsync(Guid[] importIds)
    {
        var response = await http.PostAsJsonAsync("/api/tiles/process/imports", new TileProcessImportsRequest(importIds));
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<ProcessStartResult>() ?? new ProcessStartResult();
    }

    public async Task<List<PipelineJob>> GetPipelineJobsAsync()
    {
        return await http.GetFromJsonAsync<List<PipelineJob>>("/api/tiles/process") ?? [];
    }

    public async Task<PointCloudPreview?> GetPointCloudPreviewAsync(Guid tileId)
    {
        var response = await http.GetAsync($"/api/tiles/{tileId}/pointcloud");
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<PointCloudPreview>();
    }

    public async Task<List<TileSegmentItem>> GetTileSegmentsAsync(Guid tileId)
    {
        return await http.GetFromJsonAsync<List<TileSegmentItem>>($"/api/tiles/{tileId}/segments") ?? [];
    }

    public async Task<TileAlignmentDebug?> GetTileAlignmentDebugAsync(Guid tileId)
    {
        return await http.GetFromJsonAsync<TileAlignmentDebug>($"/api/tiles/{tileId}/alignment");
    }

    public async Task<List<TileSegmentItem>> GetSegmentsAsync()
    {
        return await http.GetFromJsonAsync<List<TileSegmentItem>>("/api/segments") ?? [];
    }

    public async Task<List<SegmentReviewItem>> GetSegmentReviewsAsync(bool? reviewed = null)
    {
        var url = "/api/segments/reviews";
        if (reviewed.HasValue) url += $"?reviewed={reviewed.Value.ToString().ToLower()}";
        return await http.GetFromJsonAsync<List<SegmentReviewItem>>(url) ?? [];
    }

    public async Task SubmitSegmentReviewAsync(SegmentReviewRequest req)
    {
        var response = await http.PostAsJsonAsync("/api/segments/reviews", req);
        response.EnsureSuccessStatusCode();
    }

    public async Task ReviewSegmentAsync(Guid reviewId, bool approved, string? notes = null)
    {
        var response = await http.PostAsJsonAsync("/api/segments/reviews/review", new { reviewId, approved, notes });
        response.EnsureSuccessStatusCode();
    }

    public async Task<SegmentTrainingSummary?> GetSegmentTrainingSummaryAsync()
    {
        return await http.GetFromJsonAsync<SegmentTrainingSummary>("/api/segments/reviews/training");
    }
}

// ── DTOs ─────────────────────────────────────────────────────────

public record TileEntry
{
    [JsonPropertyName("id")] public Guid Id { get; set; }
    [JsonPropertyName("tileX")] public int? TileX { get; set; }
    [JsonPropertyName("tileY")] public int? TileY { get; set; }
    [JsonPropertyName("zoomLevel")] public int? ZoomLevel { get; set; }
    [JsonPropertyName("featureTypes")] public string[]? FeatureTypes { get; set; }
    [JsonPropertyName("meshCount")] public int? MeshCount { get; set; }
    [JsonPropertyName("segmentCount")] public int? SegmentCount { get; set; }
    [JsonPropertyName("tilesetPath")] public string? TilesetPath { get; set; }
    [JsonPropertyName("sourceTileIds")] public int[]? SourceTileIds { get; set; }
    [JsonPropertyName("sourceImportIds")] public Guid[]? SourceImportIds { get; set; }
    [JsonPropertyName("createdAt")] public DateTime CreatedAt { get; set; }
}

public record TileGenerateRequest(
    int TileX, int TileY,
    double MinLon, double MinLat, double MaxLon, double MaxLat,
    List<ClusterData> Clusters);

public record ClusterData(
    string FeatureType, int AsprsClass,
    List<double[]> Vertices, List<int[]> Faces, string? OsmName);

public record TileGenerateResult
{
    [JsonPropertyName("id")] public Guid Id { get; set; }
    [JsonPropertyName("tilesetPath")] public string? TilesetPath { get; set; }
    [JsonPropertyName("clusters")] public int Clusters { get; set; }
}

public record OsmFeatureItem
{
    [JsonPropertyName("building")] public string? Building { get; set; }
    [JsonPropertyName("name")] public string? Name { get; set; }
    [JsonPropertyName("amenity")] public string? Amenity { get; set; }
    [JsonPropertyName("shop")] public string? Shop { get; set; }
    [JsonPropertyName("lon")] public double Lon { get; set; }
    [JsonPropertyName("lat")] public double Lat { get; set; }
}

public record OsmStatItem
{
    [JsonPropertyName("feature_type")] public string FeatureType { get; set; } = "";
    [JsonPropertyName("count")] public long Count { get; set; }
}

public record CorrectionItem
{
    [JsonPropertyName("id")] public Guid Id { get; set; }
    [JsonPropertyName("tileId")] public Guid? TileId { get; set; }
    [JsonPropertyName("featureIndex")] public int? FeatureIndex { get; set; }
    [JsonPropertyName("correctionType")] public string? CorrectionType { get; set; }
    [JsonPropertyName("oldLabel")] public string? OldLabel { get; set; }
    [JsonPropertyName("newLabel")] public string? NewLabel { get; set; }
    [JsonPropertyName("submittedBy")] public string? SubmittedBy { get; set; }
    [JsonPropertyName("submittedAt")] public DateTime SubmittedAt { get; set; }
    [JsonPropertyName("reviewed")] public bool Reviewed { get; set; }
    [JsonPropertyName("approved")] public bool Approved { get; set; }
}

public record CorrectionRequest(
    Guid? TileId, int? FeatureIndex, string CorrectionType,
    string? OldLabel, string? NewLabel, string? SubmittedBy);

public record ImportAlignmentUpdateRequest(
    bool Clear,
    double YawDegrees,
    double OffsetX,
    double OffsetY,
    string? Source);

public record TileProcessImportsRequest(Guid[] ImportIds);

public record TrainingDataSummary
{
    [JsonPropertyName("count")] public int Count { get; set; }
    [JsonPropertyName("corrections")] public List<CorrectionItem>? Corrections { get; set; }
}

public class ImportJobDisplay
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
    public double ElapsedSeconds { get; set; }
    public long? UniquePointCount { get; set; }
    public long? NewPointCount { get; set; }
    public long? DuplicatePointCount { get; set; }
    public long? DuplicateWithinImportCount { get; set; }
    public string? Error { get; set; }
    public string? OutputPath { get; set; }
}

public class ImportRecordItem
{
    [JsonPropertyName("id")] public Guid Id { get; set; }
    [JsonPropertyName("originalFileName")] public string OriginalFileName { get; set; } = "";
    [JsonPropertyName("source")] public string Source { get; set; } = "";
    [JsonPropertyName("totalPointCount")] public long? TotalPointCount { get; set; }
    [JsonPropertyName("uniquePointCount")] public long? UniquePointCount { get; set; }
    [JsonPropertyName("newPointCount")] public long? NewPointCount { get; set; }
    [JsonPropertyName("duplicatePointCount")] public long? DuplicatePointCount { get; set; }
    [JsonPropertyName("duplicateWithinImportCount")] public long? DuplicateWithinImportCount { get; set; }
    [JsonPropertyName("fileSizeBytes")] public long? FileSizeBytes { get; set; }
    [JsonPropertyName("status")] public string Status { get; set; } = "";
    [JsonPropertyName("rejectionReason")] public string? RejectionReason { get; set; }
    [JsonPropertyName("alignmentYawDegrees")] public double? AlignmentYawDegrees { get; set; }
    [JsonPropertyName("alignmentOffsetX")] public double? AlignmentOffsetX { get; set; }
    [JsonPropertyName("alignmentOffsetY")] public double? AlignmentOffsetY { get; set; }
    [JsonPropertyName("alignmentSource")] public string? AlignmentSource { get; set; }
    [JsonPropertyName("alignmentUpdatedAt")] public DateTime? AlignmentUpdatedAt { get; set; }
    [JsonPropertyName("importedAt")] public DateTime ImportedAt { get; set; }
    [JsonPropertyName("deletedAt")] public DateTime? DeletedAt { get; set; }
}

public class ImportStartResult
{
    [JsonPropertyName("id")] public Guid Id { get; set; }
    [JsonPropertyName("importId")] public Guid? ImportId { get; set; }
    [JsonPropertyName("status")] public string Status { get; set; } = "";
    [JsonPropertyName("message")] public string? Message { get; set; }
    [JsonPropertyName("pipelineJobId")] public Guid? PipelineJobId { get; set; }
}

public class ProcessResult
{
    [JsonPropertyName("status")] public string Status { get; set; } = "";
    [JsonPropertyName("message")] public string? Message { get; set; }
    [JsonPropertyName("output")] public string? Output { get; set; }
    [JsonPropertyName("error")] public string? Error { get; set; }
    [JsonPropertyName("exitCode")] public int ExitCode { get; set; }
}

public class ProcessStartResult
{
    [JsonPropertyName("jobId")] public Guid JobId { get; set; }
    [JsonPropertyName("status")] public string Status { get; set; } = "";
    [JsonPropertyName("fileName")] public string? FileName { get; set; }
    [JsonPropertyName("message")] public string? Message { get; set; }
}

public class PipelineJob
{
    [JsonPropertyName("id")] public Guid Id { get; set; }
    [JsonPropertyName("importJobId")] public Guid? ImportJobId { get; set; }
    [JsonPropertyName("sourceImportIds")] public Guid[]? SourceImportIds { get; set; }
    [JsonPropertyName("fileName")] public string FileName { get; set; } = "";
    [JsonPropertyName("tileX")] public int? TileX { get; set; }
    [JsonPropertyName("tileY")] public int? TileY { get; set; }
    [JsonPropertyName("zoomLevel")] public int? ZoomLevel { get; set; }
    [JsonPropertyName("status")] public string Status { get; set; } = "";
    [JsonPropertyName("message")] public string? Message { get; set; }
    [JsonPropertyName("output")] public string? Output { get; set; }
    [JsonPropertyName("outputPath")] public string? OutputPath { get; set; }
    [JsonPropertyName("error")] public string? Error { get; set; }
    [JsonPropertyName("totalSteps")] public long? TotalSteps { get; set; }
    [JsonPropertyName("completedSteps")] public long CompletedSteps { get; set; }
    [JsonPropertyName("currentStage")] public int CurrentStage { get; set; }
    [JsonPropertyName("totalStages")] public int TotalStages { get; set; }
    [JsonPropertyName("currentBatchIndex")] public int? CurrentBatchIndex { get; set; }
    [JsonPropertyName("totalBatches")] public int? TotalBatches { get; set; }
    [JsonPropertyName("elapsedSeconds")] public double ElapsedSeconds { get; set; }
}

public class PointCloudPreview
{
    [JsonPropertyName("fileName")] public string FileName { get; set; } = "";
    [JsonPropertyName("totalPoints")] public int TotalPoints { get; set; }
    [JsonPropertyName("sampledPoints")] public int SampledPoints { get; set; }
    [JsonPropertyName("points")] public float[] Points { get; set; } = [];
    [JsonPropertyName("colors")] public float[]? Colors { get; set; }
    [JsonPropertyName("segmentIds")] public int[]? SegmentIds { get; set; }
}

public class TileAlignmentDebug
{
    [JsonPropertyName("tileId")] public Guid TileId { get; set; }
    [JsonPropertyName("tileX")] public int TileX { get; set; }
    [JsonPropertyName("tileY")] public int TileY { get; set; }
    [JsonPropertyName("zoomLevel")] public int ZoomLevel { get; set; }
    [JsonPropertyName("centerLat")] public double CenterLat { get; set; }
    [JsonPropertyName("centerLon")] public double CenterLon { get; set; }
    [JsonPropertyName("alignment")] public TileAlignmentInfo? Alignment { get; set; }
    [JsonPropertyName("tileBoundsGeoJson")] public string TileBoundsGeoJson { get; set; } = "";
    [JsonPropertyName("segments")] public List<TileAlignmentSegment> Segments { get; set; } = [];
    [JsonPropertyName("osmBuildings")] public List<TileAlignmentOsmBuilding> OsmBuildings { get; set; } = [];
    [JsonPropertyName("vicmapProperties")] public List<TileAlignmentVicmapProperty> VicmapProperties { get; set; } = [];
    [JsonPropertyName("vicmapFetchError")] public string? VicmapFetchError { get; set; }
}

public class TileAlignmentInfo
{
    [JsonPropertyName("yawDegrees")] public double YawDegrees { get; set; }
    [JsonPropertyName("offsetX")] public double OffsetX { get; set; }
    [JsonPropertyName("offsetY")] public double OffsetY { get; set; }
    [JsonPropertyName("source")] public string? Source { get; set; }
    [JsonPropertyName("updatedAt")] public DateTime? UpdatedAt { get; set; }
}

public class TileAlignmentSegment
{
    [JsonPropertyName("localSegmentId")] public int LocalSegmentId { get; set; }
    [JsonPropertyName("label")] public string Label { get; set; } = "";
    [JsonPropertyName("geometrySource")] public string GeometrySource { get; set; } = "scan";
    [JsonPropertyName("hasOsmFill")] public bool HasOsmFill { get; set; }
    [JsonPropertyName("osmIdentifier")] public long? OsmIdentifier { get; set; }
    [JsonPropertyName("osmMatchStatus")] public string OsmMatchStatus { get; set; } = "unmatched";
    [JsonPropertyName("centroidLat")] public double? CentroidLat { get; set; }
    [JsonPropertyName("centroidLon")] public double? CentroidLon { get; set; }
    [JsonPropertyName("boundsGeoJson")] public string? BoundsGeoJson { get; set; }
}

public class TileAlignmentOsmBuilding
{
    [JsonPropertyName("osmId")] public long OsmId { get; set; }
    [JsonPropertyName("name")] public string? Name { get; set; }
    [JsonPropertyName("featureType")] public string FeatureType { get; set; } = "building";
    [JsonPropertyName("geometryJson")] public string? GeometryJson { get; set; }
    [JsonPropertyName("segmentIds")] public int[] SegmentIds { get; set; } = [];
    [JsonPropertyName("isMatched")] public bool IsMatched { get; set; }
}

public class TileAlignmentVicmapProperty
{
    [JsonPropertyName("pfi")] public string? Pfi { get; set; }
    [JsonPropertyName("localExteriorRings")] public List<List<TileAlignmentLocalPoint>> LocalExteriorRings { get; set; } = [];
}

public class TileAlignmentLocalPoint
{
    [JsonPropertyName("x")] public double X { get; set; }
    [JsonPropertyName("y")] public double Y { get; set; }
}

public class TileSegmentItem
{
    [JsonPropertyName("id")] public Guid Id { get; set; }
    [JsonPropertyName("tileId")] public Guid TileId { get; set; }
    [JsonPropertyName("sourceTileId")] public int? SourceTileId { get; set; }
    [JsonPropertyName("sourceImportIds")] public Guid[]? SourceImportIds { get; set; }
    [JsonPropertyName("localSegmentId")] public int LocalSegmentId { get; set; }
    [JsonPropertyName("predictedLabel")] public string PredictedLabel { get; set; } = "";
    [JsonPropertyName("reviewedLabel")] public string? ReviewedLabel { get; set; }
    [JsonPropertyName("pointCount")] public int PointCount { get; set; }
    [JsonPropertyName("confidence")] public float? Confidence { get; set; }
    [JsonPropertyName("osmFeatureType")] public string? OsmFeatureType { get; set; }
    [JsonPropertyName("osmName")] public string? OsmName { get; set; }
    [JsonPropertyName("osmIdentifier")] public long? OsmIdentifier { get; set; }
    [JsonPropertyName("osmMatchStatus")] public string OsmMatchStatus { get; set; } = "";
    [JsonPropertyName("osmMatchScore")] public float? OsmMatchScore { get; set; }
    [JsonPropertyName("geometrySource")] public string GeometrySource { get; set; } = "scan";
    [JsonPropertyName("hasOsmFill")] public bool HasOsmFill { get; set; }
    [JsonPropertyName("createdAt")] public DateTime CreatedAt { get; set; }
    [JsonPropertyName("updatedAt")] public DateTime UpdatedAt { get; set; }
}

public class SegmentReviewItem
{
    [JsonPropertyName("id")] public Guid Id { get; set; }
    [JsonPropertyName("segmentId")] public Guid SegmentId { get; set; }
    [JsonPropertyName("tileId")] public Guid TileId { get; set; }
    [JsonPropertyName("localSegmentId")] public int LocalSegmentId { get; set; }
    [JsonPropertyName("predictedLabel")] public string PredictedLabel { get; set; } = "";
    [JsonPropertyName("reviewedLabel")] public string? ReviewedLabel { get; set; }
    [JsonPropertyName("correctionType")] public string CorrectionType { get; set; } = "";
    [JsonPropertyName("previousLabel")] public string? PreviousLabel { get; set; }
    [JsonPropertyName("requestedLabel")] public string? RequestedLabel { get; set; }
    [JsonPropertyName("relatedSegmentIds")] public string[]? RelatedSegmentIds { get; set; }
    [JsonPropertyName("notes")] public string? Notes { get; set; }
    [JsonPropertyName("submittedBy")] public string? SubmittedBy { get; set; }
    [JsonPropertyName("submittedAt")] public DateTime SubmittedAt { get; set; }
    [JsonPropertyName("reviewed")] public bool Reviewed { get; set; }
    [JsonPropertyName("approved")] public bool Approved { get; set; }
    [JsonPropertyName("reviewedAt")] public DateTime? ReviewedAt { get; set; }
    [JsonPropertyName("reviewerNotes")] public string? ReviewerNotes { get; set; }
    [JsonPropertyName("exportedAt")] public DateTime? ExportedAt { get; set; }
}

public record SegmentReviewRequest(
    Guid SegmentId,
    string CorrectionType,
    string? RequestedLabel,
    string[]? RelatedSegmentIds,
    string? Notes,
    string? SubmittedBy);

public class SegmentTrainingSummary
{
    [JsonPropertyName("count")] public int Count { get; set; }
    [JsonPropertyName("exportedCount")] public int ExportedCount { get; set; }
    [JsonPropertyName("reviews")] public List<SegmentReviewItem>? Reviews { get; set; }
}
