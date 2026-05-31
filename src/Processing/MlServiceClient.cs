using System.Net.Http.Json;
using System.Text.Json.Serialization;

namespace LibreRally.Maps.Processing;

/// <summary>
/// HTTP client for the Python ML service.
/// </summary>
public class MlServiceClient(HttpClient http)
{
    private readonly HttpClient _http = http;

    /// <summary>Health check.</summary>
    public async Task<bool> IsHealthyAsync(CancellationToken ct = default)
    {
        try
        {
            var response = await _http.GetAsync("/health", ct);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>Classify a point cloud.</summary>
    public async Task<ClassificationResult?> ClassifyAsync(
        double[][] points, string crs = "EPSG:28355", CancellationToken ct = default)
    {
        var request = new { points, crs };
        var response = await _http.PostAsJsonAsync("/classify", request, ct);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<ClassificationResult>(cancellationToken: ct);
    }

    /// <summary>Reconstruct mesh from classified points.</summary>
    public async Task<MeshResult?> ReconstructAsync(
        double[][] points, int[] labels, string method = "alpha_shape", CancellationToken ct = default)
    {
        var request = new { points, labels, method };
        var response = await _http.PostAsJsonAsync("/reconstruct", request, ct);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<MeshResult>(cancellationToken: ct);
    }
}

// ── DTOs matching Python service response models ─────────────────

public class ClassificationResult
{
    [JsonPropertyName("labels")]
    public int[] Labels { get; set; } = [];

    [JsonPropertyName("confidence")]
    public double[] Confidence { get; set; } = [];

    [JsonPropertyName("feature_types")]
    public string[] FeatureTypes { get; set; } = [];

    [JsonPropertyName("source")]
    public string Source { get; set; } = "";
}

public class MeshResult
{
    [JsonPropertyName("vertices")]
    public double[][] Vertices { get; set; } = [];

    [JsonPropertyName("faces")]
    public int[][] Faces { get; set; } = [];

    [JsonPropertyName("vertex_count")]
    public int VertexCount { get; set; }

    [JsonPropertyName("face_count")]
    public int FaceCount { get; set; }
}
