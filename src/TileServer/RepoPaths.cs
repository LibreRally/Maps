namespace LibreRally.Maps.TileServer;

internal static class RepoPaths
{
    private static readonly string RepoRootPath = Path.GetFullPath(Path.Combine(
        AppContext.BaseDirectory, "..", "..", "..", "..", ".."));

    public static string RepoRoot => RepoRootPath;

    public static string DataRoot => FromRoot("data");

    public static string RawData => FromRoot("data", "raw");

    public static string TilesData => FromRoot("data", "tiles");

    public static string PipelineInputData => FromRoot("data", "pipeline-input");

    public static string ModelsData => FromRoot("data", "models");

    public static string MlService => FromRoot("src", "ml-service");

    public static string FromRoot(params string[] segments)
    {
        var parts = new string[segments.Length + 1];
        parts[0] = RepoRootPath;
        Array.Copy(segments, 0, parts, 1, segments.Length);
        return Path.GetFullPath(Path.Combine(parts));
    }

    public static string RelativeToRoot(string path) => Path.GetRelativePath(RepoRootPath, path);

    /// <summary>
    /// Converts a host filesystem path to the corresponding path inside the
    /// mlservice Docker container (where <c>data/</c> is bind-mounted at <c>/data</c>).
    /// </summary>
    public static string ToContainerPath(string hostPath)
    {
        var relative = RelativeToRoot(hostPath).Replace('\\', '/');
        return "/" + relative;
    }
}
