using Microsoft.EntityFrameworkCore;

namespace LibreRally.Maps.TileServer;

public class MapsDbContext(DbContextOptions<MapsDbContext> options) : DbContext(options)
{
    // Import-first point cloud ingest records
    public DbSet<PointCloudImport> PointCloudImports => Set<PointCloudImport>();
    public DbSet<PointCloudPoint> PointCloudPoints => Set<PointCloudPoint>();
    public DbSet<PointCloudImportPoint> PointCloudImportPoints => Set<PointCloudImportPoint>();

    // Point cloud tiles
    public DbSet<PointCloudTile> PointCloudTiles => Set<PointCloudTile>();

    // 3D Tile catalog
    public DbSet<TileCatalogEntry> TileCatalog => Set<TileCatalogEntry>();

    // Generated feature segments
    public DbSet<TileSegment> TileSegments => Set<TileSegment>();

    // Segment review workflow
    public DbSet<SegmentReview> SegmentReviews => Set<SegmentReview>();

    // Feedback / Corrections
    public DbSet<TileCorrection> Corrections => Set<TileCorrection>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.HasDefaultSchema("public");

        modelBuilder.Entity<PointCloudTile>(entity =>
        {
            entity.ToTable("tiles", "pointcloud");
        });

        modelBuilder.Entity<PointCloudImport>(entity =>
        {
            entity.ToTable("imports", "pointcloud");
        });

        modelBuilder.Entity<PointCloudPoint>(entity =>
        {
            entity.ToTable("points", "pointcloud");
        });

        modelBuilder.Entity<PointCloudImportPoint>(entity =>
        {
            entity.ToTable("import_points", "pointcloud");
            entity.HasKey(link => new { link.ImportId, link.PointId });
        });

        modelBuilder.Entity<TileCatalogEntry>(entity =>
        {
            entity.ToTable("catalog", "tiles");
        });

        modelBuilder.Entity<TileSegment>(entity =>
        {
            entity.ToTable("segments", "tiles");
        });

        modelBuilder.Entity<SegmentReview>(entity =>
        {
            entity.ToTable("segment_reviews", "tiles");
        });

        modelBuilder.Entity<TileCorrection>(entity =>
        {
            entity.ToTable("corrections", "tiles");
        });
    }
}

public class PointCloudImport
{
    public Guid Id { get; set; }
    public string OriginalFileName { get; set; } = "";
    public string? StoredFilePath { get; set; }
    public string Source { get; set; } = "";
    public int Crs { get; set; }
    public string? PointSignature { get; set; }
    public long? TotalPointCount { get; set; }
    public long? UniquePointCount { get; set; }
    public long? NewPointCount { get; set; }
    public long? DuplicatePointCount { get; set; }
    public long? DuplicateWithinImportCount { get; set; }
    public long? FileSizeBytes { get; set; }
    public string Status { get; set; } = "pending";
    public string? RejectionReason { get; set; }
    public double? AlignmentYawDegrees { get; set; }
    public double? AlignmentOffsetX { get; set; }
    public double? AlignmentOffsetY { get; set; }
    public string? AlignmentSource { get; set; }
    public DateTime? AlignmentUpdatedAt { get; set; }
    public DateTime ImportedAt { get; set; }
    public DateTime? DeletedAt { get; set; }
}

public class PointCloudPoint
{
    public long Id { get; set; }
    public double X { get; set; }
    public double Y { get; set; }
    public double Z { get; set; }
    public int? Red { get; set; }
    public int? Green { get; set; }
    public int? Blue { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime UpdatedAt { get; set; }
}

public class PointCloudImportPoint
{
    public Guid ImportId { get; set; }
    public long PointId { get; set; }
}

public class PointCloudTile
{
    public int Id { get; set; }
    public int TileX { get; set; }
    public int TileY { get; set; }
    public string Source { get; set; } = "";
    public int Crs { get; set; }
    public long? PointCount { get; set; }
    public long? FileSizeBytes { get; set; }
    public DateTime IngestedAt { get; set; }
}

public class TileCatalogEntry
{
    public Guid Id { get; set; }
    public string TilesetPath { get; set; } = "";
    public int? TileX { get; set; }
    public int? TileY { get; set; }
    public int? ZoomLevel { get; set; }
    public string[]? FeatureTypes { get; set; }
    public int? MeshCount { get; set; }
    public long? VertexCount { get; set; }
    public long? FileSizeBytes { get; set; }
    public float? LodMin { get; set; }
    public float? LodMax { get; set; }
    public int? SegmentCount { get; set; }
    public string? SegmentsPath { get; set; }
    public int[]? SourceTileIds { get; set; }
    public Guid[]? SourceImportIds { get; set; }
    public double? AlignmentYawDegrees { get; set; }
    public double? AlignmentOffsetX { get; set; }
    public double? AlignmentOffsetY { get; set; }
    public string? AlignmentSource { get; set; }
    public DateTime? AlignmentUpdatedAt { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime UpdatedAt { get; set; }
}

public class TileSegment
{
    public Guid Id { get; set; }
    public Guid TileId { get; set; }
    public int? SourceTileId { get; set; }
    public Guid[]? SourceImportIds { get; set; }
    public int LocalSegmentId { get; set; }
    public string PredictedLabel { get; set; } = "";
    public string? ReviewedLabel { get; set; }
    public int PointCount { get; set; }
    public float? Confidence { get; set; }
    public string? OsmFeatureType { get; set; }
    public string? OsmName { get; set; }
    public long? OsmIdentifier { get; set; }
    public string OsmMatchStatus { get; set; } = "unmatched";
    public float? OsmMatchScore { get; set; }
    public string? PreviewPath { get; set; }
    public string? ArtifactPath { get; set; }
    public string GeometrySource { get; set; } = "scan";
    public bool HasOsmFill { get; set; }
    public double? BoundsMinX { get; set; }
    public double? BoundsMinY { get; set; }
    public double? BoundsMinZ { get; set; }
    public double? BoundsMaxX { get; set; }
    public double? BoundsMaxY { get; set; }
    public double? BoundsMaxZ { get; set; }
    public double? CentroidX { get; set; }
    public double? CentroidY { get; set; }
    public double? CentroidZ { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime UpdatedAt { get; set; }
}

public class SegmentReview
{
    public Guid Id { get; set; }
    public Guid SegmentId { get; set; }
    public string CorrectionType { get; set; } = "";
    public string? PreviousLabel { get; set; }
    public string? RequestedLabel { get; set; }
    public string[]? RelatedSegmentIds { get; set; }
    public string? Notes { get; set; }
    public string? SubmittedBy { get; set; }
    public DateTime SubmittedAt { get; set; }
    public bool Reviewed { get; set; }
    public bool Approved { get; set; }
    public DateTime? ReviewedAt { get; set; }
    public string? ReviewerNotes { get; set; }
    public DateTime? ExportedAt { get; set; }
}

public class TileCorrection
{
    public Guid Id { get; set; }
    public Guid? TileId { get; set; }
    public int? FeatureIndex { get; set; }
    public string CorrectionType { get; set; } = "";
    public string? OldLabel { get; set; }
    public string? NewLabel { get; set; }
    public string? SubmittedBy { get; set; }
    public DateTime SubmittedAt { get; set; }
    public bool Reviewed { get; set; }
    public bool Approved { get; set; }
    public DateTime? ReviewedAt { get; set; }
    public string? ReviewerNotes { get; set; }
}
