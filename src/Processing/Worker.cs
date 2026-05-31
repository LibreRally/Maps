using Npgsql;

namespace LibreRally.Maps.Processing;

/// <summary>
/// Background worker that monitors for new point cloud data and triggers processing.
/// </summary>
public class Worker(ILogger<Worker> logger, NpgsqlDataSource dataSource) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        logger.LogInformation("Point cloud processing worker started");

        // Ensure pgpointcloud extension is available
        await using var conn = await dataSource.OpenConnectionAsync(stoppingToken);
        await using var cmd = new NpgsqlCommand(
            "SELECT pc_version();", conn);
        
        try
        {
            var version = await cmd.ExecuteScalarAsync(stoppingToken);
            logger.LogInformation("pgpointcloud version: {Version}", version);
        }
        catch (PostgresException ex)
        {
            logger.LogWarning("pgpointcloud extension not available (SqlState {SqlState}): {Message}. The pointcloud extension is not installed in this PostgreSQL container. Install with: CREATE EXTENSION pointcloud;", ex.SqlState, ex.Message);
        }

        while (!stoppingToken.IsCancellationRequested)
        {
            await ProcessPendingTiles(stoppingToken);
            await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken);
        }
    }

    private async Task ProcessPendingTiles(CancellationToken ct)
    {
        // TODO: Query pointcloud.tiles for unprocessed tiles
        // TODO: Run PDAL pipeline via subprocess or Python bridge
        // TODO: Classify points (ground/non-ground/clusters)
        // TODO: Spatial join with OSM data
        // TODO: Generate 3D tiles and update catalog
        logger.LogDebug("Polling for pending tiles...");
    }
}
