using Microsoft.EntityFrameworkCore;
using Npgsql;

namespace LibreRally.Maps.TileServer;

/// <summary>
/// Runs database initialization on startup.
/// Reads embedded 01-schema.sql and executes it against mapsdb.
/// All statements use IF NOT EXISTS — safe to run multiple times.
/// </summary>
public class DbInitializer
{
    public static async Task InitializeAsync(WebApplication app)
    {
        using var scope = app.Services.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<MapsDbContext>();
        var logger = app.Services.GetRequiredService<ILogger<DbInitializer>>();

        try
        {
            // Read embedded schema SQL
            var assembly = typeof(DbInitializer).Assembly;
            var resourceNames = assembly.GetManifestResourceNames();
            logger.LogInformation("Available embedded resources: {Resources}", string.Join(", ", resourceNames));
            var resourceName = resourceNames.FirstOrDefault(n => n.Contains("01-schema") || n.Contains("Schema.sql"));

            if (resourceName is null)
            {
                logger.LogWarning("Schema SQL not found as embedded resource");
                return;
            }

            await using var stream = assembly.GetManifestResourceStream(resourceName);
            using var reader = new StreamReader(stream!);
            var sql = await reader.ReadToEndAsync();

            var conn = db.Database.GetDbConnection();
            await conn.OpenAsync();

            await using var cmd = conn.CreateCommand();
            cmd.CommandText = sql;

            await cmd.ExecuteNonQueryAsync();
            logger.LogInformation("Database schema initialized successfully");
        }
        catch (PostgresException ex) when (ex.Message.Contains("already exists"))
        {
            logger.LogInformation("Database schema already exists, skipping init");
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Database initialization failed: {Message}", ex.Message);
            throw;
        }
    }
}
