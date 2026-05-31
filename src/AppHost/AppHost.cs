using Microsoft.Extensions.Hosting;

var builder = DistributedApplication.CreateBuilder(args);
var dataRoot = Path.GetFullPath(Path.Combine(builder.Environment.ContentRootPath, "..", "..", "data"));
var rawDataRoot = Path.Combine(dataRoot, "raw");
var tilesDataRoot = Path.Combine(dataRoot, "tiles");
var pipelineInputRoot = Path.Combine(dataRoot, "pipeline-input");
var modelRoot = Path.Combine(dataRoot, "models");
var mlServiceRoot = Path.GetFullPath(Path.Combine(builder.Environment.ContentRootPath, "..", "ml-service"));
const string mlServiceImageTag = "librerallymaps-mlservice:local";
var gpuEnabled = true; // RTX 3080 available — always enable GPU acceleration

Directory.CreateDirectory(rawDataRoot);
Directory.CreateDirectory(tilesDataRoot);
Directory.CreateDirectory(pipelineInputRoot);
Directory.CreateDirectory(modelRoot);

// ── PostgreSQL with pgpointcloud + PostGIS ──────────────────────
// Uses custom Dockerfile that adds pgpointcloud extension
var postgres = builder.AddPostgres("pgpointcloud")
    .WithDockerfile("../../db")
    .WithPgAdmin()
    .WithDataVolume("librerallymaps-pgpointcloud-data")
    .WithLifetime(ContainerLifetime.Persistent);

#pragma warning disable ASPIREPOSTGRES001
var mapsDb = postgres.AddDatabase("mapsdb").WithPostgresMcp();
#pragma warning restore ASPIREPOSTGRES001

// ── Python ML service (FastAPI / Open3D / Open3D-ML) ──────────────
var mlServiceImageBuild = builder.AddExecutable(
    "mlservice-image-build",
    "docker",
    mlServiceRoot,
    "build",
    "-t",
    mlServiceImageTag,
    ".");

var mlService = builder.AddContainer("mlservice", mlServiceImageTag)
    .WaitForCompletion(mlServiceImageBuild)
    .WithBindMount(dataRoot, "/data", isReadOnly: false)
    .WithEnvironment("CONTAINER_DATA_ROOT", "/data")
    .WithEnvironment("HOST_DATA_ROOT_RELATIVE", "data")
    .WithEnvironment("LAS_DIR", "/data/raw")
    .WithEnvironment("OUTPUT_DIR", "/data/tiles")
    .WithEnvironment("PIPELINE_INPUT_DIR", "/data/pipeline-input")
    .WithEnvironment("OPEN3D_ML_MODEL_PATH", "/data/models/randlanet_semantickitti_202201071330utc.pth")
    .WithEnvironment("OPEN3D_ML_MODEL_URL", "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth")
    .WithEnvironment("OPEN3D_ML_MODEL_AUTO_DOWNLOAD", "1")
    .WithEnvironment("STRICT_OPEN3D_ONLY", "1")
    .WithHttpEndpoint(targetPort: 8000)
    .WithHttpHealthCheck("/health");

if (gpuEnabled)
{
    mlService.WithContainerRuntimeArgs("--gpus=all");
}

// ── Tile Server (3D Tiles API) ──────────────────────────────────
var tileServer = builder.AddProject<Projects.LibreRally_Maps_TileServer>("tileserver")
    .WithReference(mapsDb)
    .WithReference(mlService.GetEndpoint("http"))
    .WaitFor(mapsDb)
    .WaitFor(mlService);

// ── Processing Worker (point cloud pipeline) ────────────────────
var processing = builder.AddProject<Projects.LibreRally_Maps_Processing>("processing")
    .WithReference(mapsDb)
    .WithReference(mlService.GetEndpoint("http"))
    .WaitFor(mapsDb)
    .WaitFor(mlService);

// ── WebUI (React frontend) ─────────────────────────────────────
var webui = builder.AddProject<Projects.LibreRally_Maps_WebUI>("webui")
    .WithReference(tileServer)
    .WaitFor(tileServer);

builder.Build().Run();
