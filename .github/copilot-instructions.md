# LibreRally.Maps Copilot Instructions

## Commands

| Task | Command | Notes |
| --- | --- | --- |
| Build the .NET solution | `dotnet build LibreRally.Maps.slnx` | Run from the repo root. |
| Start the distributed app | `dotnet run --project src\AppHost\LibreRally.Maps.AppHost.csproj` | Starts the Aspire AppHost, PostgreSQL, TileServer, Processing worker, and WebUI. |
| Run the current .NET test entrypoint | `dotnet test LibreRally.Maps.slnx` | The solution currently has no dedicated .NET test projects, so this mainly validates project buildability. |
| Start the Python ML service | `pip install -r src\ml-service\requirements.txt` then `python src\ml-service\main.py` | `src\ml-service` is not part of the .NET solution, but the tile-processing pipeline depends on it. |
| Run one manual ML smoke test | `python src\ml-service\test_api.py` | Ad hoc script that expects a running ML service and a local LAS dataset path. |

## High-level architecture

- `src\AppHost\AppHost.cs` is the top-level orchestrator. It provisions PostgreSQL from the custom `db\Dockerfile`, creates the `mapsdb` database, starts `TileServer`, `Processing`, and `WebUI`, and wires service discovery between them.
- `src\TileServer` is the backend hub. It is a minimal API backed by `MapsDbContext`, serves generated files from `data\tiles`, handles LAS uploads and OSM imports, exposes tile/catalog/correction endpoints, and kicks off the Python pipeline by spawning `src\ml-service\pipeline.py`.
- The schema source lives in `db\init\01-schema.sql`. It defines the `pointcloud`, `osm`, and `tiles` schemas plus the core tables that TileServer queries directly.
- `src\WebUI` is a Blazor Server app with interactive pages for import, tile catalog, 3D viewing, and corrections. `TileApiClient` talks to TileServer using the Aspire logical host name `http://tileserver`, and `Program.cs` proxies `/data/tiles/*` back to TileServer so browser-side JS can fetch GLB assets.
- `src\Processing` is currently a lightweight worker that checks `pgpointcloud` availability and polls for future work. The real classification/reconstruction path still lives in the Python service rather than this worker.
- `src\ml-service` is a separate FastAPI service that exposes `/classify`, `/reconstruct`, `/health`, and `/info`. `main.py` prefers Open3D-backed inference when available and falls back to geometric classification; `pipeline.py` reads LAS files from `data\raw` and writes quadtree-organized GLBs into `data\tiles`.

## Key conventions

- Every .NET service should keep using `AddServiceDefaults()` and `MapDefaultEndpoints()` from `src\ServiceDefaults\Extensions.cs` so health checks, service discovery, resilience, and OpenTelemetry stay consistent.
- Preserve Aspire service names in service-to-service code. Inside the app, use logical names like `mapsdb` and `tileserver` instead of hard-coded localhost URLs unless the code is explicitly handling a browser/external-tool fallback.
- Generated tiles follow the quadtree naming convention exactly: `data\tiles\{zoom}.{x}.{y}\{zoom}.{x}.{y}.glb`. WebUI loading, TileServer registration, and the Python pipeline all assume that layout.
- Many file paths are resolved by walking back from `AppContext.BaseDirectory` to repo-root `data\` or `src\ml-service\`. Be careful replacing those with process-relative paths; several flows rely on repo-root-relative resolution.
- Database access mixes EF Core and raw SQL. EF is configured with `UseSnakeCaseNamingConvention()` and explicit schema/table mappings (`pointcloud.tiles`, `tiles.catalog`, `tiles.corrections`), while OSM queries assume `osm2pgsql` has populated the `osm` schema.
- Import and pipeline status are stored in in-memory dictionaries inside `src\TileServer\Program.cs` (`importJobs` and `pipelineJobs`). Restarting TileServer clears that state.
- The repo already checks in Aspire MCP config in `.mcp.json`, `.vscode\mcp.json`, and `opencode.jsonc`. Prefer using the Aspire server when you need to inspect the running distributed app.
- The Python test and pipeline scripts are exploratory rather than portable. Several assume Windows-style paths and local LAS data outside the repo, so treat them as local smoke scripts, not hermetic CI tests.
