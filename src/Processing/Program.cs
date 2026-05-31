using Microsoft.Extensions.Hosting;
using LibreRally.Maps.Processing;

var builder = Host.CreateApplicationBuilder(args);

// Aspire service defaults
builder.AddServiceDefaults();

// PostgreSQL via Aspire
builder.AddNpgsqlDataSource("mapsdb");

builder.Services.AddHostedService<Worker>();

var host = builder.Build();
host.Run();
