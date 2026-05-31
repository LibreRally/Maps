using LibreRally.Maps.WebUI.Components;
using LibreRally.Maps.WebUI;
using Microsoft.Extensions.Hosting;

var builder = WebApplication.CreateBuilder(args);

// Aspire service defaults
builder.AddServiceDefaults();

builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

// TileServer API client
builder.Services.AddHttpClient<TileApiClient>(client =>
{
    client.BaseAddress = new Uri("http://tileserver");
    client.Timeout = TimeSpan.FromMinutes(2);
});

var app = builder.Build();

// Proxy /data/tiles/ requests to TileServer (for GLB loading in browser)
var tileServerUrl = builder.Configuration["services__tileserver__http__0"] ?? "http://localhost:5034";
app.Map("/data/tiles/{**path}", async (string path, HttpContext context) =>
{
    using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
    var targetUrl = $"{tileServerUrl}/data/tiles/{path}";
    var response = await http.GetAsync(targetUrl);
    context.Response.StatusCode = (int)response.StatusCode;
    context.Response.ContentType = response.Content.Headers.ContentType?.ToString() ?? "application/octet-stream";
    await response.Content.CopyToAsync(context.Response.Body);
});

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
}

app.UseStatusCodePagesWithReExecute("/not-found", createScopeForStatusCodePages: true);
app.UseAntiforgery();

app.MapStaticAssets();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.MapDefaultEndpoints();
app.Run();
