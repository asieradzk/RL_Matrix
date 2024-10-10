using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.Extensions.FileProviders;
using RLMatrix.Dashboard;
using RLMatrix.Dashboard.Services;
using RLMatrix.Dashboard.Hubs;
using System.Reactive.Subjects;
using RLMatrix.Common.Dashboard;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;

var builder = WebApplication.CreateBuilder(args);
var environment = builder.Environment.EnvironmentName;

if (environment == "Development")
{
    builder.WebHost.UseUrls("https://localhost:7126", "http://localhost:5069");
    builder.Services.AddHttpClient("UnsafeHttpClient").ConfigurePrimaryHttpMessageHandler(() =>
    {
        return new HttpClientHandler
        {
            ServerCertificateCustomValidationCallback = (sender, cert, chain, sslPolicyErrors) => true
        };
    });
}
else
{
    builder.WebHost.UseUrls("https://localhost:7126", "http://localhost:5069");
}

builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();
builder.Services.AddSignalR(options =>
{
    options.ClientTimeoutInterval = TimeSpan.FromMinutes(5);
    options.HandshakeTimeout = TimeSpan.FromMinutes(5);
    options.KeepAliveInterval = TimeSpan.FromMinutes(3.5);
    options.MaximumReceiveMessageSize = 1024 * 1024 * 1024; // 1 GB
}).AddJsonProtocol(options => {
    options.PayloadSerializerOptions.NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals;
});

builder.Services.AddSingleton<IDashboardService, DashboardService>();
builder.Services.AddScoped<IExportService, ExportService>();
builder.Services.AddSingleton<Subject<ExperimentData>>();

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseHsts();
}
else
{
    // Disable HTTPS redirection in development
    app.UseHttpsRedirection();
}

app.UseStaticFiles();
app.UseRouting();
app.MapBlazorHub();
app.MapHub<ExperimentDataHub>("/experimentdatahub");
app.MapFallbackToPage("/_Host");

_ = Task.Run(() =>
{
    OpenBrowser("http://localhost:5069/");
    app.Run();
});

// Keep the application running
await Task.Delay(-1);

static void OpenBrowser(string url)
{
    if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
    {
        Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
    }
    else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
    {
        Process.Start("xdg-open", url);
    }
    else
    {
    }
}