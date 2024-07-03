using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.Extensions.FileProviders;
using RLMatrix.Dashboard;
using RLMatrix.Dashboard.Services;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();
builder.Services.AddSingleton<IDashboardService, DashboardService>();
builder.Services.AddSingleton<RandomDataPusher>();
builder.Services.AddScoped<IExportService, ExportService>();

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseRouting();
app.MapBlazorHub();
app.MapFallbackToPage("/_Host");

//random data points for testing
var randomDataPusher = app.Services.GetRequiredService<RandomDataPusher>();
randomDataPusher.Start();



app.Run();