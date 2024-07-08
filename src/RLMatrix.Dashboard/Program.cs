using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.Extensions.FileProviders;
using RLMatrix.Dashboard;
using RLMatrix.Dashboard.Services;
using RLMatrix.Dashboard.Hubs;
using System.Reactive.Subjects;
using RLMatrix.Common.Dashboard;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();
builder.Services.AddSignalR();
builder.Services.AddSingleton<IDashboardService, DashboardService>();
builder.Services.AddScoped<IExportService, ExportService>();
builder.Services.AddSingleton<Subject<ExperimentData>>();
builder.Services.AddSingleton<RandomDataPusher>();

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseRouting();

app.MapBlazorHub();
app.MapHub<ExperimentDataHub>("/experimentdatahub");
app.MapFallbackToPage("/_Host");

var dataPusher = app.Services.GetRequiredService<RandomDataPusher>();
//dataPusher.Start();

app.Run();