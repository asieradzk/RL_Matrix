using RLMatrix.Server;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddRLMatrixServices(@"C:\temp");

var app = builder.Build();

app.UseRLMatrixEndpoints();

app.Run();

//or configure by hand:
/*
void ConfigureServices(IServiceCollection services)
{
    services.AddSignalR()
        .AddMessagePackProtocol(options =>
        {
            // Configure MessagePack options if needed
        })
        .AddHubOptions<RLMatrixHub>(options =>
        {
            options.EnableDetailedErrors = true;
            options.ClientTimeoutInterval = TimeSpan.FromMinutes(5);
            options.KeepAliveInterval = TimeSpan.FromMinutes(2);
            options.HandshakeTimeout = TimeSpan.FromMinutes(2);
            options.MaximumReceiveMessageSize = 10240 * 10240; // 1 MB
        });

    services.AddSingleton<IDiscreteRLMatrixService>(new RLMatrixService(@"C:\temp"));
}

void Configure(IApplicationBuilder app)
{
    app.UseRouting();

    app.UseEndpoints(endpoints =>
    {
        endpoints.MapHub<RLMatrixHub>("/rlmatrixhub");
    });
}
*/
