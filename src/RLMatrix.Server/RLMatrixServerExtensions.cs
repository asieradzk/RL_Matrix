using Microsoft.Extensions.DependencyInjection;
using Microsoft.AspNetCore.Builder;
using RLMatrix.Server;
using Microsoft.AspNetCore.SignalR;

public static class RLMatrixServerExtensions
{
    public static void AddRLMatrixServices(this IServiceCollection services, string savePath)
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
                options.MaximumReceiveMessageSize = 10240 * 10240; // 10 MB
            });

        services.AddSingleton<IRLMatrixService>(new RLMatrixService(savePath));
    }

    public static void UseRLMatrixEndpoints(this IApplicationBuilder app)
    {
        app.UseRouting();

        app.UseEndpoints(endpoints =>
        {
            endpoints.MapHub<RLMatrixHub>("/rlmatrixhub");
        });
    }
}
