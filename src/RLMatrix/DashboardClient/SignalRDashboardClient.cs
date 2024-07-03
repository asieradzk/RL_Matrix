using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using RLMatrix.Common;
using RLMatrix.Common.Dashboard;

namespace RLMatrix.Dashboard
{
    public class SignalRDashboardClient : IDashboardClient, IAsyncDisposable
    {
        private HubConnection _hubConnection;

        public SignalRDashboardClient(string hubUrl = "http://127.0.0.1:5000/dashboardhub")
        {
            _hubConnection = new HubConnectionBuilder()
            .WithUrl(hubUrl).Build();
            SetupCallbacks();
        }

        public async Task StartAsync() => await _hubConnection.StartAsync();

        public async Task AddDataPoint(ExperimentData data)
            => await _hubConnection.SendAsync("AddDataPoint", data);

        public Func<string, Task> SaveModel { get; set; }
        public Func<string, Task> LoadModel { get; set; }
        public Func<string, Task> SaveBuffer { get; set; }

        private void SetupCallbacks()
        {
            _hubConnection.On<string>("SaveModel", async (path) => await SaveModel?.Invoke(path));
            _hubConnection.On<string>("LoadModel", async (path) => await LoadModel?.Invoke(path));
            _hubConnection.On<string>("SaveBuffer", async (path) => await SaveBuffer?.Invoke(path));
        }

        public async ValueTask DisposeAsync()
        {
            if (_hubConnection != null)
            {
                await _hubConnection.DisposeAsync();
            }
        }
    }
}