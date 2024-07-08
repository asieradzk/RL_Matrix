using System;
using System.Threading.Tasks;

namespace RLMatrix.Dashboard
{
    public static class DashboardProvider
    {
        private static SignalRDashboardClient _instance;
        private static readonly SemaphoreSlim _semaphore = new SemaphoreSlim(1, 1);

        public static SignalRDashboardClient Instance
        {
            get
            {
                EnsureInitialized();
                return _instance;
            }
        }

        private static void EnsureInitialized()
        {
            if (_instance == null)
            {
                _semaphore.Wait();
                try
                {
                    if (_instance == null)
                    {
                        Initialize();
                    }
                }
                finally
                {
                    _semaphore.Release();
                }
            }
        }

        private static async void Initialize(string hubUrl = "https://localhost:7126/experimentdatahub")
        {
            if (_instance != null) return;

            _instance = new SignalRDashboardClient(hubUrl);
            await _instance.StartAsync();
        }

        public static async ValueTask DisposeAsync()
        {
            if (_instance != null)
            {
                await _instance.DisposeAsync();
                _instance = null;
            }
        }
    }
}
