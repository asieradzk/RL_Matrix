using RLMatrix.Common;
using System;
using System.Threading.Tasks;

namespace RLMatrix.Dashboard
{
	public static class DashboardProvider
	{
		static bool consoleLogging = false;
		static SignalRDashboardClient _instance;
		static ConsoleClient _consoleInstance;
		static readonly SemaphoreSlim _semaphore = new SemaphoreSlim(1, 1);
		public static bool EnableConsoleLogging(int updateInterval)
		{
			if(_instance== null && _consoleInstance == null)
			{
				_consoleInstance ??= new ConsoleClient(updateInterval);
				consoleLogging = true;
				return true;
			}
			return false;
		}
		public static IDashboardClient Instance
		{
			get
			{
				if(consoleLogging)
				{
					_consoleInstance ??= new ConsoleClient();
					return _consoleInstance;
				}
				else
				{
					EnsureInitialized();
					return _instance;
				}
				
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
