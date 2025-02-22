namespace RLMatrix;

public class DashboardProvider : IAsyncDisposable
{
	private readonly SemaphoreSlim _initSemaphore = new(1, 1);
	private readonly bool _enableConsoleLogging;
	private IDashboardClient? _client;
	private readonly string _hubUrl;
	private readonly int? _consoleLoggingUpdateInterval;

	public DashboardProvider(string hubUrl = "https://localhost:7126/experimentdatahub", int? consoleLoggingUpdateInterval = null)
	{
		_hubUrl = hubUrl;
		_consoleLoggingUpdateInterval = consoleLoggingUpdateInterval;
		_enableConsoleLogging = consoleLoggingUpdateInterval > 0;

		Instance = this;
	}

	public static DashboardProvider Instance { get; private set; } = new(consoleLoggingUpdateInterval: 1);

	public async ValueTask<IDashboardClient> GetDashboardAsync()
	{
		if (_client is not null)
			return _client;

		await _initSemaphore.WaitAsync();
		if (_client is not null)
			return _client;

		try
		{
			if (_enableConsoleLogging)
			{
				return _client = new ConsoleClient(_consoleLoggingUpdateInterval ?? 1);
			}
			
			var client = new SignalRDashboardClient(_hubUrl);
			await client.StartAsync();

			return _client = client;
		}
		finally
		{
			_initSemaphore.Release();
		}
	}

	public ValueTask DisposeAsync()
	{
		return _client?.DisposeAsync() ?? new();
	}
}