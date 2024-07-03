using RLMatrix.Common.Dashboard;
using RLMatrix.Dashboard;

public class RandomDataPusher
{
    private readonly IDashboardService _dashboardService;
    private readonly Random _random = new Random();
    private readonly CancellationTokenSource _cts = new CancellationTokenSource();

    public RandomDataPusher(IDashboardService dashboardService)
    {
        _dashboardService = dashboardService;
    }

    public void Start()
    {
        for (int i = 0; i < 5; i++)
        {
            var experimentId = Guid.NewGuid();
            Task.Run(() => PushDataContinuously(experimentId), _cts.Token);
        }
    }

    public void Stop()
    {
        _cts.Cancel();
    }

    private async Task PushDataContinuously(Guid experimentId)
    {
        int i = 0;
        while (!_cts.Token.IsCancellationRequested)
        {
            var data = new ExperimentData
            {
                ExperimentId = experimentId,
                Timestamp = DateTime.Now,
                Loss = Math.Max(0, 10 - i * 0.1 + _random.NextDouble()),
                Reward = i * 0.5 + _random.NextDouble(),
                LearningRate = Math.Max(0.001, 0.1 - (i * 0.001)),
                CumulativeReward = (i * 0.5 + _random.NextDouble()) * (i + 1),
                EpisodeLength = 100 + i
            };

            _dashboardService.AddDataPoint(data);

            i++;
            await Task.Delay(1000, _cts.Token);
        }
    }
}