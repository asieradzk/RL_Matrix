using System;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using RLMatrix.Common;
using RLMatrix.Common.Dashboard;

namespace RLMatrix.Dashboard
{
    public class SignalRDashboardClient : IDashboardClient, IAsyncDisposable
    {
        private HubConnection _hubConnection;
        private readonly BehaviorSubject<double?> _lossSubject = new BehaviorSubject<double?>(null);
        private readonly BehaviorSubject<double?> _learningRateSubject = new BehaviorSubject<double?>(null);
        private readonly BehaviorSubject<(double? Reward, double? CumulativeReward, int? EpisodeLength)> _episodeDataSubject =
            new BehaviorSubject<(double? Reward, double? CumulativeReward, int? EpisodeLength)>((null, null, null));
        private readonly Subject<ExperimentData> _dataSubject = new Subject<ExperimentData>();

        public IObservable<double?> Loss => _lossSubject.AsObservable();
        public IObservable<double?> LearningRate => _learningRateSubject.AsObservable();
        public IObservable<(double? Reward, double? CumulativeReward, int? EpisodeLength)> EpisodeData => _episodeDataSubject.AsObservable();

        private readonly Guid _experimentId;
        private bool _isConnected = false;
        private IDisposable _dataSubscription;

        public SignalRDashboardClient(string hubUrl = "https://localhost:7126/experimentdatahub")
        {
            _hubConnection = new HubConnectionBuilder().WithUrl(hubUrl).Build();
            SetupCallbacks();
            _experimentId = Guid.NewGuid();
            _isConnected = true;

            var latestLossAndLR = Observable.CombineLatest(
                Loss,
                LearningRate,
                (loss, lr) => new { Loss = loss, LearningRate = lr });

            EpisodeData
                .Where(data => data.Reward.HasValue && data.CumulativeReward.HasValue && data.EpisodeLength.HasValue)
                .WithLatestFrom(latestLossAndLR, (epData, lossLR) => new ExperimentData
                {
                    ExperimentId = _experimentId,
                    Timestamp = DateTime.UtcNow,
                    Loss = lossLR.Loss,
                    LearningRate = lossLR.LearningRate,
                    Reward = epData.Reward.Value,
                    CumulativeReward = epData.CumulativeReward.Value,
                    EpisodeLength = epData.EpisodeLength.Value
                })
                .Buffer(TimeSpan.FromSeconds(1))
                .Where(batch => batch.Count > 0)
                .Subscribe(async batch => await SendDataBatch(batch));
        }

        public async Task StartAsync()
        {
            if (_isConnected)
            {
                var cts = new System.Threading.CancellationTokenSource(1000);
                await _hubConnection.StartAsync(cts.Token);
            }
        }

        private async Task SendDataBatch(IList<ExperimentData> batch)
        {
            if (_isConnected)
            {
                try
                {
                    await _hubConnection.SendAsync("AddDataBatch", batch);
                  
                }
                catch (Exception e)
                {
                    Console.WriteLine("Terminating dashboard connection due to error: " + e.Message);
                   _isConnected = false;
                }
                
            }
        }

        public Task AddDataPoint(ExperimentData data)
        {
            _dataSubject.OnNext(data);
            return Task.CompletedTask;
        }

        public void UpdateLoss(double? loss) => _lossSubject.OnNext(loss);
        public void UpdateLearningRate(double? lr) => _learningRateSubject.OnNext(lr);
        public void UpdateEpisodeData(double? reward, double? cumReward, int? epLength) =>
            _episodeDataSubject.OnNext((reward, cumReward, epLength));

        public Func<string, Task> SaveModel { get; set; }
        public Func<string, Task> LoadModel { get; set; }
        public Func<string, Task> SaveBuffer { get; set; }

        private void SetupCallbacks()
        {
            _hubConnection.On<string>("SaveModel", path => _ = SaveModel?.Invoke(path));
            _hubConnection.On<string>("LoadModel", path => _ = LoadModel?.Invoke(path));
            _hubConnection.On<string>("SaveBuffer", path => _ = SaveBuffer?.Invoke(path));
        }

        public async ValueTask DisposeAsync()
        {
            _dataSubscription?.Dispose();
            if (_isConnected)
            {
                await _hubConnection.DisposeAsync();
            }
            _lossSubject.Dispose();
            _learningRateSubject.Dispose();
            _episodeDataSubject.Dispose();
            _dataSubject.Dispose();
        }
    }
}
