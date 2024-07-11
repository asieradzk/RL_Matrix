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
        private readonly BehaviorSubject<(double? Reward, double? CumulativeReward, int? EpisodeLength)> _episodeDataSubject =
            new BehaviorSubject<(double? Reward, double? CumulativeReward, int? EpisodeLength)>((null, null, null));
        private readonly BehaviorSubject<double?> _actorLossSubject = new BehaviorSubject<double?>(null);
        private readonly BehaviorSubject<double?> _actorLearningRateSubject = new BehaviorSubject<double?>(null);
        private readonly BehaviorSubject<double?> _criticLossSubject = new BehaviorSubject<double?>(null);
        private readonly BehaviorSubject<double?> _criticLearningRateSubject = new BehaviorSubject<double?>(null);
        private readonly BehaviorSubject<double?> _klDivergenceSubject = new BehaviorSubject<double?>(null);
        private readonly BehaviorSubject<double?> _entropySubject = new BehaviorSubject<double?>(null);
        private readonly BehaviorSubject<double?> _epsilonSubject = new BehaviorSubject<double?>(null);
        private readonly BehaviorSubject<double?> _lossSubject = new BehaviorSubject<double?>(null);
        private readonly BehaviorSubject<double?> _learningRateSubject = new BehaviorSubject<double?>(null);
        private readonly Subject<ExperimentData> _dataSubject = new Subject<ExperimentData>();

        public IObservable<(double? Reward, double? CumulativeReward, int? EpisodeLength)> EpisodeData => _episodeDataSubject.AsObservable();
        public IObservable<double?> ActorLoss => _actorLossSubject.AsObservable();
        public IObservable<double?> ActorLearningRate => _actorLearningRateSubject.AsObservable();
        public IObservable<double?> CriticLoss => _criticLossSubject.AsObservable();
        public IObservable<double?> CriticLearningRate => _criticLearningRateSubject.AsObservable();
        public IObservable<double?> KLDivergence => _klDivergenceSubject.AsObservable();
        public IObservable<double?> Entropy => _entropySubject.AsObservable();
        public IObservable<double?> Epsilon => _epsilonSubject.AsObservable();
        public IObservable<double?> Loss => _lossSubject.AsObservable();
        public IObservable<double?> LearningRate => _learningRateSubject.AsObservable();

        private readonly Guid _experimentId;
        private bool _isConnected = false;
        private IDisposable _dataSubscription;

        public SignalRDashboardClient(string hubUrl = "https://localhost:7126/experimentdatahub")
        {
            _hubConnection = new HubConnectionBuilder().WithUrl(hubUrl).Build();
            SetupCallbacks();
            _experimentId = Guid.NewGuid();
            _isConnected = true;

            var latestOptimizationData = Observable.CombineLatest(
                ActorLoss, ActorLearningRate, CriticLoss, CriticLearningRate,
                KLDivergence, Entropy, Epsilon, Loss, LearningRate,
                (actorLoss, actorLR, criticLoss, criticLR, klDivergence, entropy,
                 epsilon, loss, lr) =>
                    new
                    {
                        ActorLoss = actorLoss,
                        ActorLearningRate = actorLR,
                        CriticLoss = criticLoss,
                        CriticLearningRate = criticLR,
                        KLDivergence = klDivergence,
                        Entropy = entropy,
                        Epsilon = epsilon,
                        Loss = loss,
                        LearningRate = lr
                    });

            EpisodeData
                .Where(data => data.Reward.HasValue || data.CumulativeReward.HasValue || data.EpisodeLength.HasValue)
                .WithLatestFrom(latestOptimizationData, (epData, optData) => new ExperimentData
                {
                    ExperimentId = _experimentId,
                    Timestamp = DateTime.UtcNow,
                    Reward = epData.Reward,
                    CumulativeReward = epData.CumulativeReward,
                    EpisodeLength = epData.EpisodeLength,
                    ActorLoss = optData.ActorLoss,
                    ActorLearningRate = optData.ActorLearningRate,
                    CriticLoss = optData.CriticLoss,
                    CriticLearningRate = optData.CriticLearningRate,
                    KLDivergence = optData.KLDivergence,
                    Entropy = optData.Entropy,
                    Epsilon = optData.Epsilon,
                    Loss = optData.Loss,
                    LearningRate = optData.LearningRate
                })
                .Buffer(TimeSpan.FromSeconds(1))
                .Where(batch => batch.Count > 0)
                .Subscribe(async batch => await SendDataBatch(batch));
        }


        public async Task StartAsync()
        {
            if (_isConnected)
            {
                try
                {
                    var cts = new System.Threading.CancellationTokenSource(1000);
                    await _hubConnection.StartAsync(cts.Token);
                }
                catch (Exception e)
                {

                    Console.WriteLine("Terminating dashboard connection due to error: " + e.Message);
                    _isConnected = false;
                }

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

        public void UpdateEpisodeData(double? reward, double? cumReward, int? epLength) =>
           _episodeDataSubject.OnNext((reward, cumReward, epLength));

        public void UpdateActorLoss(double? loss) => _actorLossSubject.OnNext(loss);
        public void UpdateActorLearningRate(double? lr) => _actorLearningRateSubject.OnNext(lr);
        public void UpdateCriticLoss(double? loss) => _criticLossSubject.OnNext(loss);
        public void UpdateCriticLearningRate(double? lr) => _criticLearningRateSubject.OnNext(lr);
        public void UpdateKLDivergence(double? kl) => _klDivergenceSubject.OnNext(kl);
        public void UpdateEntropy(double? entropy) => _entropySubject.OnNext(entropy);
        public void UpdateEpsilon(double? epsilon) => _epsilonSubject.OnNext(epsilon);
        public void UpdateLoss(double? loss) => _lossSubject.OnNext(loss);
        public void UpdateLearningRate(double? lr) => _learningRateSubject.OnNext(lr); public Func<string, Task> SaveModel { get; set; }
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
            _episodeDataSubject.Dispose();
            _actorLossSubject.Dispose();
            _actorLearningRateSubject.Dispose();
            _criticLossSubject.Dispose();
            _criticLearningRateSubject.Dispose();
            _klDivergenceSubject.Dispose();
            _entropySubject.Dispose();
            _epsilonSubject.Dispose();
            _lossSubject.Dispose();
            _learningRateSubject.Dispose();
            _dataSubject.Dispose();
        }
    }
}