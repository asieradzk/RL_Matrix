using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using RLMatrix.Common;
using RLMatrix.Common.Dashboard;
using System.Threading;

namespace RLMatrix.Dashboard
{
    /// <summary>
    /// Implements a SignalR-based dashboard client for real-time experiment data reporting.
    /// </summary>
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
        private bool _hasEverConnected = false;
        private IDisposable _dataSubscription;

        private ConcurrentQueue<ExperimentData> _dataQueue = new ConcurrentQueue<ExperimentData>();
        private Timer _sendTimer;
        private SemaphoreSlim _sendSemaphore = new SemaphoreSlim(1, 1);

        public SignalRDashboardClient(string hubUrl = "http://localhost:5069/experimentdatahub")
        {
            _hubConnection = new HubConnectionBuilder()
        .WithUrl(hubUrl, options =>
        {
            options.HttpMessageHandlerFactory = (message) =>
            {
                if (message is HttpClientHandler clientHandler)
                {
                    clientHandler.ServerCertificateCustomValidationCallback = (sender, certificate, chain, sslPolicyErrors) => true;
                }
                return message;
            };
        })
        .WithAutomaticReconnect(new[] { TimeSpan.FromSeconds(0), TimeSpan.FromMilliseconds(5), TimeSpan.FromMilliseconds(10), TimeSpan.FromMilliseconds(30) })
        .Build();

            SetupCallbacks();
            _experimentId = Guid.NewGuid();

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

            _dataSubscription = EpisodeData
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
                .Subscribe(data => _dataQueue.Enqueue(data));

            _hubConnection.Closed += async (error) =>
            {
                _isConnected = false;
                Console.WriteLine($"Connection closed due to error: {error?.Message}");
                await Task.Delay(new Random().Next(0, 5) * 1000);
                await StartAsync();
            };

            _sendTimer = new Timer(SendQueuedData, null, TimeSpan.Zero, TimeSpan.FromSeconds(1));
        }

        public async Task StartAsync()
        {
            if (!_isConnected)
            {
                try
                {
                    if (_hubConnection.State == HubConnectionState.Disconnected)
                    {
                        await _hubConnection.StartAsync();
                        _isConnected = true;
                        _hasEverConnected = true;
                        Console.WriteLine("Connected to dashboard");
                    }
                    else
                    {
                        await _hubConnection.StopAsync();
                        await _hubConnection.StartAsync();
                        _isConnected = true;
                        Console.WriteLine("Reconnected to dashboard");
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine($"Error starting dashboard connection: {e.Message}");
                    _isConnected = false;
                }
            }
        }

        private async void SendQueuedData(object state)
        {
            if (_isConnected && !_dataQueue.IsEmpty)
            {
                await _sendSemaphore.WaitAsync();
                List<ExperimentData> batch = new List<ExperimentData>();
                try
                {
                    while (batch.Count < 100 && _dataQueue.TryDequeue(out var data))
                    {
                        batch.Add(data);
                    }

                    if (batch.Count > 0)
                    {
                        await _hubConnection.SendAsync("AddDataBatch", batch);
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine($"Error sending data batch: {e.Message}");
                    _isConnected = false;
                    // Re-queue the data points
                    foreach (var data in batch)
                    {
                        _dataQueue.Enqueue(data);
                    }
                    batch.Clear();
                    await AttemptReconnect();
                }
                finally
                {
                    _sendSemaphore.Release();
                }
            }
        }

        private async Task AttemptReconnect()
        {
            if (_hasEverConnected)
            {
                for (int i = 0; i < 10; i++)
                {
                    try
                    {
                        if (_hubConnection.State != HubConnectionState.Disconnected)
                        {
                            await _hubConnection.StopAsync();
                        }
                        await StartAsync();
                        if (_isConnected)
                        {
                            Console.WriteLine($"Reconnected to dashboard after {i + 1} attempts");
                            return;
                        }
                    }
                    catch
                    {
                        await Task.Delay(TimeSpan.FromMilliseconds(1));
                    }
                }
                Console.WriteLine("Failed to reconnect after 10 attempts");
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
        public void UpdateLearningRate(double? lr) => _learningRateSubject.OnNext(lr);

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
            _sendTimer?.Dispose();
            if (_isConnected)
            {
                await _hubConnection.DisposeAsync();
            }
            _sendSemaphore.Dispose();
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