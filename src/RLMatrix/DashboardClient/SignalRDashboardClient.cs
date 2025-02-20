using System.Collections.Concurrent;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using Microsoft.AspNetCore.SignalR.Client;

namespace RLMatrix;

/// <summary>
/// Implements a SignalR-based dashboard client for real-time experiment data reporting.
/// </summary>
public class SignalRDashboardClient : IDashboardClient
{
    private readonly HubConnection _hubConnection;
    private readonly BehaviorSubject<(float? Reward, float? CumulativeReward, int? EpisodeLength)> _episodeDataSubject = new((null, null, null));
    private readonly BehaviorSubject<float?> _actorLossSubject = new(null);
    private readonly BehaviorSubject<float?> _actorLearningRateSubject = new(null);
    private readonly BehaviorSubject<float?> _criticLossSubject = new(null);
    private readonly BehaviorSubject<float?> _criticLearningRateSubject = new(null);
    private readonly BehaviorSubject<float?> _klDivergenceSubject = new(null);
    private readonly BehaviorSubject<float?> _entropySubject = new(null);
    private readonly BehaviorSubject<float?> _epsilonSubject = new(null);
    private readonly BehaviorSubject<float?> _lossSubject = new(null);
    private readonly BehaviorSubject<float?> _learningRateSubject = new(null);
    private readonly Subject<ExperimentData> _dataSubject = new();

    public IObservable<(float? Reward, float? CumulativeReward, int? EpisodeLength)> EpisodeData => _episodeDataSubject.AsObservable();
    public IObservable<float?> ActorLoss => _actorLossSubject.AsObservable();
    public IObservable<float?> ActorLearningRate => _actorLearningRateSubject.AsObservable();
    public IObservable<float?> CriticLoss => _criticLossSubject.AsObservable();
    public IObservable<float?> CriticLearningRate => _criticLearningRateSubject.AsObservable();
    public IObservable<float?> KLDivergence => _klDivergenceSubject.AsObservable();
    public IObservable<float?> Entropy => _entropySubject.AsObservable();
    public IObservable<float?> Epsilon => _epsilonSubject.AsObservable();
    public IObservable<float?> Loss => _lossSubject.AsObservable();
    public IObservable<float?> LearningRate => _learningRateSubject.AsObservable();

    private bool _isConnected;
    private bool _hasEverConnected;
    private readonly IDisposable _dataSubscription;

    private readonly ConcurrentQueue<ExperimentData> _dataQueue = new();
    private readonly Timer _sendTimer;
    private readonly SemaphoreSlim _sendSemaphore = new(1, 1);

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
            .WithAutomaticReconnect([TimeSpan.FromSeconds(0), TimeSpan.FromMilliseconds(5), TimeSpan.FromMilliseconds(10), TimeSpan.FromMilliseconds(30)])
            .Build();

        SetupCallbacks();
        var experimentId = Guid.NewGuid();

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
            .WithLatestFrom(latestOptimizationData, (epData, optData) => new ExperimentData(experimentId, DateTimeOffset.UtcNow, 
                Reward: epData.Reward,
                CumulativeReward: epData.CumulativeReward,
                EpisodeLength: epData.EpisodeLength,
                ActorLoss: optData.ActorLoss,
                ActorLearningRate: optData.ActorLearningRate,
                CriticLoss: optData.CriticLoss,
                CriticLearningRate: optData.CriticLearningRate,
                KLDivergence: optData.KLDivergence,
                Entropy: optData.Entropy,
                Epsilon: optData.Epsilon,
                Loss: optData.Loss,
                LearningRate: optData.LearningRate))
            .Subscribe(data => _dataQueue.Enqueue(data));

        _hubConnection.Closed += async error =>
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

    private void SendQueuedData(object? _)
    {
        Task.Run(SendQueuedDataAsync);
    }

    private async Task SendQueuedDataAsync()
    {
        await _sendSemaphore.WaitAsync();
        var batch = new List<ExperimentData>();
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
            await AttemptReconnectAsync();
        }
        finally
        {
            _sendSemaphore.Release();
        }
    }

    private async Task AttemptReconnectAsync()
    {
        if (_hasEverConnected)
        {
            for (var i = 0; i < 10; i++)
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

    public Task AddDataPointAsync(ExperimentData data)
    {
        _dataSubject.OnNext(data);
        return Task.CompletedTask;
    }

    // TODO: The original Func<string, Task> version also was never assigned or used.
    public Task SaveModelAsync(string path) => Task.CompletedTask;
    public Task LoadModelAsync(string path) => Task.CompletedTask;
    public Task SaveBufferAsync(string path) => Task.CompletedTask;
    public void UpdateEpisodeData(float? reward, float? cumReward, int? epLength) => _episodeDataSubject.OnNext((reward, cumReward, epLength));
    public void UpdateActorLoss(float? loss) => _actorLossSubject.OnNext(loss);
    public void UpdateActorLearningRate(float? lr) => _actorLearningRateSubject.OnNext(lr);
    public void UpdateCriticLoss(float? loss) => _criticLossSubject.OnNext(loss);
    public void UpdateCriticLearningRate(float? lr) => _criticLearningRateSubject.OnNext(lr);
    public void UpdateKLDivergence(float? kl) => _klDivergenceSubject.OnNext(kl);
    public void UpdateEntropy(float? entropy) => _entropySubject.OnNext(entropy);
    public void UpdateEpsilon(float? epsilon) => _epsilonSubject.OnNext(epsilon);
    public void UpdateLoss(float? loss) => _lossSubject.OnNext(loss);
    public void UpdateLearningRate(float? lr) => _learningRateSubject.OnNext(lr);

    private void SetupCallbacks()
    {
        _hubConnection.On<string>("SaveModel", SaveModelAsync);
        _hubConnection.On<string>("LoadModel", LoadModelAsync);
        _hubConnection.On<string>("SaveBuffer", SaveBufferAsync);
    }

    public async ValueTask DisposeAsync()
    {
        _dataSubscription.Dispose();
        _sendTimer.Dispose();
        
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