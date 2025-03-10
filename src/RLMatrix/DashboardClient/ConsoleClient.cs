using System.Collections.Concurrent;
using System.Reactive.Linq;
using System.Reactive.Subjects;

namespace RLMatrix;

/// <summary>
///		A simple console-logging RLMatrix dashboard client.
/// </summary>
public class ConsoleClient : IDashboardClient
{
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

	private readonly IDisposable _dataSubscription;

	private readonly ConcurrentQueue<ExperimentData> _dataQueue = new();
	private readonly Timer _sendTimer;
	
	private int processedEpisodes;
	private float cumulativeRewards;
	
	public ConsoleClient(int refreshInterval = 1)
	{
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

		_sendTimer = new Timer(SendQueuedData, null, TimeSpan.Zero, TimeSpan.FromSeconds(refreshInterval));
		Console.WriteLine("console mode");
	}

	public Task AddDataPointAsync(ExperimentData data)
	{
		_dataSubject.OnNext(data);
		return Task.CompletedTask;
	}

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

	public ValueTask DisposeAsync()
	{
		_dataSubscription.Dispose();
		_sendTimer.Dispose();
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

		return new();
	}
	
	private void SendQueuedData(object? state)
	{
		for(var i = 0; !_dataQueue.IsEmpty; i++)
		{
			if (_dataQueue.TryDequeue(out var data))
			{
				if (data.Reward != null)
					cumulativeRewards += data.Reward.Value;
				
				processedEpisodes++;

				if (i == 0)
				{
					Console.WriteLine("ep=" + processedEpisodes + " cRwrds=" + cumulativeRewards.ToString("N3") + data);
				}
			}
			else
				return;
		}
	}
}