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
	/// simple console logging
	/// </summary>
	public class ConsoleClient : IDashboardClient, IAsyncDisposable
	{
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
		private IDisposable _dataSubscription;

		private ConcurrentQueue<ExperimentData> _dataQueue = new ConcurrentQueue<ExperimentData>();
		private Timer _sendTimer;
		private int processedEpisodes;
		double cumulativeRewards = 0;
		int consoleUpdateIntervalSeconds = 1;
		public ConsoleClient(int refreshInterval = 1)
		{
			consoleUpdateIntervalSeconds = refreshInterval;
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

			_sendTimer = new Timer(SendQueuedData, null, TimeSpan.Zero, TimeSpan.FromSeconds(consoleUpdateIntervalSeconds));
			Console.WriteLine("console mode");
		}

		void SendQueuedData(object? state)
		{
			for(int i=0; !_dataQueue.IsEmpty; i++)
			{
				if (_dataQueue.TryDequeue(out var data))
				{
					if (data.Reward != null)
						cumulativeRewards += data.Reward.Value;
					processedEpisodes++;

					if (i == 0)
					{
						Console.WriteLine("ep=" + processedEpisodes + " cRwrds=" + cumulativeRewards.ToString("N3") + data.ToString());
					}
				}
				else
					return;
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

		public async ValueTask DisposeAsync()
		{
			_dataSubscription?.Dispose();
			_sendTimer?.Dispose();
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