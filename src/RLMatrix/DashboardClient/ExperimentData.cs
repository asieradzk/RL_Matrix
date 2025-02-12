using System;
using System.Xml.Linq;

namespace RLMatrix.Common.Dashboard
{
	public record ExperimentData
	{
		public Guid ExperimentId { get; set; }
		public DateTime Timestamp { get; set; }

		// General metrics
		public double? Reward { get; set; }
		public double? CumulativeReward { get; set; }
		public int? EpisodeLength { get; set; }

		// Actor-Critic specific metrics
		public double? ActorLoss { get; set; }
		public double? ActorLearningRate { get; set; }
		public double? CriticLoss { get; set; }
		public double? CriticLearningRate { get; set; }

		// Policy metrics
		public double? KLDivergence { get; set; }
		public double? Entropy { get; set; }

		// DQN/C51 specific metrics
		public double? Epsilon { get; set; }
		public double? Loss { get; set; }
		public double? LearningRate { get; set; }

		public static bool PrintReward = true;
		public static bool PrintCumulativeReward = true;
		public static bool PrintActorLoss = false;
		public static bool PrintCriticLoss = false;
		public static bool PrintEntropy = true;
		public static bool PrintLoss = true;
		public static bool PrintLearningRate = true;
		public override string ToString()
		{
			string s = "";
			if (PrintEntropy && Entropy != null)
				s += " Ent=" + Entropy.Value.ToString("N4");
			if (PrintReward && Reward != null)
				s += " Reward=" + Reward.Value.ToString("N3");
			if (PrintActorLoss && ActorLoss != null)
				s += " ActLoss=" + ActorLoss.Value.ToString("N3");
			if (PrintCriticLoss && CriticLoss != null)
				s += " CrLoss=" + CriticLoss.Value.ToString("N3");
			if (PrintLoss && Loss != null)
				s += " Loss=" + Loss.Value.ToString("N3");
			if (PrintLearningRate && LearningRate != null)
				s += " LR=" + LearningRate.Value.ToString("N3");
			return s;
		}
	}
}