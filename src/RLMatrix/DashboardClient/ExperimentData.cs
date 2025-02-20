using System.Text;

namespace RLMatrix;

public sealed record ExperimentData(
	Guid ExperimentId,
	DateTimeOffset Timestamp,
	// General metrics
	float? Reward = null,
	float? CumulativeReward = null,
	int? EpisodeLength = null,
	// Actor-Critic specific metrics
	float? ActorLoss = null,
	float? ActorLearningRate = null,
	float? CriticLoss = null,
	float? CriticLearningRate = null,
	// Policy metrics
	float? KLDivergence = null,
	float? Entropy = null,
	// DQN/C51 specific metrics
	float? Epsilon = null,
	float? Loss = null,
	float? LearningRate = null)
{
	public static bool PrintReward { get; set; } = true;
	public static bool PrintCumulativeReward { get; set; } = true;
	public static bool PrintActorLoss { get; set; } = false;
	public static bool PrintCriticLoss { get; set; } = false;
	public static bool PrintEntropy { get; set; } = true;
	public static bool PrintLoss { get; set; } = true;
	public static bool PrintLearningRate { get; set; } = true;

	private bool PrintMembers(StringBuilder builder)
	{
		var len = builder.Length;
		if (PrintEntropy && Entropy.HasValue)
			builder.Append($"Ent={Entropy:N4}, ");

		if (PrintReward && Reward.HasValue)
			builder.Append($"Reward={Reward:N3}, ");

		if (PrintActorLoss && ActorLoss.HasValue)
			builder.Append($"ActLoss={ActorLoss:N3}, ");

		if (PrintCriticLoss && CriticLoss.HasValue)
			builder.Append($"CrLoss={CriticLoss:N3}, ");

		if (PrintLoss && Loss.HasValue)
			builder.Append($"Loss={Loss:N3}, ");

		if (PrintLearningRate && LearningRate.HasValue)
			builder.Append($"LR={LearningRate:N3}");

#if NET8_0_OR_GREATER
		while (builder.Length > len && builder[^1] is ' ' or ',')
#else
		while (builder.Length > len && builder[builder.Length - 1] is ' ' or ',')
#endif
			builder.Remove(builder.Length - 1, 1);

		return true;
	}
}