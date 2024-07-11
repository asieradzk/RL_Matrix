using System;

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

    }
}