using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLMatrix.Common.Dashboard
{
    public record ExperimentData
    {
        Guid ExperimentId { get; set; }
        DateTime Timestamp { get; set; }
        double? Loss { get; set; }
        double? LearningRate { get; set; }
        double? Reward { get; set; }
        double? CumulativeReward { get; set; }
        int? EpisodeLength { get; set; }
    }
}
