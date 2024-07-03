using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLMatrix.Common.Dashboard
{
    public record ExperimentData
    {
        public Guid ExperimentId { get; set; }
        public DateTime Timestamp { get; set; }
        public double? Loss { get; set; }
        public double? LearningRate { get; set; }
        public double? Reward { get; set; }
        public double? CumulativeReward { get; set; }
        public int? EpisodeLength { get; set; }
    }
}
