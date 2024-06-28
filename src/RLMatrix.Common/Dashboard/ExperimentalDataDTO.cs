public class ExperimentDataDTO
{
    public string ExperimentId { get; set; }
    public DateTime Timestamp { get; set; }
    public double? Loss { get; set; }
    public double? LearningRate { get; set; }
    public double? Reward { get; set; }
    public double? CumulativeReward { get; set; }
    public int? EpisodeLength { get; set; }
}
