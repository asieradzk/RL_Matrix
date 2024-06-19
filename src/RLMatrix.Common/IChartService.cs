namespace RLMatrix.Agents.Common
{
    public interface IRLChartService
    {
        public void CreateOrUpdateChart(List<double> episodeRewards);
    }
}