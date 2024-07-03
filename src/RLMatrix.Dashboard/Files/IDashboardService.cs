using RLMatrix.Common.Dashboard;

namespace RLMatrix.Dashboard
{
    public interface IDashboardService
    {
        void AddDataPoint(ExperimentData data);
        IObservable<ExperimentData> GetExperimentDataStream(Guid experimentId);
        IObservable<double> GetMetricStream(Guid experimentId, Func<ExperimentData, double?> metricSelector);
        IEnumerable<(Guid Id, DateTime StartTime)> GetAllExperiments();
        Task<List<ExperimentData>> GetExperimentDataAsync(Guid experimentId);
        DateTime GetExperimentStartTime(Guid experimentId);
    }
}