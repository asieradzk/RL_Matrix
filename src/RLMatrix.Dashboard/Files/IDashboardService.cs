using RLMatrix.Common.Dashboard;

public interface IDashboardService
{
    Task AddDataPoint(ExperimentData data);
    IObservable<IList<ExperimentData>> GetExperimentDataStream(Guid experimentId);
    IEnumerable<(Guid Id, DateTime StartTime)> GetAllExperiments();
    Task<List<ExperimentData>> GetExperimentDataAsync(Guid experimentId);
    DateTime GetExperimentStartTime(Guid experimentId);
    IObservable<IList<ExperimentData>> DataStream { get; }
}