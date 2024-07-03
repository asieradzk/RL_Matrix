using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using RLMatrix.Common.Dashboard;
using System.Reactive.Subjects;
using System.Reactive.Linq;

namespace RLMatrix.Dashboard
{
    public class DashboardService : IDashboardService
    {
        private readonly ConcurrentDictionary<Guid, List<ExperimentData>> _experimentData
            = new ConcurrentDictionary<Guid, List<ExperimentData>>();
        private readonly ConcurrentDictionary<Guid, Subject<ExperimentData>> _experimentStreams
            = new ConcurrentDictionary<Guid, Subject<ExperimentData>>();
        private readonly ConcurrentDictionary<Guid, DateTime> _experimentStartTimes
            = new ConcurrentDictionary<Guid, DateTime>();

        public void AddDataPoint(ExperimentData data)
        {
            if (!_experimentStartTimes.ContainsKey(data.ExperimentId))
            {
                _experimentStartTimes[data.ExperimentId] = data.Timestamp;
            }

            _experimentData.AddOrUpdate(
                data.ExperimentId,
                new List<ExperimentData> { data },
                (_, list) => { list.Add(data); return list; });

            if (_experimentStreams.TryGetValue(data.ExperimentId, out var subject))
            {
                subject.OnNext(data);
            }
            else
            {
                var newSubject = new Subject<ExperimentData>();
                _experimentStreams[data.ExperimentId] = newSubject;
                newSubject.OnNext(data);
            }
        }

        public IObservable<ExperimentData> GetExperimentDataStream(Guid experimentId)
        {
            return _experimentStreams.GetOrAdd(experimentId, _ => new Subject<ExperimentData>());
        }

        public IEnumerable<(Guid Id, DateTime StartTime)> GetAllExperiments()
        {
            return _experimentStartTimes.Select(kvp => (kvp.Key, kvp.Value));
        }

        public Task<List<ExperimentData>> GetExperimentDataAsync(Guid experimentId)
        {
            return Task.FromResult(_experimentData.TryGetValue(experimentId, out var data)
                ? new List<ExperimentData>(data)
                : new List<ExperimentData>());
        }

        public IObservable<double> GetMetricStream(Guid experimentId, Func<ExperimentData, double?> metricSelector)
        {
            return GetExperimentDataStream(experimentId)
                .Select(metricSelector)
                .Where(m => m.HasValue)
                .Select(m => m.Value);
        }

        public DateTime GetExperimentStartTime(Guid experimentId)
        {
            return _experimentStartTimes.TryGetValue(experimentId, out var startTime) ? startTime : DateTime.MinValue;
        }
    }
}