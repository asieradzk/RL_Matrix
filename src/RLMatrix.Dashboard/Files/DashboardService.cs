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
        private readonly ConcurrentDictionary<Guid, Subject<ExperimentData>> _experimentSubjects
            = new ConcurrentDictionary<Guid, Subject<ExperimentData>>();
        private readonly ConcurrentDictionary<Guid, List<ExperimentData>> _experimentData
            = new ConcurrentDictionary<Guid, List<ExperimentData>>();
        private readonly ConcurrentDictionary<Guid, DateTime> _experimentStartTimes
            = new ConcurrentDictionary<Guid, DateTime>();
        private readonly ConcurrentDictionary<Guid, double> _cumulativeRewards
            = new ConcurrentDictionary<Guid, double>();
        private readonly Subject<ExperimentData> _globalDataSubject = new Subject<ExperimentData>();

        public IObservable<IList<ExperimentData>> DataStream =>
            _globalDataSubject.Buffer(TimeSpan.FromSeconds(1)).Where(list => list.Count > 0);

        public Task AddDataPoint(ExperimentData data)
        {
            if (!_experimentStartTimes.ContainsKey(data.ExperimentId))
            {
                _experimentStartTimes[data.ExperimentId] = data.Timestamp;
            }

            // Calculate and store cumulative reward
            _cumulativeRewards.AddOrUpdate(
                data.ExperimentId,
                data.Reward ?? 0,
                (_, currentReward) => currentReward + (data.Reward ?? 0)
            );

            // Set the cumulative reward on the data point
            data.CumulativeReward = _cumulativeRewards[data.ExperimentId];

            _experimentData.AddOrUpdate(
                data.ExperimentId,
                new List<ExperimentData> { data },
                (_, list) => { list.Add(data); return list; }
            );

            var subject = _experimentSubjects.GetOrAdd(data.ExperimentId, _ => new Subject<ExperimentData>());
            subject.OnNext(data);
            _globalDataSubject.OnNext(data);

            return Task.CompletedTask;
        }

        public IObservable<IList<ExperimentData>> GetExperimentDataStream(Guid experimentId)
        {
            return _experimentSubjects
                .GetOrAdd(experimentId, _ => new Subject<ExperimentData>())
                .Buffer(TimeSpan.FromSeconds(1))
                .Where(list => list.Count > 0);
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

        public DateTime GetExperimentStartTime(Guid experimentId)
        {
            return _experimentStartTimes.TryGetValue(experimentId, out var startTime) ? startTime : DateTime.MinValue;
        }
    }
}