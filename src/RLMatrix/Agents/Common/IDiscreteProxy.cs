using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace RLMatrix.Agents.Common
{
    public interface IDiscreteProxy<T>
    {
#if NET8_0_OR_GREATER
        ValueTask<Dictionary<Guid, int[]>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos, bool isTraining);
        ValueTask ResetStates(List<(Guid environmentId, bool dones)> environmentIds);
        ValueTask UploadTransitionsAsync(IEnumerable<TransitionPortable<T>> transitions);
        ValueTask OptimizeModelAsync();
        ValueTask SaveAsync(string path);
        ValueTask LoadAsync(string path);
#else
        Task<Dictionary<Guid, int[]>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos, bool isTraining);
        Task ResetStates(List<(Guid environmentId, bool dones)> environmentIds);
        Task UploadTransitionsAsync(IEnumerable<TransitionPortable<T>> transitions);
        Task OptimizeModelAsync();
        Task SaveAsync(string path);
        Task LoadAsync(string path);
#endif
    }
}