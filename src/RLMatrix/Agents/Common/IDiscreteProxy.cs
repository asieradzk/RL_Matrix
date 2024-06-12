namespace RLMatrix.Agents.Common
{
    public interface IDiscreteProxy<T>
    {
        public ValueTask<Dictionary<Guid, int[]>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos);
        //TODO: possibly ISP violation, only recurrent PPO uses ResetStates
        public ValueTask ResetStates(List<(Guid environmentId, bool dones)> environmentIds);
        public ValueTask UploadTransitionsAsync(IEnumerable<TransitionPortable<T>> transitions);
        public ValueTask OptimizeModelAsync();

        public ValueTask SaveAsync(string path);
        public ValueTask LoadAsync(string path);
    }
}
