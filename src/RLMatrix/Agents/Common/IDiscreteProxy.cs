namespace RLMatrix.Agents.Common
{
    public interface IDiscreteProxy<T>
    {
        public ValueTask<Dictionary<Guid, int[]>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos);
        public ValueTask UploadTransitionsAsync(IEnumerable<TransitionPortable<T>> transitions);
        public ValueTask OptimizeModelAsync();
    }
}
