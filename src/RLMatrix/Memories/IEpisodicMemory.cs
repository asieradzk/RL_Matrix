namespace RLMatrix;

public interface IEpisodicMemory<TState> : IMemory<TState>
    where TState : notnull
{
    void Push(List<MemoryTransition<TState>> episode);
}