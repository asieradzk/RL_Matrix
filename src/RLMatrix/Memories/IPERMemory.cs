namespace RLMatrix;

public interface IPERMemory<TState> : IMemory<TState>
    where TState : notnull
{
    void Push(MemoryTransition<TState> transition, float priority);
    void Push(IEnumerable<MemoryTransition<TState>> transitions, IEnumerable<float> priorities);
    void Update(int experienceId, float newPriority);
}