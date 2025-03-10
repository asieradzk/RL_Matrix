namespace RLMatrix;

public interface IMemory<TState>
    where TState: notnull
{
    int Length { get; }
    int EpisodeCount { get;}
    ValueTask PushAsync(MemoryTransition<TState> transition);
    ValueTask PushAsync(IList<MemoryTransition<TState>> transitions);
    IList<MemoryTransition<TState>> SampleEntireMemory();
    IList<MemoryTransition<TState>> Sample(int batchSize); 
    void ClearMemory();
}