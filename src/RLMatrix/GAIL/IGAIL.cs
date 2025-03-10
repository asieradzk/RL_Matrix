namespace RLMatrix;

public interface IGAIL<TState>
    where TState : notnull
{
    void OptimiseDiscriminator(IMemory<TState> replayBuffer);
    Tensor AugmentRewardBatch(Tensor stateBatch, Tensor actionBatch, Tensor rewardBatch);
}