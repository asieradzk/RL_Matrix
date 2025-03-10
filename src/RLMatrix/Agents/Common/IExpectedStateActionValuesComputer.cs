using RLMatrix.Common;

namespace RLMatrix;

public interface IExpectedStateActionValuesComputer<TState>
    where TState : notnull
{
    Tensor ComputeExpectedStateActionValues(Tensor nextStateValues, Tensor rewardBatch, Tensor nonFinalMask, DQNAgentOptions opts, IList<MemoryTransition<TState>> transitions, int[] discreteActions, Device device);
}