using RLMatrix.Common;

namespace RLMatrix;

public interface ILookAheadStepsComputer<TState>
    where TState : notnull
{
    Tensor ComputeLookAheadSteps(IList<MemoryTransition<TState>> transitions, DQNAgentOptions opts, Device device);
}