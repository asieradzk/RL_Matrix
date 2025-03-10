using RLMatrix.Common;

namespace RLMatrix;

public interface INextStateValuesComputer
{
    Tensor ComputeNextStateValues(Tensor nonFinalNextStates, TensorModule targetNet, TensorModule policyNet, DQNAgentOptions opts, int[] discreteActionDimensions, Device device);
}