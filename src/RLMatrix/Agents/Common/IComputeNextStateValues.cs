using RLMatrix.Common;

namespace RLMatrix;

public interface IComputeNextStateValues
{
    Tensor ComputeNextStateValues(Tensor nonFinalNextStates, TensorModule targetNet, TensorModule policyNet, DQNAgentOptions opts, int[] discreteActionDimensions, Device device);
}