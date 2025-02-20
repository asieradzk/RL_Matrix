namespace RLMatrix;

public interface IComputeQValues
{
    Tensor ComputeQValues(Tensor states, TensorModule policyNet);
}