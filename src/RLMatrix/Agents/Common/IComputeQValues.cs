namespace RLMatrix;

public interface IQValuesComputer
{
    Tensor ComputeQValues(Tensor states, TensorModule policyNet);
}