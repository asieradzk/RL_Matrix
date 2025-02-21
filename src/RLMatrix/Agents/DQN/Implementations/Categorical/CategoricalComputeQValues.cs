namespace RLMatrix;

public class CategoricalComputeQValues : IQValuesComputer
{
    private readonly int[] _actionSizes;
    private readonly int _numAtoms;
    
    public CategoricalComputeQValues(int[] actionSizes, int numAtoms)
    {
        _actionSizes = actionSizes;
        _numAtoms = numAtoms;
    }

    public Tensor ComputeQValues(Tensor stateBatch, torch.nn.Module<Tensor, Tensor> policyNet)
    {
        var result = policyNet.forward(stateBatch);
        return result.view(stateBatch.shape[0], _actionSizes.Length, _actionSizes[0], _numAtoms); // Shape: [batch_size, num_heads, num_actions, num_atoms]
    }
}