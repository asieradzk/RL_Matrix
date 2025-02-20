namespace RLMatrix;

public class CategoricalExtractStateActionValues : IExtractStateActionValues
{
    private readonly int _numAtoms;

    public CategoricalExtractStateActionValues(int numAtoms)
    {
        _numAtoms = numAtoms;
    }

    public Tensor ExtractStateActionValues(Tensor qValues, Tensor actions)
    {
        var expandedActionBatch = actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, _numAtoms);
        var selectedActionDistributions = qValues.gather(2, expandedActionBatch).squeeze(2);
        return selectedActionDistributions; // Shape: [batch_size, num_heads, num_atoms]
    }
}