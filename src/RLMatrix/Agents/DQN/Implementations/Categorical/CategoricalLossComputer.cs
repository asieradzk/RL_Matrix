namespace RLMatrix;

public class CategoricalLossComputer : ILossComputer
{
    public Tensor ComputeLoss(Tensor stateActionDistributions, Tensor targetDistributions)
    {
        var criterion = new KLDivLoss(false, reduction: Reduction.None);
        var loss = criterion.forward(stateActionDistributions.log(), targetDistributions).mean([0, -1]).sum();
        return loss;
    }
}