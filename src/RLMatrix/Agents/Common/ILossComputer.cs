namespace RLMatrix;

public interface ILossComputer
{
    Tensor ComputeLoss(Tensor expectedStateActionValues, Tensor stateActionValues);
}

