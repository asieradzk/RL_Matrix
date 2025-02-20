using RLMatrix.Common;

namespace RLMatrix;

public abstract class EnvironmentSizeDTO
{
    protected EnvironmentSizeDTO(int[] discreteActionDimensions, StateDimensions stateDimensions)
    {
        DiscreteActionDimensions = discreteActionDimensions;
        StateDimensions = stateDimensions;
    }
    
    public int[] DiscreteActionDimensions { get; }
    
    public StateDimensions StateDimensions { get; }
}