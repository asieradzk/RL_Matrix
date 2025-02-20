using RLMatrix.Common;

namespace RLMatrix;

public class ContinuousEnvironmentSizeDTO : EnvironmentSizeDTO
{
    public ContinuousEnvironmentSizeDTO(int[] discreteActionDimensions, StateDimensions stateDimensions, ContinuousActionDimensions[] continuousActionDimensions) 
        : base(discreteActionDimensions, stateDimensions)
    {
        ContinuousActionDimensions = continuousActionDimensions;
    }
    
    public ContinuousActionDimensions[] ContinuousActionDimensions { get; }
}