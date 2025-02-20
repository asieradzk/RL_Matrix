using RLMatrix.Common;

namespace RLMatrix;

public class DiscreteEnvironmentSizeDTO : EnvironmentSizeDTO
{
    public DiscreteEnvironmentSizeDTO(int[] discreteActionDimensions, StateDimensions stateDimensions) 
        : base(discreteActionDimensions, stateDimensions)
    {
        if (discreteActionDimensions is not { Length: > 0 })
            throw new ArgumentException("Action size array cannot be null or empty.");

        for (var i = 1; i < discreteActionDimensions.Length; i++)
        {
            if (discreteActionDimensions[i] != discreteActionDimensions[i - 1])
                throw new ArgumentException("All discrete heads must have identical size - action size array must be uniform.");
        }
    }
}