namespace RLMatrix.Common;

/// <summary>
///     Represents the dimensions of a continuous action.
/// </summary>
/// <param name="Min">The minimum value for this action.</param>
/// <param name="Max">The maximum value for this action.</param>
public record ContinuousActionDimensions(float Min, float Max)
{
    public static implicit operator ContinuousActionDimensions((float Min, float Max) tuple) => new(tuple.Min, tuple.Max);
}