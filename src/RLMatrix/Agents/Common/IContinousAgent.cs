using RLMatrix.Common;

namespace RLMatrix;

/// <summary>
///     Represents a continuous agent with both discrete and continuous action capabilities.
/// </summary>
/// <typeparam name="TState">The type of state the agent operates on.</typeparam>
public interface IContinuousAgent<TState> : IAgent<TState>
    where TState : notnull
{
    /// <summary>
    ///     Gets the dimensions for continuous actions.
    /// </summary>
    ContinuousActionDimensions[] ContinuousActionDimensions { get; }
}