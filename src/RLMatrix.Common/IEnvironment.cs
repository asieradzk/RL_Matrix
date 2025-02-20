namespace RLMatrix.Common;

/// <summary>
///     Defines the structure of an environment in which the reinforcement learning algorithm operates.
/// </summary>
/// <typeparam name="TState">The type that describes the state of the environment.</typeparam>
public interface IEnvironment<TState>
    where TState : notnull
{
    /// <summary>
    ///     The dimensionality of this <see cref="IEnvironment{TState}"/>'s state.
    ///     It can either be a single integer (for 1D states) or a tuple of two integers (for 2D states).
    /// </summary>
    StateDimensions StateDimensions { get; }
    
    /// <summary>
    ///     Gets the current state of this <see cref="IEnvironment{TState}"/>.
    /// </summary>
    ValueTask<TState> GetCurrentStateAsync();
    
    /// <summary>
    ///     Resets this <see cref="IEnvironment{TState}"/> to its starting state.
    /// </summary>
    ValueTask ResetAsync();
    
    /// <summary>
    ///     Advances this <see cref="IDiscreteEnvironment{TState}"/> a single step based on the actions chosen by the reinforcement learning algorithm.
    /// </summary>
    /// <param name="actions">The actions taken, decided by the model and within the bounds set by its dimensions.</param>
    ValueTask<StepResult> StepAsync(RLActions actions);
}