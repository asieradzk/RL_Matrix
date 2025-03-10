namespace RLMatrix;

/// <summary>
///     Defines an interface for objects that have an optimizer.
/// </summary>
/// <typeparam name="TState">The type of state the optimizer operates on.</typeparam>
public interface IOptimizable<TState>
    where TState : notnull
{
    /// <summary>
    ///     Gets the optimizer of the object.
    /// </summary>
    IOptimizer<TState> Optimizer { get; }
}