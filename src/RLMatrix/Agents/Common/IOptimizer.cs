namespace RLMatrix;

/// <summary>
///     Defines an interface for objects that can optimize based on a memory.
/// </summary>
/// <typeparam name="TState">The type of state the optimizer operates on.</typeparam>
public interface IOptimizer<TState>
    where TState : notnull
{
    /// <summary>
    ///     Optimizes based on the given replay buffer.
    /// </summary>
    /// <param name="replayBuffer">The replay buffer to optimize from.</param>
    ValueTask OptimizeAsync(IMemory<TState> replayBuffer);

    /// <summary>
    ///     Updates the optimizers using the given scheduler.
    /// </summary>
    /// <param name="scheduler">The learning rate scheduler to use for updating.</param>
    ValueTask UpdateOptimizersAsync(LRScheduler scheduler);
}