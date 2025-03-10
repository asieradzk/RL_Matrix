using RLMatrix.Common;

namespace RLMatrix;

/// <summary>
///     Defines the core of any reinforcement learning agent.
/// </summary>
/// <typeparam name="TState"></typeparam>
public interface IAgent<TState> : IHasMemory<TState>, IActionSelectable<TState>, IOptimizable<TState>, ISavable
    where TState : notnull
{
    /// <summary>
    ///     Gets the dimensions of the discrete action space.
    /// </summary>
    int[] DiscreteActionDimensions { get; }

    /// <summary>
    ///     Optimizes the agent's model.
    /// </summary>
    ValueTask OptimizeModelAsync();
}