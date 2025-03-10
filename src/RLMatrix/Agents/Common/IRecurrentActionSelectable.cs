namespace RLMatrix;

/// <summary>
///     Defines an interface for objects that can select actions in a recurrent manner.
/// </summary>
/// <typeparam name="TState">The type of state used to select actions.</typeparam>
public interface IRecurrentActionSelectable<TState>
    where TState : notnull
{
    /// <summary>
    ///     Selects actions based on the given states and memory states in a recurrent manner.
    /// </summary>
    /// <param name="states">An array of <see cref="ActionsState"/> containing the state and memory states.</param>
    /// <param name="isTraining">Indicates whether the agent is in training mode.</param>
    /// <returns>An array of tuples containing selected actions and updated memory states.</returns>
    ValueTask<ActionsState[]> SelectActionsRecurrentAsync(RLMemoryState<TState>[] states, bool isTraining);
}