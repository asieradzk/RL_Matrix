using RLMatrix.Common;

namespace RLMatrix;

/// <summary>
///     Defines an interface for objects that can select actions.
/// </summary>
/// <typeparam name="TState">The type of state used to select actions.</typeparam>
public interface IActionSelectable<TState>
    where TState : notnull
{
    /// <summary>
    ///     Selects actions based on the given states.
    /// </summary>
    /// <param name="states">The states to base the action selection on.</param>
    /// <param name="isTraining">Indicates whether the agent is in training mode.</param>
    /// <returns>An array of selected actions.</returns>
    ValueTask<RLActions[]> SelectActionsAsync(TState[] states, bool isTraining);
}