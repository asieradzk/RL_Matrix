using RLMatrix.Common;

namespace RLMatrix;

// TODO: better XML docs? Winging this one
/// <summary>
///     Represents a proxy for a reinforcement learning agent and its environment.
/// </summary>
/// <typeparam name="TState">The type of the state for the agent/environment.</typeparam>
public interface IProxy<TState> : ISavable
    where TState : notnull
{
    /// <summary>
    ///     Selects actions for a batch of environments based on their current states.
    /// </summary>
    /// <param name="states">A list of <see cref="EnvironmentState{TState}"/> containing the environment ID and the current state.</param>
    /// <param name="isTraining">Indicates whether the agent is in training mode.</param>
    /// <returns>A dictionary mapping the environment ID to the selected actions.</returns>
    ValueTask<Dictionary<Guid, RLActions>> SelectBatchActionsAsync(IEnumerable<EnvironmentState<TState>> states, bool isTraining);
    
    /// <summary>
    ///     Resets the internal states of the proxy for the specified environments.
    ///     Only makes difference for recurrent models.
    /// </summary>
    /// <param name="environments">A list of tuples containing the environment ID and a boolean indicating if the environment is done.</param>
    ValueTask ResetStatesAsync(List<(Guid EnvironmentId, bool IsDone)> environments);
    
    /// <summary>
    ///     Uploads a batch of transitions to the proxy for training.
    /// </summary>
    /// <param name="transitions">The transitions containing the state, actions, reward, and next state.</param>
    ValueTask UploadTransitionsAsync(IEnumerable<Transition<TState>> transitions);
    
    /// <summary>
    ///     Optimizes the underlying model based on the uploaded transitions.
    /// </summary>
    ValueTask OptimizeModelAsync();
}