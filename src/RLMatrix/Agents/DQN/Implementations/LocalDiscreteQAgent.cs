using RLMatrix.Common;

namespace RLMatrix;

/// <summary>
/// Represents a local discrete Q-learning agent.
/// </summary>
/// <typeparam name="TState">The type of the state.</typeparam>
public class LocalDiscreteQAgent<TState> : IDiscreteProxy<TState>
    where TState : notnull
{
    private readonly ComposableQDiscreteAgent<TState> _agent;

    /// <summary>
    ///     Initializes a new instance of the <see cref="LocalDiscreteQAgent{T}"/> class.
    /// </summary>
    /// <param name="opts">The DQN agent options.</param>
    /// <param name="actionSizes">The sizes of the action space.</param>
    /// <param name="stateDimensions">The sizes of the state space.</param>
    /// <param name="agentComposer">The optional agent composer.</param>
    public LocalDiscreteQAgent(DQNAgentOptions opts, int[] actionSizes, StateDimensions stateDimensions, IDiscreteQAgentFactory<TState>? agentComposer = null)
    {
        _agent = agentComposer?.ComposeAgent(opts) ?? DiscreteQAgentFactory<TState>.ComposeQAgent(opts, actionSizes, stateDimensions);
    }
    
    /// <summary>
    ///     Saves the agent's state.
    /// </summary>
    /// <param name="path">The path to save the agent's state to.</param>
    public ValueTask SaveAsync(string path)
    {
        return _agent.SaveAsync(path);
    }

    /// <summary>
    ///     Loads the agent's state.
    /// </summary>
    /// <param name="path">The path to load the agent's state from.</param>
    public ValueTask LoadAsync(string path)
    {
        return _agent.LoadAsync(path);
    }

    /// <summary>
    ///     Optimizes the agent's model.
    /// </summary>
    public ValueTask OptimizeModelAsync()
    {
        return _agent.OptimizeModelAsync();
    }

    /// <summary>
    ///     Resets the states of the provided environments.
    /// </summary>
    /// <param name="environments">The list of environment IDs and their done states.</param>
    // TODO: this method did nothing in the original impl -> (Value)Task.CompletedTask?
    public ValueTask ResetStatesAsync(List<(Guid EnvironmentId, bool IsDone)> environments)
    {
        return new ValueTask();
    }

    /// <summary>
    ///     Selects actions for a batch of states.
    /// </summary>
    /// <param name="states">The list of state information.</param>
    /// <param name="isTraining">Indicates whether the agent is in training mode.</param>
    /// <returns>A dictionary of actions for each environment.</returns>
    public async ValueTask<Dictionary<Guid, RLActions>> SelectBatchActionsAsync(IEnumerable<EnvironmentState<TState>> states, bool isTraining)
    {
        var stateList = states as IList<EnvironmentState<TState>> ?? states.ToList();
        var selectedActions = await _agent.SelectActionsAsync(stateList.Select(x => x.State).ToArray(), isTraining);
        var actionsMap = new Dictionary<Guid, RLActions>();

        for (var i = 0; i < stateList.Count; i++)
        {
            var environmentId = stateList[i].EnvironmentId;
            var actions = selectedActions[i];
            actionsMap[environmentId] = actions;
        }

        return actionsMap;
    }

    /// <summary>
    ///     Uploads transitions to the agent.
    /// </summary>
    /// <param name="transitions">The transitions to upload.</param>
    public ValueTask UploadTransitionsAsync(IEnumerable<Transition<TState>> transitions)
        => _agent.AddTransitionsAsync(transitions);
}