using RLMatrix.Common;

namespace RLMatrix;

public class LocalContinuousPPOAgent<TState> : IContinuousProxy<TState>
    where TState : notnull
{
    private readonly IContinuousPPOAgent<TState> _agent;
    private readonly bool _useRnn;
    private readonly Dictionary<Guid, (Tensor?, Tensor?)?> _memoriesStore = new();

    public LocalContinuousPPOAgent(PPOAgentOptions options, int[] discreteActionDimensions, ContinuousActionDimensions[] continuousActionDimensions, StateDimensions stateDimensions)
    {
        _agent = PPOAgentFactory.ComposeContinuousPPOAgent<TState>(options, discreteActionDimensions, stateDimensions, continuousActionDimensions);
        _useRnn = options.UseRNN;
    }

    public async ValueTask<Dictionary<Guid, RLActions>> SelectBatchActionsAsync(IEnumerable<EnvironmentState<TState>> statesEnumerable, bool isTraining)
    {
        var states = statesEnumerable.ToList();
        var actionDict = new Dictionary<Guid, RLActions>();
        
        if (_useRnn)
        {
            var statesWithMemory = states.Select(info =>
            {
                if (!_memoriesStore.TryGetValue(info.EnvironmentId, out var memoryTuple))
                {
                    memoryTuple = null;
                    _memoriesStore[info.EnvironmentId] = memoryTuple;
                }
                
                return new RLMemoryState<TState>(info.State, memoryTuple?.Item1, memoryTuple?.Item2);
            }).ToArray();

            var actionsWithMemory = await _agent.SelectActionsRecurrentAsync(statesWithMemory, isTraining);

            for (var i = 0; i < states.Count; i++)
            {
                var environmentId = states[i].EnvironmentId;
                var action = actionsWithMemory[i].Actions;

                actionDict[environmentId] = action;
                _memoriesStore[environmentId] = (actionsWithMemory[i].MemoryState, actionsWithMemory[i].MemoryState2);
            }

            return actionDict;
        }
        
        var actions = await _agent.SelectActionsAsync(states.Select(x => x.State).ToArray(), isTraining);
        for (var i = 0; i < states.Count; i++)
        {
            var environmentId = states[i].EnvironmentId;
            var action = actions[i];
            actionDict[environmentId] = action;
        }

        return actionDict;
    }

    public ValueTask ResetStatesAsync(List<(Guid EnvironmentId, bool IsDone)> environments)
    {
        foreach (var (envId, isDone) in environments)
        {
            if (isDone && _memoriesStore.ContainsKey(envId))
            {
                _memoriesStore[envId] = (null, null);
            }
        }

        return new();
    }

    public ValueTask UploadTransitionsAsync(IEnumerable<Transition<TState>> transitions)
    {
        return _agent.AddTransitionsAsync(transitions);
    }

    public ValueTask OptimizeModelAsync()
    {
        return _agent.OptimizeModelAsync();
    }

    public ValueTask SaveAsync(string path)
    {
        return _agent.SaveAsync(path);
    }

    public ValueTask LoadAsync(string path)
    {
        return _agent.LoadAsync(path);
    }
}