using System.Collections.Concurrent;
using RLMatrix.Common;

namespace RLMatrix;

// TODO: This base class reduces code duplication by a lot via abstraction. Unsure if the continuous and discrete variants are needed or just a "nice to have".
public abstract class LocalRolloutAgent<TState, TProxy, TEnvironment> : IRolloutAgent
    where TState : notnull
    where TProxy : IProxy<TState>
    where TEnvironment : IEnvironment<TState>
{
    private protected readonly Dictionary<Guid, TEnvironment> _environments;
    private protected readonly Dictionary<Guid, Episode<TState>> _episodes;
    private protected readonly TProxy _proxy;

    protected LocalRolloutAgent(ICollection<TEnvironment> environments, TProxy proxy)
    {
        _environments = environments.ToDictionary(_ => Guid.NewGuid());
        _episodes = _environments.ToDictionary(pair => pair.Key, _ => new Episode<TState>());
        _proxy = proxy;
    }

    public async ValueTask StepAsync(bool isTraining = true)
    {
        var stateTasks = _environments.Select(env => GetStateAsync(env.Key, env.Value).AsTask());
        var stateResults = await Task.WhenAll(stateTasks);

        var actionsList = await _proxy.SelectBatchActionsAsync(stateResults, isTraining);
        
        var rewardTasks = new List<Task<(Guid EnvironmentId, StepResult Result)>>();
        foreach (var pair in actionsList)
        {
            var id = pair.Key;
            var actions = pair.Value;
            
            var env = _environments[id];
            var rewardTask = Task.Run(async () =>
            {
                var result = await env.StepAsync(actions);
                return (id, result);
            });
            
            rewardTasks.Add(rewardTask);
        }
        
        var rewardTaskResults = await Task.WhenAll(rewardTasks);

        await Task.WhenAll(rewardTasks);

        // TODO: "next state"s are never consumed. Unfunished feature or a bug?
        //var nextStateTasks = _environments.Select(env => GetStateAsync(env.Key, env.Value).AsTask());
        //var nextStateResults = await Task.WhenAll(nextStateTasks);

        ConcurrentBag<Transition<TState>> transitionsToShip = [];
        //ConcurrentBag<float> rewards = []; TODO: rewards is only added to and is never consumed.
        List<(Guid EnvironmentId, bool IsDone)> completedTransitions = [];

        foreach (var pair in _environments)
        {
            var key = pair.Key;
            var episode = _episodes[key];
            var stateResult = stateResults.First(x => x.EnvironmentId == key);
            var state = stateResult.State;
            var actions = actionsList[key];
            var (reward, isDone) = rewardTaskResults.First(x => x.EnvironmentId == key).Result;
            // var nextState = nextStateResults.First(x => x.EnvironmentId == key).State;
            episode.AddTransition(state, isDone, actions, reward);

            if (isDone)
            {
                foreach (var transition in episode.CompletedTransitions)
                {
                    transitionsToShip.Add(transition);
                }
                
                // rewards.Add(episode.CumulativeReward); uncomment if restoring rewards
                episode.CompletedTransitions.Clear();
                completedTransitions.Add((key, true));
            }
        }

        if (!transitionsToShip.IsEmpty)
        {
            await _proxy.UploadTransitionsAsync(transitionsToShip);
        }
        else
        {
            // TODO: transitionsToShip is never used after this point. Is clearing necessary?
#if NET8_0_OR_GREATER
            transitionsToShip.Clear();
#else
            transitionsToShip = [];
#endif
        }

        await _proxy.ResetStatesAsync(completedTransitions);

        if (isTraining)
            await _proxy.OptimizeModelAsync();
    }

    /// <inheritdoc />
    public ValueTask SaveAsync(string path)
    {
        return _proxy.SaveAsync(path);
    }

    /// <inheritdoc />
    public ValueTask LoadAsync(string path)
    {
        return _proxy.LoadAsync(path);
    }

    /// <inheritdoc cref="IProxy{TState}.SelectBatchActionsAsync" />
    public async Task<IReadOnlyDictionary<Guid, RLActions>> SelectBatchActionsAsync(List<EnvironmentState<TState>> states, bool isTraining)
    {
        var dict = await _proxy.SelectBatchActionsAsync(states, isTraining);
        return dict;
    }

    private static async ValueTask<EnvironmentState<TState>> GetStateAsync(Guid environmentId, TEnvironment env)
    {
        var state = await env.GetCurrentStateAsync();
        return new EnvironmentState<TState>(environmentId, Utilities<TState>.DeepCopy(state));
    }
}