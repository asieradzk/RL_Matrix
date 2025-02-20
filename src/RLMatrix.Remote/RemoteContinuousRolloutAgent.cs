using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.DependencyInjection;
using RLMatrix.Common;

namespace RLMatrix.Remote;

public sealed class RemoteContinuousRolloutAgent<TState> : IContinuousRolloutAgent
    where TState : notnull
{
    private readonly SemaphoreSlim _semaphore = new(1, 1);
    private readonly PPOAgentOptions _options;
    private bool _initialized;
    
    private readonly HubConnection _connection;
    private readonly Dictionary<Guid, IContinuousEnvironment<TState>> _environments;
    private readonly Dictionary<Guid, Episode<TState>> _envPairs;

    public RemoteContinuousRolloutAgent(string hubUrl, PPOAgentOptions options, IEnumerable<IContinuousEnvironment<TState>> environments)
    {
        _options = options;
        _connection = new HubConnectionBuilder()
            .WithUrl(hubUrl)
            .AddMessagePackProtocol()
            .Build();

        _environments = environments.ToDictionary(_ => Guid.NewGuid());
        _envPairs = _environments.ToDictionary(pair => pair.Key, _ => new Episode<TState>());
    }

    public async ValueTask StepAsync(bool isTraining = true)
    {
        await WaitUntilInitializedAsync();
        
        var stateTaskList = new List<Task<EnvironmentState<TState>>>();
        foreach (var env in _environments)
        {
            var stateTask = GetStateAsync(env.Key, env.Value);
            stateTaskList.Add(stateTask);
        }
        
        var states = await Task.WhenAll(stateTaskList);

        var actions = await GetBatchActionsAsync(states.ToList(), isTraining);
        var rewardTaskList = new List<Task<(Guid EnvironmentId, StepResult Result)>>();
        foreach (var action in actions)
        {
            var env = _environments[action.Key];
            var rewardTask = Task.Run(async () =>
            {
                var reward = await env.StepAsync(action.Value);
                return (action.Key, reward);
            });
            
            rewardTaskList.Add(rewardTask);
        }

        await Task.WhenAll(rewardTaskList);

        /* TODO: next states are...unused?
        var nextStateTaskList = new List<Task<EnvironmentState<TState>>>();
        foreach (var env in _environments)
        {
            var stateTask = GetStateAsync(env.Key, env.Value);
            nextStateTaskList.Add(stateTask);
        }
        var nextStateResults = await Task.WhenAll(nextStateTaskList);*/

        var transitionsByEnvironment = new Dictionary<Guid, List<Transition<TState>>>();
        // var rewards = new ConcurrentBag<double>(); TODO: rewards are...unused?
        var completedEpisodes = new List<(Guid environmentId, bool done)>();

        foreach (var env in _environments)
        {
            var key = env.Key;
            var episode = _envPairs[key];
            var stateResult = states.First(x => x.EnvironmentId == key);
            var state = stateResult.State;
            var action = actions[key];
            var (_, result) = rewardTaskList.First(x => x.Result.EnvironmentId == key).Result;
            //var nextState = nextStateResults.First(x => x.EnvironmentId == key).State;
            episode.AddTransition(state, result.IsDone, RLActions.Continuous(action.DiscreteActions, action.ContinuousActions), result.Reward);

            if (result.IsDone)
            {
                transitionsByEnvironment[key] = episode.CompletedTransitions.ToList();
                //rewards.Add(episode.CumulativeReward); uncomment if restoring rewards
                episode.CompletedTransitions.Clear();
                completedEpisodes.Add((key, true));
            }
        }

        if (isTraining)
        {
            foreach (var kvp in transitionsByEnvironment)
            {
                if (kvp.Value.Count > 0)
                {
                    await _connection.InvokeAsync("UploadTransitions", kvp.Value.ToDTOList());
                }
            }
        }

        await _connection.InvokeAsync("ResetStates", completedEpisodes);

        if (isTraining)
        {
            await _connection.InvokeAsync("OptimizeModel");
        }
    }

    public async ValueTask<Dictionary<Guid, RLActions>> GetBatchActionsAsync(List<EnvironmentState<TState>> states, bool isTraining)
    {
        await WaitUntilInitializedAsync();
        
        var stateInfoDTOs = states.ToDTOList();
        var actionResponse = await _connection.InvokeAsync<ActionResponseDTO>("SelectActions", stateInfoDTOs, isTraining);

        return actionResponse.Actions.ToDictionary(
            kvp => kvp.Key,
            kvp => RLActions.Continuous(kvp.Value.DiscreteActions, kvp.Value.ContinuousActions)
        );
    }
    
    public async ValueTask SaveAsync()
    {
        if (!_initialized)
            throw new InvalidOperationException("Connection not initialized.");
        
        await _connection.InvokeAsync("Save");
    }

    public async ValueTask LoadAsync()
    {
        if (!_initialized)
            throw new InvalidOperationException("Connection not initialized.");
        
        await _connection.InvokeAsync("Load");
    }

    private async Task WaitUntilInitializedAsync()
    {
        if (_initialized) // after the first time, this should hopefully short-circuit to prevent any waiting or deadlocks.
            return;
        
        // TODO: I don't like doing async work in the ctor, so we're gonna lazy init the first time anything does anything.
        await _semaphore.WaitAsync();
        if (_initialized) // check again, just in case.
            return;
        
        await _connection.StartAsync();

        var firstEnv = _environments.First().Value;
        var discreteActionDimensions = firstEnv.DiscreteActionDimensions;
        var continuousActionDimensions = firstEnv.ContinuousActionDimensions;
        var stateDimensions = firstEnv.StateDimensions;

        var optsDTO = _options.ToAgentOptionsDTO();
        var stateSizesDTO = stateDimensions.ToStateSizesDTO();

        await _connection.InvokeAsync("Initialize", optsDTO, discreteActionDimensions, continuousActionDimensions, stateSizesDTO);
        _initialized = true;
        _semaphore.Release();
    }
    
    private static async Task<EnvironmentState<TState>> GetStateAsync(Guid environmentId, IContinuousEnvironment<TState> env)
    {
        var remoteState = await env.GetCurrentStateAsync();
        var state = Utilities<TState>.DeepCopy(remoteState);
        return new EnvironmentState<TState>(environmentId, state);
    }

    ValueTask ISavable.SaveAsync(string path) => SaveAsync();
    ValueTask ISavable.LoadAsync(string path) => LoadAsync();
}