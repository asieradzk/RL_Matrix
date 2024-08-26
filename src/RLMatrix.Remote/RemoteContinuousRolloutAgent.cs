using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.DependencyInjection;
using OneOf;
using RLMatrix.Agents.Common;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using RLMatrix.Common.Remote;

namespace RLMatrix.Agents.SignalR
{
    public class RemoteContinuousRolloutAgent<TState> : IContinuousRolloutAgent<TState>
    {
        private readonly HubConnection _connection;
        private readonly Dictionary<Guid, IContinuousEnvironmentAsync<TState>> _environments;
        private readonly Dictionary<Guid, Episode<TState>> _envPairs;
        private readonly bool _useRNN;

        public RemoteContinuousRolloutAgent(string hubUrl, PPOAgentOptions options, IEnumerable<IContinuousEnvironmentAsync<TState>> environments)
        {
            _connection = new HubConnectionBuilder()
                .WithUrl(hubUrl)
                .AddMessagePackProtocol()
                .Build();

            _environments = environments.ToDictionary(env => Guid.NewGuid(), env => env);
            _envPairs = _environments.ToDictionary(pair => pair.Key, pair => new Episode<TState>());
            _useRNN = options.UseRNN;

            InitializeAsync(options).GetAwaiter().GetResult();
        }

        private async Task InitializeAsync(PPOAgentOptions options)
        {
            await _connection.StartAsync();

            var firstEnv = _environments.First().Value;
            var discreteActionSizes = firstEnv.DiscreteActionSize;
            var continuousActionBounds = firstEnv.ContinuousActionBounds;
            var stateSizes = firstEnv.StateSize;

            var optsDTO = ((OneOf<DQNAgentOptions, PPOAgentOptions>)options).ToAgentOptionsDTO();
            var stateSizesDTO = stateSizes.ToStateSizesDTO();

            await _connection.InvokeAsync("Initialize", optsDTO, discreteActionSizes, continuousActionBounds, stateSizesDTO);
        }

        public async Task Step(bool isTraining = true)
        {
            List<Task<(Guid environmentId, TState state)>> stateTaskList = new List<Task<(Guid environmentId, TState state)>>();
            foreach (var env in _environments)
            {
                var stateTask = GetStateAsync(env.Key, env.Value);
                stateTaskList.Add(stateTask);
            }
            var stateResults = await Task.WhenAll(stateTaskList);

            List<(Guid environmentId, TState state)> payload = stateResults.ToList();
            var actions = await GetActionsBatchAsync(payload, isTraining);
            List<Task<(Guid environmentId, (float, bool) reward)>> rewardTaskList = new List<Task<(Guid environmentId, (float, bool) reward)>>();
            foreach (var action in actions)
            {
                var env = _environments[action.Key];
                var rewardTask = env.Step(action.Value.discreteActions, action.Value.continuousActions)
                    .ContinueWith(t => (action.Key, t.Result));
                rewardTaskList.Add(rewardTask);
            }

            await Task.WhenAll(rewardTaskList);

            List<Task<(Guid environmentId, TState state)>> nextStateTaskList = new List<Task<(Guid environmentId, TState state)>>();
            foreach (var env in _environments)
            {
                var stateTask = GetStateAsync(env.Key, env.Value);
                nextStateTaskList.Add(stateTask);
            }
            var nextStateResults = await Task.WhenAll(nextStateTaskList);

            var transitionsByEnvironment = new Dictionary<Guid, List<TransitionPortable<TState>>>();
            var rewards = new ConcurrentBag<double>();
            var completedEpisodes = new List<(Guid environmentId, bool done)>();

            foreach (var env in _environments)
            {
                var key = env.Key;
                var episode = _envPairs[key];
                var stateResult = stateResults.First(x => x.environmentId == key);
                var state = stateResult.state;
                var action = actions[key];
                var stepResult = rewardTaskList.First(x => x.Result.environmentId == key).Result;
                var reward = stepResult.reward.Item1;
                var isDone = stepResult.reward.Item2;
                var nextState = nextStateResults.First(x => x.environmentId == key).state;
                episode.AddTransition(state, isDone, action.discreteActions, action.continuousActions, reward);

                if (isDone)
                {
                    transitionsByEnvironment[key] = episode.CompletedEpisodes.ToList();
                    rewards.Add(episode.cumulativeReward);
                    episode.CompletedEpisodes.Clear();
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

        public void StepSync(bool isTraining = true)
        {
            List<(Guid environmentId, TState state)> stateResults = new List<(Guid environmentId, TState state)>();
            foreach (var env in _environments)
            {
                var state = GetStateSync(env.Key, env.Value);
                stateResults.Add(state);
            }

            var actions = GetActionsBatchSync(stateResults, isTraining);

            List<(Guid environmentId, (float, bool) reward)> rewardResults = new List<(Guid environmentId, (float, bool) reward)>();
            foreach (var action in actions)
            {
                var env = _environments[action.Key];
                var reward = env.Step(action.Value.discreteActions, action.Value.continuousActions).GetAwaiter().GetResult();
                rewardResults.Add((action.Key, reward));
            }

            List<(Guid environmentId, TState state)> nextStateResults = new List<(Guid environmentId, TState state)>();
            foreach (var env in _environments)
            {
                var state = GetStateSync(env.Key, env.Value);
                nextStateResults.Add(state);
            }

            var transitionsByEnvironment = new Dictionary<Guid, List<TransitionPortable<TState>>>();
            var rewards = new List<double>();
            var completedEpisodes = new List<(Guid environmentId, bool done)>();

            foreach (var env in _environments)
            {
                var key = env.Key;
                var episode = _envPairs[key];
                var stateResult = stateResults.First(x => x.environmentId == key);
                var state = stateResult.state;
                var action = actions[key];
                var stepResult = rewardResults.First(x => x.environmentId == key);
                var reward = stepResult.Item2.Item1;
                var isDone = stepResult.Item2.Item2;
                var nextState = nextStateResults.First(x => x.environmentId == key).state;
                episode.AddTransition(state, isDone, action.discreteActions, action.continuousActions, reward);
                if (isDone)
                {
                    transitionsByEnvironment[key] = episode.CompletedEpisodes.ToList();
                    rewards.Add(episode.cumulativeReward);
                    episode.CompletedEpisodes.Clear();
                    completedEpisodes.Add((key, true));
                }
            }

            if (isTraining)
            {
                foreach (var kvp in transitionsByEnvironment)
                {
                    if (kvp.Value.Count > 0)
                    {
                        _connection.InvokeAsync("UploadTransitions", kvp.Value.ToDTOList()).GetAwaiter().GetResult();
                    }
                }
            }

            _connection.InvokeAsync("ResetStates", completedEpisodes).GetAwaiter().GetResult();

            if (isTraining)
            {
                _connection.InvokeAsync("OptimizeModel").GetAwaiter().GetResult();
            }
        }

#if NET8_0_OR_GREATER
        public async ValueTask<Dictionary<Guid, (int[] discreteActions, float[] continuousActions)>> GetActionsBatchAsync(List<(Guid environmentId, TState state)> stateInfos, bool isTraining)
#else
        public async Task<Dictionary<Guid, (int[] discreteActions, float[] continuousActions)>> GetActionsBatchAsync(List<(Guid environmentId, TState state)> stateInfos, bool isTraining)
#endif
        {

            var stateInfoDTOs = stateInfos.PackList();
            var actionResponse = await _connection.InvokeAsync<ActionResponseDTO>("SelectActions", stateInfoDTOs, isTraining);

            return actionResponse.Actions.ToDictionary(
                kvp => kvp.Key,
                kvp => (kvp.Value.DiscreteActions, kvp.Value.ContinuousActions)
            );
        }

        private TState DeepCopy(TState input)
        {
            if (input is float[] array1D)
            {
                return (TState)(object)array1D.ToArray();
            }
            else if (input is float[,] array2D)
            {
                int rows = array2D.GetLength(0);
                int cols = array2D.GetLength(1);
                var clone = new float[rows, cols];
                Buffer.BlockCopy(array2D, 0, clone, 0, array2D.Length * sizeof(float));
                return (TState)(object)clone;
            }
            else
            {
                throw new InvalidOperationException("This method can only be used with float[] or float[,].");
            }
        }

        private async Task<(Guid environmentId, TState state)> GetStateAsync(Guid environmentId, IContinuousEnvironmentAsync<TState> env)
        {
            var state = DeepCopy(await env.GetCurrentState());
            return (environmentId, state);
        }


        private (Guid environmentId, TState state) GetStateSync(Guid environmentId, IContinuousEnvironmentAsync<TState> env)
        {
            var state = DeepCopy(env.GetCurrentState().GetAwaiter().GetResult());
            return (environmentId, state);
        }

        private Dictionary<Guid, (int[] discreteActions, float[] continuousActions)> GetActionsBatchSync(List<(Guid environmentId, TState state)> stateInfos, bool isTraining)
        {
            var stateInfoDTOs = stateInfos.PackList();
            var actionResponse = _connection.InvokeAsync<ActionResponseDTO>("SelectActions", stateInfoDTOs, isTraining).ConfigureAwait(false).GetAwaiter().GetResult();

            return actionResponse.Actions.ToDictionary(
                kvp => kvp.Key,
                kvp => (kvp.Value.DiscreteActions, kvp.Value.ContinuousActions)
            );
        }



#if NET8_0_OR_GREATER
        public async ValueTask Save()
#else
        public async Task Save()
#endif
        {
            await _connection.InvokeAsync("Save");
        }

#if NET8_0_OR_GREATER
        public async ValueTask Load()
#else
        public async Task Load()
#endif
        {
            await _connection.InvokeAsync("Load");
        }

#if NET8_0_OR_GREATER
        public ValueTask Save(string path)
#else
        public Task Save(string path)
#endif
        {
            Console.WriteLine("Path not supported in remote agent.");
            _connection.InvokeAsync("Save");
#if NET8_0_OR_GREATER
            return ValueTask.CompletedTask;
#else
            return Task.CompletedTask;
#endif
        }

#if NET8_0_OR_GREATER
        public ValueTask Load(string path)
#else
        public Task Load(string path)
#endif
        {
            Console.WriteLine("Path not supported in remote agent.");
            _connection.InvokeAsync("Load");
#if NET8_0_OR_GREATER
            return ValueTask.CompletedTask;
#else
            return Task.CompletedTask;
#endif
        }
    }
}