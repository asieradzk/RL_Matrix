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
    //TODO: DRY violation, this can be somehow married with DiscreteRolloutAgent
    public class RemoteDiscreteRolloutAgent<TState> : IDiscreteRolloutAgent<TState>
    {
        private readonly HubConnection _connection;
        private readonly Dictionary<Guid, IEnvironmentAsync<TState>> _environments;
        private readonly Dictionary<Guid, Episode<TState>> _envPairs;
        private readonly IRLChartService? _chartService;

        public RemoteDiscreteRolloutAgent(string hubUrl, IAgentOptions options, IEnumerable<IEnvironmentAsync<TState>> environments, IRLChartService chartService = null)
        {
            _connection = new HubConnectionBuilder()
                .WithUrl(hubUrl)
                .AddMessagePackProtocol()
                .Build();


            _environments = environments.ToDictionary(env => Guid.NewGuid(), env => env);
            _envPairs = _environments.ToDictionary(pair => pair.Key, pair => new Episode<TState>());
            _chartService = chartService;
            //cast options to concrete type
            var optionsType = options.GetType();
            if (optionsType == typeof(DQNAgentOptions))
            {
                InitializeAsync((DQNAgentOptions)options).GetAwaiter().GetResult();
            }
            else if (optionsType == typeof(PPOAgentOptions))
            {
                InitializeAsync((PPOAgentOptions)options).GetAwaiter().GetResult();
            }
            else
            {
                throw new ArgumentException("Invalid options type");
            }
        }

        private async Task InitializeAsync(OneOf<DQNAgentOptions, PPOAgentOptions> options)
        {   
            await _connection.StartAsync();

            var actionSizes = _environments.First().Value.actionSize;
            var stateSizes = _environments.First().Value.stateSize;

            var optsDTO = options.ToAgentOptionsDTO();
            var stateSizesDTO = stateSizes.ToStateSizesDTO();

            await _connection.InvokeAsync("Initialize", optsDTO, actionSizes, stateSizesDTO);
        }
        List<double> chart = new();
        public async Task Step(bool isTraining = true)
        {
            List<Task<(Guid environmentId, TState state)>> stateTaskList = new();
            foreach (var env in _environments)
            {
                var stateTask = GetStateAsync(env.Key, env.Value);
                stateTaskList.Add(stateTask);
            }
            var stateResults = await Task.WhenAll(stateTaskList);

            List<(Guid environmentId, TState state)> payload = stateResults.ToList();
            var actions = await GetActionsBatchAsync(payload);

            List<Task<(Guid environmentId, (float, bool) reward)>> rewardTaskList = new();
            foreach (var action in actions)
            {
                var env = _environments[action.Key];
                var rewardTask = env.Step(action.Value)
                    .ContinueWith(t => (action.Key, t.Result));
                rewardTaskList.Add(rewardTask);
            }

            await Task.WhenAll(rewardTaskList);

            List<Task<(Guid environmentId, TState state)>> nextStateTaskList = new();
            foreach (var env in _environments)
            {
                var stateTask = GetStateAsync(env.Key, env.Value);
                nextStateTaskList.Add(stateTask);
            }
            var nextStateResults = await Task.WhenAll(nextStateTaskList);

            ConcurrentBag<TransitionPortable<TState>> transitionsToShip = new();
            ConcurrentBag<double> rewards = new();
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
                episode.AddTransition(state, isDone, action, reward);
                if (isDone)
                {
                    foreach (var transition in episode.CompletedEpisodes)
                    {
                        transitionsToShip.Add(transition);
                    }
                    rewards.Add(episode.cumulativeReward);
                    episode.CompletedEpisodes.Clear();
                    completedEpisodes.Add((key, true));
                }
            }

            if (_chartService != null)
            {
                chart.AddRange(rewards);
                //remove starting from 0 till it has less than 100 elements
                chart.RemoveRange(0, Math.Max(0, chart.Count - 100));

                _chartService.CreateOrUpdateChart(chart);
            }

            if (transitionsToShip.Count > 0)
            {
                await _connection.InvokeAsync("UploadTransitions", transitionsToShip.ToList().ToDTOList());
            }

            await _connection.InvokeAsync("ResetStates", completedEpisodes);
            await _connection.InvokeAsync("OptimizeModel");
        }

        public async ValueTask<Dictionary<Guid, int[]>> GetActionsBatchAsync(List<(Guid environmentId, TState state)> stateInfos)
        {
            var stateInfoDTOs = stateInfos.PackList();
            var actionsDictionary = await _connection.InvokeAsync<Dictionary<Guid, int[]>>("SelectActions", stateInfoDTOs);

            return actionsDictionary;
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

        private async Task<(Guid environmentId, TState state)> GetStateAsync(Guid environmentId, IEnvironmentAsync<TState> env)
        {
            var state = DeepCopy(await env.GetCurrentState());
            return (environmentId, state);
        }

        public async ValueTask Save()
        {
            await _connection.InvokeAsync("Save");
        }

        public async ValueTask Load()
        {
            await _connection.InvokeAsync("Load");
        }

        public ValueTask Save(string path)
        {
            Console.WriteLine("Path not supported in remote agent.");
            _connection.InvokeAsync("Save");
            return ValueTask.CompletedTask;
        }

        public ValueTask Load(string path)
        {
            Console.WriteLine("Path not supported in remote agent.");
            _connection.InvokeAsync("Load");
            return ValueTask.CompletedTask;
        }
    }
}