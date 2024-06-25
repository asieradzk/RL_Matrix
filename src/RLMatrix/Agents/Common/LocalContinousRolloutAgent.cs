﻿using RLMatrix.Agents.DQN.Domain;
using RLMatrix.Agents.PPO.Implementations;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLMatrix.Agents.Common
{
    public partial class LocalContinuousRolloutAgent<TState> : IContinuousRolloutAgent<TState>
    {
        protected readonly Dictionary<Guid, IContinuousEnvironmentAsync<TState>> _environments;
        protected readonly Dictionary<Guid, Episode<TState>> _ennvPairs;
        protected readonly IContinuousProxy<TState> _agent;
        protected readonly IRLChartService? _chartService;

        public LocalContinuousRolloutAgent(PPOAgentOptions options, IEnumerable<IContinuousEnvironmentAsync<TState>> environments, IRLChartService chartService = null)
        {
            _environments = environments.ToDictionary(env => Guid.NewGuid(), env => env);
            _ennvPairs = _environments.ToDictionary(pair => pair.Key, pair => new Episode<TState>());
            _chartService = chartService;
            _agent = new LocalContinuousPPOAgent<TState>(options, environments.First().DiscreteActionSize, environments.First().StateSize, environments.First().ContinuousActionBounds);
        }

        List<double> chart = new();

        public async Task Step(bool isTraining = true)
        {
            //GETS INITIAL STATES FOR ALL ENVS
            List<Task<(Guid environmentId, TState state)>> stateTaskList = new();
            foreach (var env in _environments)
            {
                var stateTask = GetStateAsync(env.Key, env.Value);
                stateTaskList.Add(stateTask);
            }
            var stateResults = await Task.WhenAll(stateTaskList);

            //GETS ACTIONS FOR ALL DETERMINED STATES
            List<(Guid environmentId, TState state)> payload = stateResults.ToList();
            var actions = await _agent.SelectActionsBatchAsync(payload, isTraining);

            //STEPS ALL ENVS AND GETS REWARDS AND DONES
            List<Task<(Guid environmentId, (float, bool) reward)>> rewardTaskList = new();
            foreach (var action in actions)
            {
                var env = _environments[action.Key];
                var rewardTask = env.Step(action.Value.discreteActions, action.Value.continuousActions)
                    .ContinueWith(t => (action.Key, t.Result));  // Ensure that the task returns a tuple with the Guid
                rewardTaskList.Add(rewardTask);
            }

            await Task.WhenAll(rewardTaskList);

            //GETS NEXT STATES FOR ALL ENVS
            List<Task<(Guid environmentId, TState state)>> nextStateTaskList = new();
            foreach (var env in _environments)
            {
                var stateTask = GetStateAsync(env.Key, env.Value);
                nextStateTaskList.Add(stateTask);
            }
            var nextStateResults = await Task.WhenAll(nextStateTaskList);

            //Process episodes
            ConcurrentBag<TransitionPortable<TState>> transitionsToShip = new();
            ConcurrentBag<double> rewards = new();
            var completedEpisodes = new List<(Guid environmentId, bool done)>();

            foreach (var env in _environments)
            {
                var key = env.Key;
                var episode = _ennvPairs[key];
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

            if (transitionsToShip.Count > 0 && isTraining)
            {
                await _agent.UploadTransitionsAsync(transitionsToShip.ToList());
            }
            else
            {
                transitionsToShip.Clear();
            }

            await _agent.ResetStates(completedEpisodes);

            if (!isTraining)
                return;

            await _agent.OptimizeModelAsync();
        }

        public async ValueTask<Dictionary<Guid, (int[] discreteActions, float[] continuousActions)>> GetActionsBatchAsync(List<(Guid environmentId, TState state)> stateInfos, bool isTraining)
        {
            return await _agent.SelectActionsBatchAsync(stateInfos, isTraining);
        }

        private TState DeepCopy(TState input)
        {
            if (input is float[] array1D)
            {
                return (TState)(object)array1D.ToArray(); // Create a new array with the same elements
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

        public ValueTask Save(string path)
        {
            return _agent.SaveAsync(path);
        }

        public ValueTask Load(string path)
        {
            return _agent.LoadAsync(path);
        }
    }
}