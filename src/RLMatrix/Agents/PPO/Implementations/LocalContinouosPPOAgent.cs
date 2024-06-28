using OneOf;
using RLMatrix.Agents.Common;
using Tensorboard;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace RLMatrix.Agents.PPO.Implementations
{
    public class LocalContinuousPPOAgent<T> : IContinuousProxy<T>
    {
        private readonly IContinuousPPOAgent<T> _agent;
        bool useRnn = false;
        private Dictionary<Guid, (Tensor?, Tensor?)?> memoriesStore = new Dictionary<Guid, (Tensor?, Tensor?)?>();

        public LocalContinuousPPOAgent(PPOAgentOptions options, int[] DiscreteDimensions, OneOf<int, (int, int)> StateSizes, (float min, float max)[] ContinuousActionBounds)
        {
            _agent = PPOAgentFactory<T>.ComposeContinuousPPOAgent(options, DiscreteDimensions, StateSizes, ContinuousActionBounds);
            useRnn = options.UseRNN;
        }

#if NET8_0_OR_GREATER
        public ValueTask LoadAsync(string path)
        {
            _agent.Load(path);
            return ValueTask.CompletedTask;
        }

        public ValueTask OptimizeModelAsync()
        {
            _agent.OptimizeModel();
            return ValueTask.CompletedTask;
        }

        public ValueTask ResetStates(List<(Guid environmentId, bool dones)> environmentIds)
        {
            foreach (var (envId, done) in environmentIds)
            {
                if (done && memoriesStore.ContainsKey(envId))
                {
                    memoriesStore[envId] = (null, null);
                }
            }
            return ValueTask.CompletedTask;
        }

        public ValueTask SaveAsync(string path)
        {
            _agent.Save(path);
            return ValueTask.CompletedTask;
        }

        public ValueTask<Dictionary<Guid, (int[] discreteActions, float[] continuousActions)>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos, bool isTraining)
#else
        public Task LoadAsync(string path)
        {
            _agent.Load(path);
            return Task.CompletedTask;
        }

        public Task OptimizeModelAsync()
        {
            _agent.OptimizeModel();
            return Task.CompletedTask;
        }

        public Task ResetStates(List<(Guid environmentId, bool dones)> environmentIds)
        {
            foreach (var (envId, done) in environmentIds)
            {
                if (done && memoriesStore.ContainsKey(envId))
                {
                    memoriesStore[envId] = (null, null);
                }
            }
            return Task.CompletedTask;
        }

        public Task SaveAsync(string path)
        {
            _agent.Save(path);
            return Task.CompletedTask;
        }

        public Task<Dictionary<Guid, (int[] discreteActions, float[] continuousActions)>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos, bool isTraining)
#endif
        {
            if (useRnn)
            {
                Dictionary<Guid, (int[] discreteActions, float[] continuousActions)> actionDict = new Dictionary<Guid, (int[] discreteActions, float[] continuousActions)>();

                (T state, Tensor? memoryState, Tensor? memoryState2)[] statesWithMemory = stateInfos.Select(info =>
                {
                    if (!memoriesStore.TryGetValue(info.environmentId, out var memoryTuple))
                    {
                        memoryTuple = null;
                        memoriesStore[info.environmentId] = memoryTuple;
                    }
                    return (info.state, memoryTuple?.Item1, memoryTuple?.Item2);
                }).ToArray();

                ((int[] discreteActions, float[] continuousActions) actions, Tensor? memoryState, Tensor? memoryState2)[] actionsWithMemory = _agent.SelectActionsRecurrent(statesWithMemory, isTraining);

                for (int i = 0; i < stateInfos.Count; i++)
                {
                    Guid environmentId = stateInfos[i].environmentId;
                    (int[] discreteActions, float[] continuousActions) action = actionsWithMemory[i].actions;

                    actionDict[environmentId] = action;

                    memoriesStore[environmentId] = (actionsWithMemory[i].memoryState, actionsWithMemory[i].memoryState2);
                }

#if NET8_0_OR_GREATER
                return ValueTask.FromResult(actionDict);
#else
                return Task.FromResult(actionDict);
#endif
            }
            else
            {
                T[] states = stateInfos.Select(info => info.state).ToArray();
                (int[] discreteActions, float[] continuousActions)[] actions = _agent.SelectActions(states, isTraining);
                Dictionary<Guid, (int[] discreteActions, float[] continuousActions)> actionDict = new Dictionary<Guid, (int[] discreteActions, float[] continuousActions)>();

                for (int i = 0; i < stateInfos.Count; i++)
                {
                    Guid environmentId = stateInfos[i].environmentId;
                    (int[] discreteActions, float[] continuousActions) action = actions[i];
                    actionDict[environmentId] = action;
                }

#if NET8_0_OR_GREATER
                return ValueTask.FromResult(actionDict);
#else
                return Task.FromResult(actionDict);
#endif
            }
        }

#if NET8_0_OR_GREATER
        public ValueTask UploadTransitionsAsync(IEnumerable<TransitionPortable<T>> transitions)
        {
            _agent.AddTransition(transitions);
            return ValueTask.CompletedTask;
        }
#else
        public Task UploadTransitionsAsync(IEnumerable<TransitionPortable<T>> transitions)
        {
            _agent.AddTransition(transitions);
            return Task.CompletedTask;
        }
#endif
    }
}