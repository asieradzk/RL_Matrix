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
    public class LocalDiscretePPOAgent<T> : IDiscreteProxy<T>
    {
        private readonly IDiscretePPOAgent<T> _agent;
        bool useRnn = false;
        private Dictionary<Guid, (Tensor?, Tensor?)?> memoriesStore = new();

        public LocalDiscretePPOAgent(PPOAgentOptions options, int[] ActionSizes, OneOf<int, (int, int)> StateSizes)
        {
            _agent = PPOAgentFactory<T>.ComposeDiscretePPOAgent(options, ActionSizes, StateSizes);
            useRnn = options.UseRNN;
        }

#if NET8_0_OR_GREATER
        public ValueTask LoadAsync(string path)
#else
        public Task LoadAsync(string path)
#endif
        {
            _agent.Load(path);
#if NET8_0_OR_GREATER
            return ValueTask.CompletedTask;
#else
            return Task.CompletedTask;
#endif
        }

#if NET8_0_OR_GREATER
        public ValueTask OptimizeModelAsync()
#else
        public Task OptimizeModelAsync()
#endif
        {
            _agent.OptimizeModel();
#if NET8_0_OR_GREATER
            return ValueTask.CompletedTask;
#else
            return Task.CompletedTask;
#endif
        }

#if NET8_0_OR_GREATER
        public ValueTask ResetStates(List<(Guid environmentId, bool dones)> environmentIds)
#else
        public Task ResetStates(List<(Guid environmentId, bool dones)> environmentIds)
#endif
        {
            foreach (var (envId, done) in environmentIds)
            {
                if (done && memoriesStore.ContainsKey(envId))
                {
                    memoriesStore[envId] = (null, null);
                }
            }
#if NET8_0_OR_GREATER
            return ValueTask.CompletedTask;
#else
            return Task.CompletedTask;
#endif
        }

#if NET8_0_OR_GREATER
        public ValueTask SaveAsync(string path)
#else
        public Task SaveAsync(string path)
#endif
        {
            _agent.Save(path);
#if NET8_0_OR_GREATER
            return ValueTask.CompletedTask;
#else
            return Task.CompletedTask;
#endif
        }

#if NET8_0_OR_GREATER
        public ValueTask<Dictionary<Guid, int[]>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos, bool isTraining)
#else
        public Task<Dictionary<Guid, int[]>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos, bool isTraining)
#endif
        {
            Dictionary<Guid, int[]> actionDict = new Dictionary<Guid, int[]>();

            if (useRnn)
            {
                (T state, Tensor? memoryState, Tensor? memoryState2)[] statesWithMemory = stateInfos.Select(info =>
                {
                    if (!memoriesStore.TryGetValue(info.environmentId, out var memoryTuple))
                    {
                        memoryTuple = null;
                        memoriesStore[info.environmentId] = memoryTuple;
                    }
                    return (info.state, memoryTuple?.Item1, memoryTuple?.Item2);
                }).ToArray();

                (int[] actions, Tensor? memoryState, Tensor? memoryState2)[] actionsWithMemory = _agent.SelectActionsRecurrent(statesWithMemory, isTraining);

                for (int i = 0; i < stateInfos.Count; i++)
                {
                    Guid environmentId = stateInfos[i].environmentId;
                    int[] action = actionsWithMemory[i].actions;
                    memoriesStore[environmentId] = (actionsWithMemory[i].memoryState, actionsWithMemory[i].memoryState2);
                    actionDict[environmentId] = action;
                }
            }
            else
            {
                T[] states = stateInfos.Select(info => info.state).ToArray();
                int[][] actions = _agent.SelectActions(states, isTraining);

                for (int i = 0; i < stateInfos.Count; i++)
                {
                    Guid environmentId = stateInfos[i].environmentId;
                    int[] action = actions[i];
                    actionDict[environmentId] = action;
                }
            }

#if NET8_0_OR_GREATER
            return ValueTask.FromResult(actionDict);
#else
            return Task.FromResult(actionDict);
#endif
        }

#if NET8_0_OR_GREATER
        public ValueTask UploadTransitionsAsync(IEnumerable<TransitionPortable<T>> transitions)
#else
        public Task UploadTransitionsAsync(IEnumerable<TransitionPortable<T>> transitions)
#endif
        {
            _agent.AddTransition(transitions);
#if NET8_0_OR_GREATER
            return ValueTask.CompletedTask;
#else
            return Task.CompletedTask;
#endif
        }
    }
}