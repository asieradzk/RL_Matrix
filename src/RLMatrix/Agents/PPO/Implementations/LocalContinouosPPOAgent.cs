using OneOf;
using RLMatrix.Agents.Common;
using Tensorboard;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

namespace RLMatrix.Agents.PPO.Implementations
{
    public class LocalContinuousPPOAgent<T> : IContinuousProxy<T>
    {
        private readonly IContinuousPPOAgent<T> _agent;
        bool useRnn = false;
        private Dictionary<Guid, (Tensor?, Tensor?)?> memoriesStore = new();

        public LocalContinuousPPOAgent(PPOAgentOptions options, int[] DiscreteDimensions, OneOf<int, (int, int)> StateSizes, (float min, float max)[] ContinuousActionBounds)
        {
            _agent = PPOAgentFactory<T>.ComposeContinuousPPOAgent(options, DiscreteDimensions, StateSizes, ContinuousActionBounds);
            useRnn = options.UseRNN;
        }

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

                return ValueTask.FromResult(actionDict);
            }
            else
            {
                // Extract the states from the stateInfos list
                T[] states = stateInfos.Select(info => info.state).ToArray();

                // Select actions for the batch of states
                (int[] discreteActions, float[] continuousActions)[] actions = _agent.SelectActions(states, isTraining);

                // Create a dictionary to map environment IDs to their corresponding actions
                Dictionary<Guid, (int[] discreteActions, float[] continuousActions)> actionDict = new Dictionary<Guid, (int[] discreteActions, float[] continuousActions)>();

                // Iterate over the stateInfos and populate the actionDict
                for (int i = 0; i < stateInfos.Count; i++)
                {
                    Guid environmentId = stateInfos[i].environmentId;
                    (int[] discreteActions, float[] continuousActions) action = actions[i];
                    actionDict[environmentId] = action;
                }

                return ValueTask.FromResult(actionDict);
            }
        }
        public ValueTask UploadTransitionsAsync(IEnumerable<TransitionPortable<T>> transitions)
        {
            _agent.AddTransition(transitions);
            return ValueTask.CompletedTask;
        }
    }
}