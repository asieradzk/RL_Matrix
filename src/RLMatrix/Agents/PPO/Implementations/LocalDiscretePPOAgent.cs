using OneOf;
using RLMatrix.Agents.Common;
using Tensorboard;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

namespace RLMatrix.Agents.PPO.Implementations
{
    public class LocalDiscretePPOAgent<T> : IDiscreteProxy<T>
    {
        private readonly IDiscretePPOAgent<T> _agent;
        bool useRnn = false;
        private Dictionary<Guid, (Tensor?, Tensor?)?> memoriesStore = new();

        //TODO: Composer param
        public LocalDiscretePPOAgent(PPOAgentOptions options, int[] ActionSizes, OneOf<int, (int, int)> StateSizes /*, IDiscretePPOAgentCOmposer<T> agentComposer = null*/)
        {
            _agent = PPOAgentFactory<T>.ComposeDiscretePPOAgent(options, ActionSizes, StateSizes);
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

        public ValueTask<Dictionary<Guid, int[]>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos)
        {
            

            if (useRnn)
            {
                Dictionary<Guid, int[]> actionDict = new Dictionary<Guid, int[]>();
                if (memoriesStore.Count == 0)
                {
                    foreach (var stateInfo in stateInfos)
                    {
                        memoriesStore[stateInfo.environmentId] = null;
                    }
                }

                (T state, Tensor? memoryState, Tensor? memoryState2)[] statesWithMemory = stateInfos.Select(info =>
                {
                    var memoryTuple = memoriesStore[info.environmentId];
                    return (info.state, memoryTuple?.Item1, memoryTuple?.Item2);
                }).ToArray();

                (int[] actions, Tensor? memoryState, Tensor? memoryState2)[] actionsWithMemory = _agent.SelectActionsRecurrent(statesWithMemory, isTraining: true);

                for (int i = 0; i < stateInfos.Count; i++)
                {
                    Guid environmentId = stateInfos[i].environmentId;
                    int[] action = actionsWithMemory[i].actions;
                    memoriesStore[environmentId] = (actionsWithMemory[i].memoryState, actionsWithMemory[i].memoryState2);
                    actionDict[environmentId] = action;
                }

                return ValueTask.FromResult(actionDict);
            }else
            {

                // Extract the states from the stateInfos list
                T[] states = stateInfos.Select(info => info.state).ToArray();

                // Select actions for the batch of states
                int[][] actions = _agent.SelectActions(states, isTraining: true);

                // Create a dictionary to map environment IDs to their corresponding actions
                Dictionary<Guid, int[]> actionDict = new Dictionary<Guid, int[]>();

                // Iterate over the stateInfos and populate the actionDict
                for (int i = 0; i < stateInfos.Count; i++)
                {
                    Guid environmentId = stateInfos[i].environmentId;
                    int[] action = actions[i];
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


    

