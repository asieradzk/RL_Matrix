using OneOf;
using RLMatrix.Agents.Common;

namespace RLMatrix
{
    //misleading name? since this is DQN
    public class LocalDiscreteQAgent<T> : IDiscreteProxy<T>
    {
        private readonly ComposableQDiscreteAgent<T> _agent;

        public LocalDiscreteQAgent(DQNAgentOptions opts, int[] actionSizes, OneOf<int, (int, int)> stateSizes, IDiscreteQAgentFactory<T> agentComposer = null)
        {
            //chekd if null and create default
            _agent = agentComposer?.ComposeAgent(opts) ?? DiscreteQAgentFactory<T>.ComposeQAgent(opts, actionSizes, stateSizes);
        }


        public ValueTask OptimizeModelAsync()
        {
            _agent.OptimizeModel();
            return ValueTask.CompletedTask;
        }

        public ValueTask<Dictionary<Guid, int[]>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos)
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

        public ValueTask UploadTransitionsAsync(IEnumerable<TransitionPortable<T>> transitions)
        {
            _agent.AddTransition(transitions);
            return ValueTask.CompletedTask;
        }
    }
}
