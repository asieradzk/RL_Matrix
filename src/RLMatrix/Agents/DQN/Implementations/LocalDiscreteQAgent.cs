using OneOf;
using RLMatrix.Agents.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace RLMatrix
{
    /// <summary>
    /// Represents a local discrete Q-learning agent.
    /// </summary>
    /// <typeparam name="T">The type of the state.</typeparam>
    public class LocalDiscreteQAgent<T> : IDiscreteProxy<T>
    {
        private readonly ComposableQDiscreteAgent<T> _agent;

        /// <summary>
        /// Initializes a new instance of the <see cref="LocalDiscreteQAgent{T}"/> class.
        /// </summary>
        /// <param name="opts">The DQN agent options.</param>
        /// <param name="actionSizes">The sizes of the action space.</param>
        /// <param name="stateSizes">The sizes of the state space.</param>
        /// <param name="agentComposer">The optional agent composer.</param>
        public LocalDiscreteQAgent(DQNAgentOptions opts, int[] actionSizes, OneOf<int, (int, int)> stateSizes, 
          IDiscreteQAgentFactory<T> agentComposer = null)
        {
            _agent = agentComposer?.ComposeAgent(opts) ?? DiscreteQAgentFactory<T>.ComposeQAgent(opts, actionSizes, stateSizes);
        }

        /// <summary>
        /// Loads the agent's state asynchronously.
        /// </summary>
        /// <param name="path">The path to load the agent's state from.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
#if NET8_0_OR_GREATER
        public ValueTask LoadAsync(string path)
        {
            _agent.Load(path);
            return ValueTask.CompletedTask;
        }
#else
        public Task LoadAsync(string path)
        {
            _agent.Load(path);
            return Task.CompletedTask;
        }
#endif

        /// <summary>
        /// Optimizes the agent's model asynchronously.
        /// </summary>
        /// <returns>A task representing the asynchronous operation.</returns>
#if NET8_0_OR_GREATER
        public ValueTask OptimizeModelAsync()
        {
            _agent.OptimizeModel();
            return ValueTask.CompletedTask;
        }
#else
        public Task OptimizeModelAsync()
        {
            _agent.OptimizeModel();
            return Task.CompletedTask;
        }
#endif

        /// <summary>
        /// Resets the states of specified environments asynchronously.
        /// </summary>
        /// <param name="environmentIds">The list of environment IDs and their done states.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
#if NET8_0_OR_GREATER
        public ValueTask ResetStates(List<(Guid environmentId, bool dones)> environmentIds)
        {
            return ValueTask.CompletedTask;
        }
#else
        public Task ResetStates(List<(Guid environmentId, bool dones)> environmentIds)
        {
            return Task.CompletedTask;
        }
#endif

        /// <summary>
        /// Saves the agent's state asynchronously.
        /// </summary>
        /// <param name="path">The path to save the agent's state to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
#if NET8_0_OR_GREATER
        public ValueTask SaveAsync(string path)
        {
            _agent.Save(path);
            return ValueTask.CompletedTask;
        }
#else
        public Task SaveAsync(string path)
        {
            _agent.Save(path);
            return Task.CompletedTask;
        }
#endif

        /// <summary>
        /// Selects actions for a batch of states asynchronously.
        /// </summary>
        /// <param name="stateInfos">The list of state information.</param>
        /// <param name="isTraining">Indicates whether the agent is in training mode.</param>
        /// <returns>A dictionary of actions for each environment.</returns>
#if NET8_0_OR_GREATER
        public ValueTask<Dictionary<Guid, int[]>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos, bool isTraining)
        {
            T[] states = stateInfos.Select(info => info.state).ToArray();
            int[][] actions = _agent.SelectActions(states, isTraining);
            Dictionary<Guid, int[]> actionDict = new Dictionary<Guid, int[]>();
            for (int i = 0; i < stateInfos.Count; i++)
            {
                Guid environmentId = stateInfos[i].environmentId;
                int[] action = actions[i];
                actionDict[environmentId] = action;
            }
            return ValueTask.FromResult(actionDict);
        }
#else
        public Task<Dictionary<Guid, int[]>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos, bool isTraining)
        {
            T[] states = stateInfos.Select(info => info.state).ToArray();
            int[][] actions = _agent.SelectActions(states, isTraining);
            Dictionary<Guid, int[]> actionDict = new Dictionary<Guid, int[]>();
            for (int i = 0; i < stateInfos.Count; i++)
            {
                Guid environmentId = stateInfos[i].environmentId;
                int[] action = actions[i];
                actionDict[environmentId] = action;
            }
            return Task.FromResult(actionDict);
        }
#endif

        /// <summary>
        /// Uploads transitions to the agent asynchronously.
        /// </summary>
        /// <param name="transitions">The transitions to upload.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
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