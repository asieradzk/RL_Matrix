using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace RLMatrix.Agents.Common
{
    public interface IContinuousProxy<T>
    {
        /// <summary>
        /// Selects discrete and continuous actions for a batch of environments based on their current states.
        /// </summary>
        /// <param name="stateInfos">A list of tuples containing the environment ID and the current state.</param>
        /// <returns>A tuple containing dictionaries mapping the environment ID to the selected discrete actions and continuous actions.</returns>
        public ValueTask<Dictionary<Guid, (int[] discreteActions, float[] continuousActions)>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos, bool isTraining);

        /// <summary>
        /// Resets the internal states of the proxy for the specified environments.
        /// Only makes difference for recurrent models.
        /// </summary>
        /// <param name="environmentIds">A list of tuples containing the environment ID and a boolean indicating if the environment is done.</param>
        ValueTask ResetStates(List<(Guid environmentId, bool dones)> environmentIds);

        /// <summary>
        /// Uploads a batch of transitions to the proxy for training.
        /// </summary>
        /// <param name="transitions">An enumerable of transitions containing the state, discrete action, continuous action, reward, and next state.</param>
        ValueTask UploadTransitionsAsync(IEnumerable<TransitionPortable<T>> transitions);

        /// <summary>
        /// Optimizes the underlying model based on the uploaded transitions.
        /// </summary>
        ValueTask OptimizeModelAsync();

        /// <summary>
        /// Saves the current state of the proxy to the specified path.
        /// </summary>
        /// <param name="path">The path where the proxy state should be saved.</param>
        ValueTask SaveAsync(string path);

        /// <summary>
        /// Loads the proxy state from the specified path.
        /// </summary>
        /// <param name="path">The path from which the proxy state should be loaded.</param>
        ValueTask LoadAsync(string path);
    }
}