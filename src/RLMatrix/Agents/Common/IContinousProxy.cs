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
        /// <param name="isTraining">Indicates whether the agent is in training mode.</param>
        /// <returns>A tuple containing dictionaries mapping the environment ID to the selected discrete actions and continuous actions.</returns>
#if NET8_0_OR_GREATER
        ValueTask<Dictionary<Guid, (int[] discreteActions, float[] continuousActions)>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos, bool isTraining);
#else
        Task<Dictionary<Guid, (int[] discreteActions, float[] continuousActions)>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos, bool isTraining);
#endif

        /// <summary>
        /// Resets the internal states of the proxy for the specified environments.
        /// Only makes difference for recurrent models.
        /// </summary>
        /// <param name="environmentIds">A list of tuples containing the environment ID and a boolean indicating if the environment is done.</param>
#if NET8_0_OR_GREATER
        ValueTask ResetStates(List<(Guid environmentId, bool dones)> environmentIds);
#else
        Task ResetStates(List<(Guid environmentId, bool dones)> environmentIds);
#endif

        /// <summary>
        /// Uploads a batch of transitions to the proxy for training.
        /// </summary>
        /// <param name="transitions">An enumerable of transitions containing the state, discrete action, continuous action, reward, and next state.</param>
#if NET8_0_OR_GREATER
        ValueTask UploadTransitionsAsync(IEnumerable<TransitionPortable<T>> transitions);
#else
        Task UploadTransitionsAsync(IEnumerable<TransitionPortable<T>> transitions);
#endif

        /// <summary>
        /// Optimizes the underlying model based on the uploaded transitions.
        /// </summary>
#if NET8_0_OR_GREATER
        ValueTask OptimizeModelAsync();
#else
        Task OptimizeModelAsync();
#endif

        /// <summary>
        /// Saves the current state of the proxy to the specified path.
        /// </summary>
        /// <param name="path">The path where the proxy state should be saved.</param>
#if NET8_0_OR_GREATER
        ValueTask SaveAsync(string path);
#else
        Task SaveAsync(string path);
#endif

        /// <summary>
        /// Loads the proxy state from the specified path.
        /// </summary>
        /// <param name="path">The path from which the proxy state should be loaded.</param>
#if NET8_0_OR_GREATER
        ValueTask LoadAsync(string path);
#else
        Task LoadAsync(string path);
#endif
    }
}