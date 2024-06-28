using RLMatrix.Agents.DQN.Domain;
using RLMatrix.Memories;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;

namespace RLMatrix.Agents.Common
{
    /// <summary>
    /// Represents a continuous agent with both discrete and continuous action capabilities.
    /// </summary>
    /// <typeparam name="T">The type of state the agent operates on.</typeparam>
    public interface IContinuousAgent<T> : IHasMemory<T>, ISelectContinuousAndDiscreteActions<T>, IHasOptimizer<T>, ISavable
    {
        /// <summary>
        /// Gets the dimensions of the discrete action space.
        /// </summary>
#if NET8_0_OR_GREATER
        int[] DiscreteDimensions { get; init; }
#else
        int[] DiscreteDimensions { get; set; }
#endif

        /// <summary>
        /// Gets the bounds for continuous actions.
        /// </summary>
#if NET8_0_OR_GREATER
        (float min, float max)[] ContinuousActionBounds { get; init; }
#else
        (float min, float max)[] ContinuousActionBounds { get; set; }
#endif

        /// <summary>
        /// Optimizes the agent's model.
        /// </summary>
        void OptimizeModel();
    }

    /// <summary>
    /// Defines methods for selecting both continuous and discrete actions.
    /// </summary>
    /// <typeparam name="T">The type of state the agent operates on.</typeparam>
    public interface ISelectContinuousAndDiscreteActions<T>
    {
        /// <summary>
        /// Selects actions based on the given states.
        /// </summary>
        /// <param name="states">The states to base the action selection on.</param>
        /// <param name="isTraining">Indicates whether the agent is in training mode.</param>
        /// <returns>An array of tuples containing discrete and continuous actions.</returns>
        (int[] discreteActions, float[] continuousActions)[] SelectActions(T[] states, bool isTraining);
    }

    /// <summary>
    /// Defines methods for selecting both continuous and discrete actions in a recurrent manner.
    /// </summary>
    /// <typeparam name="T">The type of state the agent operates on.</typeparam>
    public interface ISelectContinuousAndDiscreteActionsRecurrent<T>
    {
        /// <summary>
        /// Selects actions based on the given states and memory states in a recurrent manner.
        /// </summary>
        /// <param name="states">An array of tuples containing the state and memory states.</param>
        /// <param name="isTraining">Indicates whether the agent is in training mode.</param>
        /// <returns>An array of tuples containing actions and updated memory states.</returns>
#if NET8_0_OR_GREATER
        ((int[] discreteActions, float[] continuousActions) actions, Tensor? memoryState, Tensor? memoryState2)[] SelectActionsRecurrent((T state, Tensor? memoryState, Tensor? memoryState2)[] states, bool isTraining);
#else
        ((int[] discreteActions, float[] continuousActions) actions, Tensor memoryState, Tensor memoryState2)[] SelectActionsRecurrent((T state, Tensor memoryState, Tensor memoryState2)[] states, bool isTraining);
#endif
    }
}