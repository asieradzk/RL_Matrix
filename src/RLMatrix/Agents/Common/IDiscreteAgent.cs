using RLMatrix.Agents.DQN.Domain;
using RLMatrix.Memories;
using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;

namespace RLMatrix.Agents.Common
{
    /// <summary>
    /// Defines an interface for objects that have a memory.
    /// </summary>
    /// <typeparam name="T">The type of state stored in the memory.</typeparam>
    public interface IHasMemory<T>
    {
        /// <summary>
        /// Gets or sets the memory of the object.
        /// </summary>
        IMemory<T> Memory { get; set; }

        /// <summary>
        /// Adds transitions to the memory.
        /// </summary>
        /// <param name="transitions">The transitions to add.</param>
        void AddTransition(IEnumerable<TransitionPortable<T>> transitions);
    }

    /// <summary>
    /// Defines an interface for objects that can select actions.
    /// </summary>
    /// <typeparam name="T">The type of state used to select actions.</typeparam>
    public interface ISelectActions<T>
    {
        /// <summary>
        /// Selects actions based on the given states.
        /// </summary>
        /// <param name="states">The states to base the action selection on.</param>
        /// <param name="isTraining">Indicates whether the agent is in training mode.</param>
        /// <returns>An array of selected actions.</returns>
        int[][] SelectActions(T[] states, bool isTraining);
    }

    /// <summary>
    /// Defines an interface for objects that can select actions in a recurrent manner.
    /// </summary>
    /// <typeparam name="T">The type of state used to select actions.</typeparam>
    public interface ISelectActionsRecurrent<T>
    {
        /// <summary>
        /// Selects actions based on the given states and memory states in a recurrent manner.
        /// </summary>
        /// <param name="states">An array of tuples containing the state and memory states.</param>
        /// <param name="isTraining">Indicates whether the agent is in training mode.</param>
        /// <returns>An array of tuples containing selected actions and updated memory states.</returns>
#if NET8_0_OR_GREATER
        (int[] actions, Tensor? memoryState, Tensor? memoryState2)[] SelectActionsRecurrent((T state, Tensor? memoryState, Tensor? memoryState2)[] states, bool isTraining);
#else
        (int[] actions, Tensor memoryState, Tensor memoryState2)[] SelectActionsRecurrent((T state, Tensor memoryState, Tensor memoryState2)[] states, bool isTraining);
#endif
    }

    /// <summary>
    /// Defines an interface for objects that have an optimizer.
    /// </summary>
    /// <typeparam name="T">The type of state the optimizer operates on.</typeparam>
    public interface IHasOptimizer<T>
    {
        /// <summary>
        /// Gets the optimizer of the object.
        /// </summary>
#if NET8_0_OR_GREATER
        IOptimize<T> Optimizer { get; init; }
#else
        IOptimize<T> Optimizer { get; }
#endif
    }

    /// <summary>
    /// Defines an interface for objects that can be saved and loaded.
    /// </summary>
    public interface ISavable
    {
        /// <summary>
        /// Saves the object to the specified path.
        /// </summary>
        /// <param name="path">The path to save the object to.</param>
        void Save(string path);

        /// <summary>
        /// Loads the object from the specified path.
        /// </summary>
        /// <param name="path">The path to load the object from.</param>
        /// <param name="scheduler">Optional learning rate scheduler.</param>
        void Load(string path, LRScheduler scheduler = null);
    }

    /// <summary>
    /// Defines an interface for objects that can optimize based on a memory.
    /// </summary>
    /// <typeparam name="T">The type of state the optimizer operates on.</typeparam>
    public interface IOptimize<T>
    {
        /// <summary>
        /// Optimizes based on the given replay buffer.
        /// </summary>
        /// <param name="replayBuffer">The replay buffer to optimize from.</param>
        void Optimize(IMemory<T> replayBuffer);

        /// <summary>
        /// Updates the optimizers using the given scheduler.
        /// </summary>
        /// <param name="scheduler">The learning rate scheduler to use for updating.</param>
        void UpdateOptimizers(LRScheduler scheduler);
    }

    /// <summary>
    /// Defines an interface for the core of a discrete agent.
    /// </summary>
    /// <typeparam name="T">The type of state the agent operates on.</typeparam>
    public interface IDiscreteAgentCore<T>
    {
        /// <summary>
        /// Gets the sizes of the action space.
        /// </summary>
#if NET8_0_OR_GREATER
        int[] ActionSizes { get; init; }
#else
        int[] ActionSizes { get; }
#endif

        /// <summary>
        /// Selects actions based on the given states.
        /// </summary>
        /// <param name="states">The states to base the action selection on.</param>
        /// <param name="isTraining">Indicates whether the agent is in training mode.</param>
        /// <returns>An array of selected actions.</returns>
        int[][] SelectActions(T[] states, bool isTraining);

        /// <summary>
        /// Optimizes the agent's model.
        /// </summary>
        void OptimizeModel();
    }
}