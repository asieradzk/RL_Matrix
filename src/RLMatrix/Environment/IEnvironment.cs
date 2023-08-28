using OneOf;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLMatrix
{
    /// <summary>
    /// This interface defines the structure of the environment in which the reinforcement learning algorithm operates.
    /// </summary>
    /// <typeparam name="TState">The type that describes the state of the environment.</typeparam>
    public interface IEnvironment<TState>
    {
        /// <summary>
        /// Represents the number of steps that have been taken in the current episode.
        /// </summary>
        public int stepCounter { get; set; }

        /// <summary>
        /// Specifies the maximum number of steps that an episode can last.
        /// </summary>
        public int maxSteps { get; set; }

        /// <summary>
        /// Boolean indicator to denote whether the current episode has ended.
        /// </summary>
        public bool isDone { get; set; }

        /// <summary>
        /// Represents the dimensionality of the environment's state. It can either be a single integer (for 1D states) 
        /// or a tuple of two integers (for 2D states).
        /// </summary>
        public OneOf<int, (int, int)> stateSize { get; set; }

        /// <summary>
        /// Represents the dimensionality of the environment's action space.
        /// </summary>
        public int actionSize { get; set; }

        /// <summary>
        /// Initializes the environment for a new episode.
        /// </summary>
        public void Initialise();

        /// <summary>
        /// Returns the current state of the environment.
        /// </summary>
        /// <returns>Current state of the environment of type <typeparamref name="TState"/></returns>
        public TState GetCurrentState();

        /// <summary>
        /// Resets the environment to its initial state at the beginning of a new episode.
        /// </summary>
        public void Reset();

        /// <summary>
        /// Advances the state of the environment by one step based on the action provided.
        /// </summary>
        /// <param name="actionId">The action to be taken in the current state.</param>
        /// <returns>The reward associated with the taken action.</returns>
        public float Step(int actionId);
    }
}
