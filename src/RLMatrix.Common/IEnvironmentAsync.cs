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
    public interface IEnvironmentAsync<TState>
    {
        /// <summary>
        /// Represents the dimensionality of the environment's state. It can either be a single integer (for 1D states) 
        /// or a tuple of two integers (for 2D states).
        /// </summary>
        public OneOf<int, (int, int)> stateSize { get; set; }

        /// <summary>
        /// Represents the dimensionality of the environment's action space.
        /// For instance if action size is set to 4, the network will learn to output an integer between 0 and 3.
        /// The length of array is number of possible actions so if you input [2, 4] the agent will output int[2] where first int is between 0 and 1 and second between 0 and 3.
        /// </summary>
        public int[] actionSize { get; set; }


        /// <summary>
        /// Returns the current state of the environment.
        /// </summary>
        /// <returns>Current state of the environment of type <typeparamref name="TState"/></returns>
        public Task<TState> GetCurrentState();

        /// <summary>
        /// Resets the environment to its initial state at the beginning of a new episode.
        /// </summary>
        public Task Reset();

        /// <summary>
        /// Advances the state of the environment by one step based on the action provided.
        /// </summary>
        /// <param name="actionsIds">The action to be taken in the current state.</param>
        /// <returns>The reward associated with the taken action. Bool trajectory done</returns>
        public Task<(float, bool)> Step(int[] actionsIds);
    }
}
