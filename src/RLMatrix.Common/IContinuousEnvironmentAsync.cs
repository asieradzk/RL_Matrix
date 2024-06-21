using OneOf;
using System.Threading.Tasks;

namespace RLMatrix
{
    /// <summary>
    /// This interface defines the structure of an environment that supports both discrete and continuous actions.
    /// </summary>
    /// <typeparam name="TState">The type that describes the state of the environment.</typeparam>
    public interface IContinuousEnvironmentAsync<TState>
    {
        /// <summary>
        /// Represents the dimensionality of the environment's state. It can either be a single integer (for 1D states) 
        /// or a tuple of two integers (for 2D states).
        /// </summary>
        OneOf<int, (int, int)> StateSize { get; set; }

        /// <summary>
        /// Represents the dimensionality of the environment's discrete action space.
        /// For instance, if action size is set to 4, the network will learn to output an integer between 0 and 3.
        /// The length of the array is the number of possible actions, so if you input [2, 4], the agent will output int[2] where the first int is between 0 and 1, and the second is between 0 and 3.
        /// </summary>
        int[] DiscreteActionSize { get; set; }

        /// <summary>
        /// Represents the dimensionality of the environment's continuous action space.
        /// Each element indicates the range for each continuous action.
        /// For instance, if a continuous action can range between -1 and 1, the tuple will be (-1, 1).
        /// </summary>
        (float min, float max)[] ContinuousActionBounds { get; set; }

        /// <summary>
        /// Returns the current state of the environment.
        /// </summary>
        /// <returns>Current state of the environment of type <typeparamref name="TState"/></returns>
        Task<TState> GetCurrentState();

        /// <summary>
        /// Resets the environment to its initial state at the beginning of a new episode.
        /// </summary>
        Task Reset();

        /// <summary>
        /// Advances the state of the environment by one step based on the discrete and continuous actions provided.
        /// </summary>
        /// <param name="discreteActions">The discrete actions to be taken in the current state.</param>
        /// <param name="continuousActions">The continuous actions to be taken in the current state.</param>
        /// <returns>The reward associated with the taken actions. Bool trajectory done</returns>
        Task<(float reward, bool done)> Step(int[] discreteActions, float[] continuousActions);
    }
}