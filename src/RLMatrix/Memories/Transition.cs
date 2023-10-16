using System;
using System.Collections.Generic;
using System.Linq;

namespace RLMatrix
{
    /// <summary>
    /// The Transition record represents a single transition in reinforcement learning.
    /// It contains the current state, discrete and continuous actions taken, reward received, and the next state.
    /// </summary>
    [Serializable]
    public sealed record Transition<TState>(TState state, int[] discreteActions, float[] continuousActions, float reward, TState nextState);

    /// <summary>
    /// TransitionListExtensions provides extension methods for a List of Transitions.
    /// </summary>
    public static class TransitionListExtensions
    {
        /// <summary>
        /// Transposes a list of transitions into separate lists: 
        /// states, discrete actions, continuous actions, rewards, and next states.
        /// </summary>
        /// <param name="transitions">The list of transitions to transpose.</param>
        /// <returns>A tuple containing the separate lists.</returns>
        public static (List<TState>, List<int[]>, List<float[]>, List<float>, List<TState>) Transpose<TState>(this List<Transition<TState>> transitions)
        {
            var stateList = transitions.Select(t => t.state).ToList();
            var discreteActionList = transitions.Select(t => t.discreteActions).ToList();
            var continuousActionList = transitions.Select(t => t.continuousActions).ToList();
            var rewardList = transitions.Select(t => t.reward).ToList();
            var nextStateList = transitions.Select(t => t.nextState).ToList();

            return (stateList, discreteActionList, continuousActionList, rewardList, nextStateList);
        }
    }
}
