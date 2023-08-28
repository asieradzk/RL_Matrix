namespace RLMatrix
{
    /// <summary>
    /// The Transition record represents a single transition in reinforcement learning.
    /// It contains the current state, action taken, reward received, and the next state.
    /// </summary>
    [Serializable]
    public sealed record Transition<TState>(TState state, int action, float reward, TState nextState);

    /// <summary>
    /// TransitionListExtensions provides extension methods for a List of Transitions.
    /// </summary>
    public static class TransitionListExtensions
    {
        /// <summary>
        /// Transposes a list of transitions into four separate lists: 
        /// states, actions, rewards, and next states.
        /// </summary>
        /// <param name="transitions">The list of transitions to transpose.</param>
        /// <returns>A tuple containing four lists: states, actions, rewards, and next states.</returns>
        public static (List<TState>, List<int>, List<float>, List<TState>) Transpose<TState>(this List<Transition<TState>> transitions)
        {
            var stateList = transitions.Select(t => t.state).ToList();
            var actionList = transitions.Select(t => t.action).ToList();
            var rewardList = transitions.Select(t => t.reward).ToList();
            var nextStateList = transitions.Select(t => t.nextState).ToList();

            return (stateList, actionList, rewardList, nextStateList);
        }
    }
}
