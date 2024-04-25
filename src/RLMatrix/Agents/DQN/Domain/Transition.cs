using System;
using System.Collections.Generic;
using System.Linq;

namespace RLMatrix
{
    /// <summary>
    /// The Transition record represents a single transition in reinforcement learning.
    /// It contains the current state, discrete and continuous actions taken, reward received, and the next state.
    /// </summary>
    public sealed record TransitionInMemory<TState>(TState state, int[] discreteActions, float[] continuousActions, float reward, TState? nextState, TransitionInMemory<TState>? nextTransition);

    public sealed record TransitionPortable<TState>(Guid Guid, TState state, int[] discreteActions, float[] continuousActions, float reward, Guid? NextTransitionGuid);

    public static class TransitionExtensions
    {
        public static IEnumerable<TransitionInMemory<TState>> ToTransitionInMemory<TState>(this IEnumerable<TransitionPortable<TState>> portableTransitions)
        {
            var transitionMap = new Dictionary<Guid, TransitionInMemory<TState>>();

            foreach (var portableTransition in portableTransitions)
            {
                transitionMap[portableTransition.Guid] = new TransitionInMemory<TState>(
                    portableTransition.state,
                    portableTransition.discreteActions,
                    portableTransition.continuousActions,
                    portableTransition.reward,
                    default,  // nextState initialized to null or default
                    null      // nextTransition initialized to null
                );
            }

            foreach (var portableTransition in portableTransitions)
            {
                if (portableTransition.NextTransitionGuid.HasValue &&
                    transitionMap.TryGetValue(portableTransition.NextTransitionGuid.Value, out var nextTransition))
                {
                    transitionMap[portableTransition.Guid] = transitionMap[portableTransition.Guid] with
                    {
                        nextState = nextTransition.state,
                        nextTransition = nextTransition
                    };
                }
            }

            return transitionMap.Values;
        }

    }
}
