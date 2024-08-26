using System;
using System.Collections.Generic;
using System.Linq;
using System.Collections;
using System.Runtime.CompilerServices;
using IEnumerableUnpacker;
using RLMatrix.Agents.Common;
using RLMatrix;

namespace RLMatrix.Agents.Common
{
    /// <summary>
    /// The Transition record represents a single transition in reinforcement learning.
    /// It contains the current state, discrete and continuous actions taken, reward received, and the next state.
    /// </summary>
    [Unpackable]
    public sealed class TransitionInMemory<TState> : IEquatable<TransitionInMemory<TState>>
    {
        [Unpack("batchStates")]
        public TState state;
        [Unpack("batchDiscreteActions")]
        public int[] discreteActions;
        [Unpack("batchContinuousActions")]
        public float[] continuousActions;
        public float reward;
        public TState? nextState;
        public TransitionInMemory<TState>? nextTransition;
        public TransitionInMemory<TState>? previousTransition;

        public TransitionInMemory(TState state, int[] discreteActions, float[] continuousActions, float reward, TState? nextState, TransitionInMemory<TState>? nextTransition, TransitionInMemory<TState>? previousTransition)
        {
            this.state = state;
            this.discreteActions = discreteActions;
            this.continuousActions = continuousActions;
            this.reward = reward;
            this.nextState = nextState;
            this.nextTransition = nextTransition;
            this.previousTransition = previousTransition;
        }

        public override bool Equals(object obj)
        {
            return Equals(obj as TransitionInMemory<TState>);
        }

        public bool Equals(TransitionInMemory<TState>? other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;

            return EqualityComparer<TState>.Default.Equals(state, other.state)
                && discreteActions.SequenceEqual(other.discreteActions)
                && continuousActions.SequenceEqual(other.continuousActions)
                && reward.Equals(other.reward)
                && EqualityComparer<TState>.Default.Equals(nextState, other.nextState);
        }

        public override int GetHashCode()
        {
#if NET8_0_OR_GREATER
            var hash = new HashCode();
            hash.Add(state);
            hash.Add(discreteActions);
            hash.Add(continuousActions);
            hash.Add(reward);
            hash.Add(nextState);
            return hash.ToHashCode();
#else
           unchecked
           {
               int hash = 17;
               hash = hash * 23 + EqualityComparer<TState>.Default.GetHashCode(state);
               hash = hash * 23 + (discreteActions != null ? discreteActions.GetHashCode() : 0);
               hash = hash * 23 + (continuousActions != null ? continuousActions.GetHashCode() : 0);
               hash = hash * 23 + reward.GetHashCode();
               hash = hash * 23 + (nextState != null ? EqualityComparer<TState>.Default.GetHashCode(nextState) : 0);
               return hash;
           }
#endif
        }
    }
}

public static class TransitionExtensions
{
    //TODO: Not optimised for multi episode batches
    //TODO: This can be multi-threaded optimised
    /// <summary>
    /// Converts a collection of portable transitions to a list of in-memory transitions, preserving episode boundaries.
    /// </summary>
    /// <typeparam name="TState">The type of the state in the transitions.</typeparam>
    /// <param name="portableTransitions">The collection of portable transitions to convert.</param>
    /// <returns>A list of in-memory transitions, with episode boundaries preserved.</returns>
    public static IList<TransitionInMemory<TState>> ToTransitionInMemory<TState>(this IEnumerable<TransitionPortable<TState>> portableTransitions)
    {
        var transitionMap = new Dictionary<Guid, TransitionInMemory<TState>>();

        // Create TransitionInMemory objects
        foreach (var portableTransition in portableTransitions)
        {
            var transition = new TransitionInMemory<TState>(
                portableTransition.state,
                portableTransition.discreteActions,
                portableTransition.continuousActions,
                portableTransition.reward,
                default,
                null,
                null
            );
            transitionMap[portableTransition.guid] = transition;
        }

        // Link transitions and set next states
        foreach (var portableTransition in portableTransitions)
        {
            var transition = transitionMap[portableTransition.guid];
            if (portableTransition.nextTransitionGuid.HasValue)
            {
                var nextTransition = transitionMap[portableTransition.nextTransitionGuid.Value];
                transition.nextState = nextTransition.state;
                transition.nextTransition = nextTransition;
                nextTransition.previousTransition = transition;
            }
        }

        // Find all first transitions (start of episodes)
        var firstTransitions = transitionMap.Values.Where(t => t.previousTransition == null).ToList();

        var transitions = new List<TransitionInMemory<TState>>();

        // Process each episode
        foreach (var firstTransition in firstTransitions)
        {
            var currentTransition = firstTransition;
            while (currentTransition != null)
            {
                transitions.Add(currentTransition);
                currentTransition = currentTransition.nextTransition;
            }
        }

        return transitions;
    }
}