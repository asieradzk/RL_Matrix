using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Collections;
using System.Runtime.CompilerServices;
using IEnumerableUnpacker;

namespace RLMatrix.Agents.Common
{
    public static class TransitionInMemoryExtensions2
    {
        private static unsafe void ProcessTransition<TState>(ParallelState<TState> state, int i)
        {
            var transition = state.Transitions[i];
            state.PBatchStates[i] = transition.state;

            float* pContinuousDest = state.PBatchContinuousActions + i * state.FixedContinuousActionSize;
            int* pDiscreteDest = state.PBatchDiscreteActions + i * state.FixedDiscreteActionSize;

            fixed (float* pContinuousActionsSrc = transition.continuousActions)
            fixed (int* pDiscreteActionsSrc = transition.discreteActions)
            {
                // Copy continuous actions
                int continuousActionSize = state.FixedContinuousActionSize;
                int continuousActionsByteSize = continuousActionSize * sizeof(float);
                Unsafe.CopyBlockUnaligned(pContinuousDest, pContinuousActionsSrc, (uint)continuousActionsByteSize);

                // Copy discrete actions
                int discreteActionSize = state.FixedDiscreteActionSize;
                int discreteActionsByteSize = discreteActionSize * sizeof(int);
                Unsafe.CopyBlockUnaligned(pDiscreteDest, pDiscreteActionsSrc, (uint)discreteActionsByteSize);
            }
        }
        private unsafe struct ParallelState<TState>
        {
            public TransitionInMemory<TState>[] Transitions;
            public TState* PBatchStates;
            public float* PBatchContinuousActions;
            public int* PBatchDiscreteActions;
            public int FixedContinuousActionSize;
            public int FixedDiscreteActionSize;
        }
    }


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
            var hash = new HashCode();
            hash.Add(state);
            hash.Add(discreteActions);
            hash.Add(continuousActions);
            hash.Add(reward);
            hash.Add(nextState);
            return hash.ToHashCode();
        }
    }
    public sealed record TransitionPortable<TState>(Guid Guid, TState state, int[] discreteActions, float[] continuousActions, float reward, Guid? NextTransitionGuid);

    public static class TransitionExtensions
    {
        public static IList<TransitionInMemory<TState>> ToTransitionInMemory<TState>(this IEnumerable<TransitionPortable<TState>> portableTransitions)
        {
            var transitionMap = new Dictionary<Guid, TransitionInMemory<TState>>();

            // Create TransitionInMemory objects and populate the transitionMap
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

                transitionMap[portableTransition.Guid] = transition;
            }

            // Set the nextState, nextTransition, and previousTransition references
            foreach (var portableTransition in portableTransitions)
            {
                var transition = transitionMap[portableTransition.Guid];

                if (portableTransition.NextTransitionGuid.HasValue)
                {
                    var nextTransition = transitionMap[portableTransition.NextTransitionGuid.Value];
                    transition.nextState = nextTransition.state;
                    transition.nextTransition = nextTransition;
                    nextTransition.previousTransition = transition;
                }
            }

            // Find the first transition (the one without a previous transition)
            var firstTransition = transitionMap.Values.FirstOrDefault(t => t.previousTransition == null);

            // Create a list to store the transitions
            var transitions = new List<TransitionInMemory<TState>>();

            // Traverse the doubly linked list starting from the first transition
            var currentTransition = firstTransition;
            while (currentTransition != null)
            {
                transitions.Add(currentTransition);
                currentTransition = currentTransition.nextTransition;
            }

            return transitions;
        }

    }
}