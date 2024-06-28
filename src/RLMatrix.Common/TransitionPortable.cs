using System;
using System.Collections.Generic;
using System.Linq;

namespace RLMatrix
{
#if NET8_0_OR_GREATER
    public sealed record TransitionPortable<TState>(Guid guid, TState state, int[] discreteActions, float[] continuousActions, float reward, Guid? nextTransitionGuid);
#elif NETSTANDARD2_0
    public sealed class TransitionPortable<TState>
    {
        public Guid guid { get; }
        public TState state { get; }
        public int[] discreteActions { get; }
        public float[] continuousActions { get; }
        public float reward { get; }
        public Guid? nextTransitionGuid { get; }

        public TransitionPortable(Guid guid, TState state, int[] discreteActions, float[] continuousActions, float reward, Guid? nextTransitionGuid)
        {
            this.guid = guid;
            this.state = state;
            this.discreteActions = discreteActions;
            this.continuousActions = continuousActions;
            this.reward = reward;
            this.nextTransitionGuid = nextTransitionGuid;
        }

        public TransitionPortable<TState> With(Guid? guid = null, TState state = default(TState), int[] discreteActions = null, float[] continuousActions = null, float? reward = null, Guid? nextTransitionGuid = null)
        {
            return new TransitionPortable<TState>(
                guid ?? this.guid,
                state.Equals(default(TState)) ? this.state : state,
                discreteActions ?? this.discreteActions,
                continuousActions ?? this.continuousActions,
                reward ?? this.reward,
                nextTransitionGuid ?? this.nextTransitionGuid);
        }

        public override bool Equals(object obj)
        {
            if (obj is TransitionPortable<TState> other)
            {
                return guid == other.guid &&
                       EqualityComparer<TState>.Default.Equals(state, other.state) &&
                       discreteActions.SequenceEqual(other.discreteActions) &&
                       continuousActions.SequenceEqual(other.continuousActions) &&
                       reward == other.reward &&
                       nextTransitionGuid == other.nextTransitionGuid;
            }
            return false;
        }

        public override int GetHashCode()
        {
            int hash = 17;
            hash = hash * 31 + guid.GetHashCode();
            hash = hash * 31 + (state != null ? state.GetHashCode() : 0);
            hash = hash * 31 + (discreteActions != null ? discreteActions.GetHashCode() : 0);
            hash = hash * 31 + (continuousActions != null ? continuousActions.GetHashCode() : 0);
            hash = hash * 31 + reward.GetHashCode();
            hash = hash * 31 + (nextTransitionGuid != null ? nextTransitionGuid.GetHashCode() : 0);
            return hash;
        }
    }
#endif
}