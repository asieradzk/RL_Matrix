using System;
using System.Collections.Generic;
using System.Linq;

namespace RLMatrix
{
#if NET8_0_OR_GREATER
    public sealed record TransitionPortable<TState>(Guid Guid, TState state, int[] discreteActions, float[] continuousActions, float reward, Guid? NextTransitionGuid);
#elif NETSTANDARD2_0
    public sealed class TransitionPortable<TState>
    {
        public Guid Guid { get; }
        public TState State { get; }
        public int[] DiscreteActions { get; }
        public float[] ContinuousActions { get; }
        public float Reward { get; }
        public Guid? NextTransitionGuid { get; }

        public TransitionPortable(Guid guid, TState state, int[] discreteActions, float[] continuousActions, float reward, Guid? nextTransitionGuid)
        {
            Guid = guid;
            State = state;
            DiscreteActions = discreteActions;
            ContinuousActions = continuousActions;
            Reward = reward;
            NextTransitionGuid = nextTransitionGuid;
        }

        public TransitionPortable<TState> With(Guid? guid = null, TState state = default(TState), int[] discreteActions = null, float[] continuousActions = null, float? reward = null, Guid? nextTransitionGuid = null)
        {
            return new TransitionPortable<TState>(
                guid ?? this.Guid,
                state.Equals(default(TState)) ? this.State : state,
                discreteActions ?? this.DiscreteActions,
                continuousActions ?? this.ContinuousActions,
                reward ?? this.Reward,
                nextTransitionGuid ?? this.NextTransitionGuid);
        }

        public override bool Equals(object obj)
        {
            if (obj is TransitionPortable<TState> other)
            {
                return Guid == other.Guid &&
                       EqualityComparer<TState>.Default.Equals(State, other.State) &&
                       DiscreteActions.SequenceEqual(other.DiscreteActions) &&
                       ContinuousActions.SequenceEqual(other.ContinuousActions) &&
                       Reward == other.Reward &&
                       NextTransitionGuid == other.NextTransitionGuid;
            }
            return false;
        }

        public override int GetHashCode()
        {
            int hash = 17;
            hash = hash * 31 + Guid.GetHashCode();
            hash = hash * 31 + (State != null ? State.GetHashCode() : 0);
            hash = hash * 31 + (DiscreteActions != null ? DiscreteActions.GetHashCode() : 0);
            hash = hash * 31 + (ContinuousActions != null ? ContinuousActions.GetHashCode() : 0);
            hash = hash * 31 + Reward.GetHashCode();
            hash = hash * 31 + (NextTransitionGuid != null ? NextTransitionGuid.GetHashCode() : 0);
            return hash;
        }
    }
#endif
}
