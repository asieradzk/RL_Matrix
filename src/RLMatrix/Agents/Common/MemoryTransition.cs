using IEnumerableUnpacker;
using RLMatrix.Common;

// DO NOT SWITCH TO FILE-SCOPED NAMESPACES HERE. THIS BREAKS IENUMERABLEUNPACKER!
namespace RLMatrix
{
    [Unpackable]
    public sealed class MemoryTransition<TState>(TState state, RLActions actions, float reward, TState? nextState, 
        MemoryTransition<TState>? nextTransition, MemoryTransition<TState>? previousTransition) : IEquatable<MemoryTransition<TState>>
        where TState : notnull
    {
        [Unpack("batchStates")] internal readonly TState _state = state;
        [Unpack("batchDiscreteActions")] internal readonly int[] _discreteActions = actions.DiscreteActions;
        [Unpack("batchContinuousActions")] internal readonly float[] _continuousActions = actions.ContinuousActions;

        public TState State { get; } = state;

        public RLActions Actions { get; } = actions;

        public float Reward { get; } = reward;

        public TState? NextState { get; } = nextState;
        
        public MemoryTransition<TState>? NextTransition { get; internal set; } = nextTransition;

        public MemoryTransition<TState>? PreviousTransition { get; } = previousTransition;

        public bool Equals(MemoryTransition<TState>? other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;

            return EqualityComparer<TState>.Default.Equals(State, other.State)
                   && Actions.DiscreteActions.SequenceEqual(other.Actions.DiscreteActions)
                   && Actions.ContinuousActions.SequenceEqual(other.Actions.ContinuousActions)
                   && Reward.Equals(other.Reward)
                   && NextState != null && other.NextState != null
                   && EqualityComparer<TState>.Default.Equals(NextState, other.NextState);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                var hashCode = EqualityComparer<TState>.Default.GetHashCode(State);
                hashCode = (hashCode * 397) ^ Actions.ContinuousActions.GetHashCode();
                hashCode = (hashCode * 397) ^ Actions.ContinuousActions.GetHashCode();
                hashCode = (hashCode * 397) ^ Reward.GetHashCode();
                hashCode = (hashCode * 397) ^ EqualityComparer<TState?>.Default.GetHashCode(NextState ?? default!);
                return hashCode;
            }
        }

        public static implicit operator MemoryTransition<TState>(Transition<TState> transition)
            => new(transition.State, transition.Actions, transition.Reward, default, null, null);
    }
}