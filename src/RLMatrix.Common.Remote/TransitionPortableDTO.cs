using MessagePack;
using System;

namespace RLMatrix.Common.Remote
{
    [MessagePackObject]
    public class TransitionPortableDTO
    {
        [Key(0)]
        public Guid Guid { get; set; }

        [Key(1)]
        public float[]? State1D { get; set; }

        [Key(2)]
        public float[,]? State2D { get; set; }

        [Key(3)]
        public int[] DiscreteActions { get; set; }

        [Key(4)]
        public float[] ContinuousActions { get; set; }

        [Key(5)]
        public float Reward { get; set; }

        [Key(6)]
        public Guid? NextTransitionGuid { get; set; }
    }
}

namespace RLMatrix.Common.Remote
{
    public static class TransitionPortableExtensions
    {
        public static TransitionPortableDTO ToDTO<TState>(this TransitionPortable<TState> transition)
        {
            if (typeof(TState) == typeof(float[]))
            {
                return new TransitionPortableDTO
                {
                    Guid = transition.guid,
                    State1D = (float[])(object)transition.state,
                    State2D = null,
                    DiscreteActions = transition.discreteActions,
                    ContinuousActions = transition.continuousActions,
                    Reward = transition.reward,
                    NextTransitionGuid = transition.nextTransitionGuid
                };
            }
            else if (typeof(TState) == typeof(float[,]))
            {
                return new TransitionPortableDTO
                {
                    Guid = transition.guid,
                    State1D = null,
                    State2D = (float[,])(object)transition.state,
                    DiscreteActions = transition.discreteActions,
                    ContinuousActions = transition.continuousActions,
                    Reward = transition.reward,
                    NextTransitionGuid = transition.nextTransitionGuid
                };
            }
            else
            {
                throw new ArgumentException("State must be either float[] or float[,]");
            }
        }

        public static TransitionPortable<TState> FromDTO<TState>(this TransitionPortableDTO dto)
        {
            if (typeof(TState) == typeof(float[]))
            {
                if (dto.State1D != null)
                {
                    return new TransitionPortable<TState>(
                        dto.Guid,
                        (TState)(object)dto.State1D,
                        dto.DiscreteActions,
                        dto.ContinuousActions,
                        dto.Reward,
                        dto.NextTransitionGuid);
                }
                else
                {
                    throw new InvalidCastException($"State is not of type {typeof(TState)}");
                }
            }
            else if (typeof(TState) == typeof(float[,]))
            {
                if (dto.State2D != null)
                {
                    return new TransitionPortable<TState>(
                        dto.Guid,
                        (TState)(object)dto.State2D,
                        dto.DiscreteActions,
                        dto.ContinuousActions,
                        dto.Reward,
                        dto.NextTransitionGuid);
                }
                else
                {
                    throw new InvalidCastException($"State is not of type {typeof(TState)}");
                }
            }
            else
            {
                throw new InvalidCastException($"Unsupported state type: {typeof(TState)}");
            }
        }

        public static List<TransitionPortableDTO> ToDTOList<TState>(this List<TransitionPortable<TState>> transitions)
        {
            return transitions.Select(transition => transition.ToDTO()).ToList();
        }

        public static List<TransitionPortable<TState>> FromDTOList<TState>(this List<TransitionPortableDTO> dtos)
        {
            return dtos.Select(dto => dto.FromDTO<TState>()).ToList();
        }
    }
}
