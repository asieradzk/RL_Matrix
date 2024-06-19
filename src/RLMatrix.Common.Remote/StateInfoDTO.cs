using MessagePack;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RLMatrix.Common.Remote
{
    [MessagePackObject]
    public class StateInfoDTO
    {
        [Key(0)]
        public Guid EnvironmentId { get; set; }

        [Key(1)]
        public float[]? State1D { get; set; }

        [Key(2)]
        public float[,]? State2D { get; set; }
    }

    public static class StateInfoDTOExtensions
    {
        public static StateInfoDTO Pack<TState>(Guid environmentId, TState state)
        {
            if (state is float[] state1D)
            {
                return new StateInfoDTO
                {
                    EnvironmentId = environmentId,
                    State1D = state1D,
                    State2D = null
                };
            }
            else if (state is float[,] state2D)
            {
                return new StateInfoDTO
                {
                    EnvironmentId = environmentId,
                    State1D = null,
                    State2D = state2D
                };
            }
            else
            {
                throw new ArgumentException("State must be either float[] or float[,]");
            }
        }

        public static (Guid, TState) Unpack<TState>(this StateInfoDTO dto)
        {
            if (typeof(TState) == typeof(float[]))
            {
                if (dto.State1D != null)
                {
                    return (dto.EnvironmentId, (TState)(object)dto.State1D);
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
                    return (dto.EnvironmentId, (TState)(object)dto.State2D);
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

        public static List<StateInfoDTO> PackList<TState>(this List<(Guid, TState)> stateInfos)
        {
            return stateInfos.Select(info => Pack(info.Item1, info.Item2)).ToList();
        }

        public static List<(Guid, TState)> UnpackList<TState>(this List<StateInfoDTO> dtos)
        {
            return dtos.Select(dto => dto.Unpack<TState>()).ToList();
        }
    }
}
