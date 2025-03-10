using MessagePack;
using RLMatrix.Common;

namespace RLMatrix.Remote;

[MessagePackObject]
public class StateInfoDTO(Guid environmentId, float[]? state1D, float[,]? state2D)
{
    [Key(0)]
    public Guid EnvironmentId { get; set; } = environmentId;

    [Key(1)]
    public float[]? State1D { get; set; } = state1D;

    [Key(2)]
    public float[,]? State2D { get; set; } = state2D;

    internal static StateInfoDTO Create1D(Guid environmentId, float[] state)
        => new(environmentId, state, null);

    internal static StateInfoDTO Create2D(Guid environmentId, float[,] state)
        => new(environmentId, null, state);
}

public static class StateInfoDTOExtensions
{
    public static StateInfoDTO ToDTO<TState>(this EnvironmentState<TState> state)
        where TState : notnull
    {
        return state.State switch
        {
            float[] array1D => StateInfoDTO.Create1D(state.EnvironmentId, array1D),
            float[,] array2D => StateInfoDTO.Create2D(state.EnvironmentId, array2D),
            _ => throw new ArgumentException("State must be either float[] or float[,]")
        };
    }

    public static EnvironmentState<TState> FromDTO<TState>(this StateInfoDTO dto)
        where TState : notnull
    {
        if (typeof(TState) == typeof(float[]))
        {
            if (dto.State1D is null)
                throw new InvalidCastException($"State is not of type {typeof(TState)}");
            
            return new EnvironmentState<TState>(dto.EnvironmentId, (TState)(object)dto.State1D);
        }
        
        if (typeof(TState) == typeof(float[,]))
        {
            if (dto.State2D is null)
                throw new InvalidCastException($"State is not of type {typeof(TState)}");

            return new EnvironmentState<TState>(dto.EnvironmentId, (TState)(object)dto.State2D);
        }
        
        throw new InvalidCastException($"Unsupported state type: {typeof(TState)}");
    }

    public static List<StateInfoDTO> ToDTOList<TState>(this List<EnvironmentState<TState>> states)
        where TState : notnull
    {
        return states.Select(state => state.ToDTO()).ToList();
    }

    public static List<EnvironmentState<TState>> FromDTOList<TState>(this List<StateInfoDTO> dtos)
        where TState : notnull
    {
        return dtos.Select(dto => dto.FromDTO<TState>()).ToList();
    }
}