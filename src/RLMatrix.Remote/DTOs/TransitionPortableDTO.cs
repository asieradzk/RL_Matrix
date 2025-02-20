using MessagePack;
using RLMatrix.Common;

namespace RLMatrix.Remote;

[MessagePackObject]
public class TransitionPortableDTO(
    Guid guid,
    float[]? state1D,
    float[,]? state2D,
    int[] discreteActions,
    float[] continuousActions,
    float reward,
    Guid? nextTransitionGuid)
{

    [Key(0)]
    public Guid Guid { get; set; } = guid;

    [Key(1)]
    public float[]? State1D { get; set; } = state1D;

    [Key(2)]
    public float[,]? State2D { get; set; } = state2D;

    [Key(3)]
    public int[] DiscreteActions { get; set; } = discreteActions;

    [Key(4)]
    public float[] ContinuousActions { get; set; } = continuousActions;

    [Key(5)]
    public float Reward { get; set; } = reward;

    [Key(6)]
    public Guid? NextTransitionGuid { get; set; } = nextTransitionGuid;
}

public static class TransitionPortableExtensions
{
    public static TransitionPortableDTO ToDTO<TState>(this Transition<TState> transition)
        where TState : notnull
    {
        return transition.State switch
        {
            float[] array1D => new TransitionPortableDTO(transition.Id, array1D, null, transition.Actions.DiscreteActions,
                transition.Actions.ContinuousActions, transition.Reward, transition.NextTransitionId),
            float[,] array2D => new TransitionPortableDTO(transition.Id, null, array2D, transition.Actions.DiscreteActions,
                transition.Actions.ContinuousActions, transition.Reward, transition.NextTransitionId),
            _ => throw new ArgumentException("State must be either float[] or float[,]")
        };
    }

    public static Transition<TState> FromDTO<TState>(this TransitionPortableDTO dto)
        where TState : notnull
    {
        if (typeof(TState) == typeof(float[]))
        {
            if (dto.State1D is null)
                throw new InvalidCastException($"State is not of type {typeof(TState)}");

            return new Transition<TState>(dto.Guid, 
                (TState)(object)dto.State1D, 
                RLActions.Interpret(dto.DiscreteActions, dto.ContinuousActions),
                dto.Reward, 
                dto.NextTransitionGuid);
        }
            
        if (typeof(TState) == typeof(float[,]))
        {
            if (dto.State2D is null)
                throw new InvalidCastException($"State is not of type {typeof(TState)}");
                
            return new Transition<TState>(
                dto.Guid,
                (TState)(object)dto.State2D,
                RLActions.Interpret(dto.DiscreteActions, dto.ContinuousActions),
                dto.Reward,
                dto.NextTransitionGuid);
        }
            
        throw new InvalidCastException($"Unsupported state type: {typeof(TState)}");
    }

    public static List<TransitionPortableDTO> ToDTOList<TState>(this List<Transition<TState>> transitions)
        where TState : notnull
    {
        return transitions.Select(transition => transition.ToDTO()).ToList();
    }

    public static List<Transition<TState>> FromDTOList<TState>(this List<TransitionPortableDTO> dtos)
        where TState : notnull
    {
        return dtos.Select(dto => dto.FromDTO<TState>()).ToList();
    }
}