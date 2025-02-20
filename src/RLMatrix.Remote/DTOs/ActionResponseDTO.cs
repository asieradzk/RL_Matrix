using MessagePack;
using RLMatrix.Common;

namespace RLMatrix.Remote;

[MessagePackObject]
public class ActionResponseDTO(Dictionary<Guid, ActionDTO> actions)
{
    [Key(0)]
    public Dictionary<Guid, ActionDTO> Actions { get; } = actions;
}

[MessagePackObject]
public class ActionDTO(int[] discreteActions, float[]? continuousActions = null)
{
    [Key(0)]
    public int[] DiscreteActions { get; } = discreteActions;

    [Key(1)]
    public float[] ContinuousActions { get; } = continuousActions ?? [];
}

public static class ActionResponseDTOExtensions
{
    public static ActionResponseDTO ToDTO(this Dictionary<Guid, RLActions> actions)
    {
        var actionDTOs = actions.ToDictionary(
            x => x.Key,
            x => new ActionDTO(x.Value.DiscreteActions, x.Value.IsContinuous ? x.Value.ContinuousActions : null));
            
        return new ActionResponseDTO(actionDTOs);
    }

    public static Dictionary<Guid, RLActions> FromDTO(this ActionResponseDTO dto)
    {
        var actions = dto.Actions.ToDictionary(
            x => x.Key,
            x => x.Value.ContinuousActions.Length > 0
                ? RLActions.Continuous(x.Value.DiscreteActions, x.Value.ContinuousActions)
                : RLActions.Discrete(x.Value.DiscreteActions));

        return actions;
    }
}