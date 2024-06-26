using MessagePack;
using OneOf;
using System;
using System.Collections.Generic;

namespace RLMatrix.Common.Remote
{
    [MessagePackObject]
    public class ActionResponseDTO
    {
        [Key(0)]
        public Dictionary<Guid, ActionDTO> Actions { get; set; }

        public ActionResponseDTO(Dictionary<Guid, ActionDTO> actions)
        {
            Actions = actions;
        }
    }

    [MessagePackObject]
    public class ActionDTO
    {
        [Key(0)]
        public int[] DiscreteActions { get; set; }

        [Key(1)]
        public float[] ContinuousActions { get; set; }

        public ActionDTO(int[] discreteActions, float[] continuousActions = null)
        {
            DiscreteActions = discreteActions;
            ContinuousActions = continuousActions ?? new float[0];
        }
    }

    public static class ActionResponseDTOExtensions
    {
        public static ActionResponseDTO ToDTO(this Dictionary<Guid, OneOf<int[], (int[] discreteActions, float[] continuousActions)>> actions)
        {
            var actionDTOs = new Dictionary<Guid, ActionDTO>();
            foreach (var kvp in actions)
            {
                actionDTOs[kvp.Key] = kvp.Value.Match(
                    discreteActions => new ActionDTO(discreteActions),
                    continuousActions => new ActionDTO(continuousActions.discreteActions, continuousActions.continuousActions)
                );
            }
            return new ActionResponseDTO(actionDTOs);
        }

        public static Dictionary<Guid, OneOf<int[], (int[] discreteActions, float[] continuousActions)>> FromDTO(this ActionResponseDTO dto)
        {
            var actions = new Dictionary<Guid, OneOf<int[], (int[] discreteActions, float[] continuousActions)>>();
            foreach (var kvp in dto.Actions)
            {
                actions[kvp.Key] = kvp.Value.ContinuousActions.Length > 0
                    ? OneOf<int[], (int[] discreteActions, float[] continuousActions)>.FromT1((kvp.Value.DiscreteActions, kvp.Value.ContinuousActions))
                    : OneOf<int[], (int[] discreteActions, float[] continuousActions)>.FromT0(kvp.Value.DiscreteActions);
            }
            return actions;
        }
    }
}