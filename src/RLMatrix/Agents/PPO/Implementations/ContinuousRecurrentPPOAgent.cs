using RLMatrix.Common;

namespace RLMatrix;

public class ContinuousRecurrentPPOAgent<TState> : ContinuousPPOAgent<TState>
    where TState : notnull
{
    public ContinuousRecurrentPPOAgent(PPOActorNet actorNet, PPOCriticNet criticNet, IOptimizer<TState> optimizer, IMemory<TState> memory, 
        int[] discreteActionDimensions, ContinuousActionDimensions[] continuousActionDimensions, PPOAgentOptions options, Device device) 
        : base(actorNet, criticNet, optimizer, memory, discreteActionDimensions, continuousActionDimensions, options, device)
    { }

    public override ValueTask<ActionsState[]> SelectActionsRecurrentAsync(RLMemoryState<TState>[] states, bool isTraining)
    {
        var result = new ActionsState[states.Length];

        for (var i = 0; i < states.Length; i++)
        {
            using (torch.no_grad())
            {
                var stateTensor = Utilities<TState>.StateToTensor(states[i].State, Device);
                var (res, memoryState, memoryState2) = ActorNet.forward(stateTensor, states[i].MemoryState, states[i].MemoryState2);

                if (isTraining)
                {
                    result[i] = new ActionsState(
                        RLActions.Continuous(
                            PPOActionSelection.SelectDiscreteActionsFromProbabilities(res, DiscreteActionDimensions),
                            PPOActionSelection.SampleContinuousActions(res, DiscreteActionDimensions, ContinuousActionDimensions)),
                        memoryState,
                        memoryState2);
                }
                else
                {
                    result[i] = new ActionsState(
                        RLActions.Continuous(
                            PPOActionSelection.SelectGreedyDiscreteActions(res, DiscreteActionDimensions),
                            PPOActionSelection.SelectMeanContinuousActions(res, DiscreteActionDimensions, ContinuousActionDimensions)),
                        memoryState,
                        memoryState2);
                }
            }
        }

        return new(result);
    }
    
    public override ValueTask<RLActions[]> SelectActionsAsync(TState[] states, bool isTraining)
    {
        throw new NotSupportedException($"Using non recurrent action selection with recurrent agent, use {nameof(SelectActionsRecurrentAsync)} instead.");
    }
}