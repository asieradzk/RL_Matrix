using RLMatrix.Common;

namespace RLMatrix;

public class DiscreteRecurrentPPOAgent<TState> : DiscretePPOAgent<TState>
    where TState : notnull
{
    public DiscreteRecurrentPPOAgent(PPOActorNet actorNet, PPOCriticNet criticNet, IOptimizer<TState> optimizer, IMemory<TState> memory, 
        int[] discreteActionDimensions, PPOAgentOptions options, Device device) 
        : base(actorNet, criticNet, optimizer, memory, discreteActionDimensions, options, device)
    { }

    public override ValueTask<ActionsState[]> SelectActionsRecurrentAsync(RLMemoryState<TState>[] states, bool isTraining)
    {
        var results = new ActionsState[states.Length];
        
        for (var i = 0; i < states.Length; i++)
        {
            using (torch.no_grad())
            {
                var stateTensor = Utilities<TState>.StateToTensor(states[i].State, Device);
                var (res, memoryState, memoryState2) = ActorNet.forward(stateTensor, states[i].MemoryState, states[i].MemoryState2);
                
                if (isTraining)
                {
                    results[i] = new ActionsState(
                        RLActions.Discrete(PPOActionSelection.SelectDiscreteActionsFromProbabilities(res, DiscreteActionDimensions)),
                        memoryState,
                        memoryState2);
                }
                else
                {
                    results[i] = new ActionsState(
                        RLActions.Discrete(PPOActionSelection.SelectGreedyDiscreteActions(res, DiscreteActionDimensions)),
                        memoryState,
                        memoryState2);
                }
            }
        }
        
        return new(results);
    }

    public override ValueTask<RLActions[]> SelectActionsAsync(TState[] states, bool isTraining)
    {
        throw new Exception($"Using non recurrent action selection with recurrent agent, use {nameof(SelectActionsRecurrentAsync)} instead.");
    }
}