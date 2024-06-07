using RLMatrix.Agents.Common;
using RLMatrix.Memories;
using TorchSharp;
using static TorchSharp.torch;

namespace RLMatrix.Agents.PPO.Implementations
{
    public class DiscretePPOAgent<T> : IDiscretePPOAgent<T>
    {
        //Todo: these PPOActor/Critic should be interfaces
        public required PPOActorNet actorNet { get; set; }
        public required PPOCriticNet criticNet { get; set; }
        public required IOptimize<T> Optimizer { get; init; }
        public required IMemory<T> Memory { get; set; }
        public required int[] ActionSizes { get; init; }
        public required PPOAgentOptions Options { get; init; }
        public required Device Device { get; init; }

        public void AddTransition(IEnumerable<TransitionPortable<T>> transitions)
        {
            Memory.Push(transitions.ToTransitionInMemory());
        }

        public void OptimizeModel()
        {
            Optimizer.Optimize(Memory);
        }

        public int[][] SelectActions(T[] states, bool isTraining)
        {
            using (var scope = torch.no_grad())
            {
                Tensor stateTensor = Utilities<T>.StateBatchToTensor(states, Device);
                var result = actorNet.forward(stateTensor);
                int[][] actions = new int[states.Length][];

                if (isTraining)
                {
                    // Discrete Actions
                    for (int i = 0; i < states.Length; i++)
                    {
                        actions[i] = new int[ActionSizes.Length];
                        for (int j = 0; j < ActionSizes.Length; j++)
                        {
                            var actionProbs = result[i, j];
                            var actionSample = torch.multinomial(actionProbs, 1, true);
                            actions[i][j] = (int)actionSample.item<long>();
                        }
                    }
                }
                else
                {
                    // Discrete Actions
                    for (int i = 0; i < states.Length; i++)
                    {
                        actions[i] = new int[ActionSizes.Length];
                        for (int j = 0; j < ActionSizes.Length; j++)
                        {
                            var actionProbs = result[i, j];
                            var actionIndex = actionProbs.argmax();
                            actions[i][j] = (int)actionIndex.item<long>();
                        }
                    }
                }

                return actions;
            }
        }
        int[][] SelectActions2(T[] states, bool isTraining)
        {
            int[][] actions = new int[states.Length][];
            float[][] continuousActions = new float[states.Length][];

            for (int i = 0; i < states.Length; i++)
            {
                using (var scope = torch.no_grad())
                {
                    Tensor stateTensor = Utilities<T>.StateToTensor(states[i], Device);
                    var result = actorNet.forward(stateTensor);

                    if (isTraining)
                    {
                        // Discrete Actions
                        actions[i] = PPOActionSelection<T>.SelectDiscreteActionsFromProbs(result, ActionSizes);
                        // Continuous Actions
                        continuousActions[i] = PPOActionSelection<T>.SampleContinuousActions(result, ActionSizes, new (float, float)[0]);
                    }
                    else
                    {
                        // Discrete Actions
                        actions[i] = PPOActionSelection<T>.SelectGreedyDiscreteActions(result, ActionSizes);
                        // Continuous Actions
                        continuousActions[i] = PPOActionSelection<T>.SelectMeanContinuousActions(result, ActionSizes, new (float, float)[0]);
                    }
                }
            }

            return actions;
        }

        public virtual (int[] actions, Tensor? memoryState)[] SelectActionsRecurrent((T state, Tensor? memoryState)[] states, bool isTraining)
        {
            throw new Exception("Using recurrent action selection with non recurrent agent, use int[][] SelectActions(T[] states, bool isTraining) signature instead");
        }
    }
}


    

