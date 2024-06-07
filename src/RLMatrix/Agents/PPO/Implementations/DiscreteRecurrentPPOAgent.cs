using RLMatrix.Agents.Common;
using TorchSharp;
using static TorchSharp.torch;

namespace RLMatrix.Agents.PPO.Implementations
{
    public class DiscreteRecurrentPPOAgent<T> : DiscretePPOAgent<T>
    {
        public override (int[] actions, Tensor? memoryState)[] SelectActionsRecurrent((T state, Tensor? memoryState)[] states, bool isTraining)
        {
            int[][] actions = new int[states.Length][];
            float[][] continuousActions = new float[states.Length][];
            Tensor[] memoryStates = new Tensor[states.Length];

            for (int i = 0; i < states.Length; i++)
            {
                using (var scope = torch.no_grad())
                {
                    Tensor stateTensor = Utilities<T>.StateToTensor(states[i].state, Device);
                    var result = actorNet.forward(stateTensor, states[i].memoryState);

                    if (isTraining)
                    {
                        // Discrete Actions
                        actions[i] = PPOActionSelection<T>.SelectDiscreteActionsFromProbs(result.Item1, ActionSizes);
                        // Continuous Actions
                        continuousActions[i] = PPOActionSelection<T>.SampleContinuousActions(result.Item1, ActionSizes, new (float, float)[0]);
                    }
                    else
                    {
                        // Discrete Actions
                        actions[i] = PPOActionSelection<T>.SelectGreedyDiscreteActions(result.Item1, ActionSizes);
                        // Continuous Actions
                        continuousActions[i] = PPOActionSelection<T>.SelectMeanContinuousActions(result.Item1, ActionSizes, new (float, float)[0]);
                    }

                    memoryStates[i] = result.Item2;
                }

            }

            return actions.Zip(memoryStates, (actions, memoryState) => (actions, memoryState)).ToArray();
        }


        public new int[][] SelectActions(T[] states, bool isTraining)
        {
            Console.WriteLine("Using non recurrent action selection with recurrent agent, use (int[] actions, Tensor ? memoryState)[] signature instead");
            throw new Exception("Using non recurrent action selection with recurrent agent, use (int[] actions, Tensor ? memoryState)[] signature instead");
        }
    }
}


    

