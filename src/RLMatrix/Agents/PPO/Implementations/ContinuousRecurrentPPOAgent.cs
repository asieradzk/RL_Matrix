using RLMatrix.Agents.Common;
using TorchSharp;
using static TorchSharp.torch;

namespace RLMatrix.Agents.PPO.Implementations
{
    public class ContinuousRecurrentPPOAgent<T> : ContinuousPPOAgent<T>
    {
        public override ((int[] discreteActions, float[] continuousActions) actions, Tensor? memoryState, Tensor? memoryState2)[] SelectActionsRecurrent((T state, Tensor? memoryState, Tensor? memoryState2)[] states, bool isTraining)
        {
            var result = new ((int[] discreteActions, float[] continuousActions) actions, Tensor? memoryState, Tensor? memoryState2)[states.Length];

            for (int i = 0; i < states.Length; i++)
            {
                using (var scope = torch.no_grad())
                {
                    Tensor stateTensor = Utilities<T>.StateToTensor(states[i].state, Device);
                    var forwardResult = actorNet.forward(stateTensor, states[i].memoryState, states[i].memoryState2);

                    if (isTraining)
                    {
                        int[] discreteActions = PPOActionSelection<T>.SelectDiscreteActionsFromProbs(forwardResult.Item1, DiscreteDimensions);
                        float[] continuousActions = PPOActionSelection<T>.SampleContinuousActions(forwardResult.Item1, DiscreteDimensions, ContinuousActionBounds);
                        result[i] = ((discreteActions, continuousActions), forwardResult.Item2, forwardResult.Item3);
                    }
                    else
                    {
                        int[] discreteActions = PPOActionSelection<T>.SelectGreedyDiscreteActions(forwardResult.Item1, DiscreteDimensions);
                        float[] continuousActions = PPOActionSelection<T>.SelectMeanContinuousActions(forwardResult.Item1, DiscreteDimensions, ContinuousActionBounds);
                        result[i] = ((discreteActions, continuousActions), forwardResult.Item2, forwardResult.Item3);
                    }
                }
            }

            return result;
        }
        public new (int[] discreteActions, float[][] continuousActions) SelectActions(T[] states, bool isTraining)
        {
            Console.WriteLine("Using non recurrent action selection with recurrent agent, use ((int[] discreteActions, float[] continuousActions) actions, Tensor? memoryState, Tensor? memoryState2)[] signature instead");
            throw new Exception("Using non recurrent action selection with recurrent agent, use ((int[] discreteActions, float[] continuousActions) actions, Tensor? memoryState, Tensor? memoryState2)[] signature instead");
        }
    }
}