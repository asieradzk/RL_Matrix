using RLMatrix.Agents.DQN.Domain;
using RLMatrix.Memories;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace RLMatrix
{
    public class BaseComputeQValues : IComputeQValues
    {
        public Tensor ComputeQValues(Tensor states, Module<Tensor, Tensor> policyNet, int[] ActionSizes, int numAtoms)
        {
            var res = policyNet.forward(states);
            return res;
        }
    }

    public class BaseExtractStateActionValues : IExtractStateActionValues
    {
        public Tensor ExtractStateActionValues(Tensor qValues, Tensor actions)
        {
            Tensor expandedActionBatch = actions.unsqueeze(2);
            var res = qValues.gather(2, expandedActionBatch).squeeze(2);
            return res;
        }
    }

    public class BaseComputeNextStateValues : IComputeNextStateValues
    {
        public Tensor ComputeNextStateValues(Tensor nonFinalNextStates, Module<Tensor, Tensor> targetNet, Module<Tensor, Tensor> policyNet, DQNAgentOptions opts, int[] ActionSize, Device device)
        {
            Tensor nextStateValues;
            using (no_grad())
            {
                if (nonFinalNextStates.shape[0] > 0)
                {
                    //code smell? should be bool instead of volatile dependency? 
                    if (opts.DoubleDQN)
                    {
                        // Use policyNet to select the best action for each next state based on the current policy
                        Tensor nextActions = policyNet.forward(nonFinalNextStates).max(2).indexes;
                        // Evaluate the selected actions' Q-values using targetNet
                        nextStateValues = targetNet.forward(nonFinalNextStates).gather(2, nextActions.unsqueeze(-1)).squeeze(-1);
                    }
                    else
                    {
                        nextStateValues = targetNet.forward(nonFinalNextStates).max(2).values; // [batchSize, numHeads]
                    }
                }
                else
                {
                    nextStateValues = zeros(new long[] { opts.BatchSize, ActionSize.Length}, device: device);
                }
            }
            return nextStateValues;
        }
    }

    public class BaseComputeNStepReturns<T> : IComputeNStepReturns<T>
    {
        public Tensor ComputeNStepReturns(ref ReadOnlySpan<TransitionInMemory<T>> transitions, DQNAgentOptions opts, Device device)
        {
            int batchSize = transitions.Length;
            Tensor returns = torch.zeros(batchSize, device: device);

            for (int i = 0; i < batchSize; i++)
            {
                TransitionInMemory<T> currentTransition = transitions[i];
                float nStepReturn = 0;
                float discount = 1;

                for (int j = 0; j < opts.NStepReturn; j++)
                {
                    nStepReturn += discount * currentTransition.reward;
                    if (currentTransition.nextTransition is null)
                    {
                        break;
                    }
                    currentTransition = currentTransition.nextTransition;

                    discount *= opts.GAMMA;
                }
                
                returns[i] = nStepReturn;
            }
            return returns;
        }
    }

    public class BaseComputeExpectedStateActionValues<T> : IComputeExpectedStateActionValues<T>
    {
        public Tensor ComputeExpectedStateActionValues(Tensor nextStateValues, Tensor rewardBatch, Tensor nonFinalMask, DQNAgentOptions opts, ref ReadOnlySpan<TransitionInMemory<T>> transitions, int[] ActionCount, Device device)
        {
            Tensor maskedNextStateValues = zeros(new long[] { opts.BatchSize, ActionCount.Length }, device: device);
            maskedNextStateValues.masked_scatter_(nonFinalMask.unsqueeze(1), nextStateValues);
            if (opts.NStepReturn <= 1)
            {
                var res = (maskedNextStateValues * opts.GAMMA) + rewardBatch.unsqueeze(1);
                return res;
            }
            else
            {
                // Compute n-step returns
                var computeNStepReturns = new BaseComputeNStepReturns<T>();
                Tensor nStepRewards = computeNStepReturns.ComputeNStepReturns(ref transitions, opts, device);

                return (maskedNextStateValues * Math.Pow(opts.GAMMA, opts.NStepReturn)) + nStepRewards.unsqueeze(1);
            }
        }
    }

    public class BaseComputeLoss : IComputeLoss
    {
        public Tensor ComputeLoss(Tensor stateActionValues, Tensor expectedStateActionValues)
        {
            SmoothL1Loss criterion = torch.nn.SmoothL1Loss();
            var loss = criterion.forward(stateActionValues, expectedStateActionValues);
            return loss;
        }
    }
}