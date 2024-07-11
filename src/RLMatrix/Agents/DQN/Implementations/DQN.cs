using RLMatrix.Agents.Common;
using RLMatrix.Agents.DQN.Domain;
using RLMatrix.Memories;
using System.Numerics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace RLMatrix
{
    public class BaseComputeQValues : IComputeQValues
    {
        public Tensor ComputeQValues(Tensor states, Module<Tensor, Tensor> policyNet)
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
    //TODO: This takes somewhat long to execute?
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

/// <summary>
/// Provides an optimized implementation for computing n-step returns using SIMD and parallelization.
/// </summary>
/// <typeparam name="T">The type of state in the transitions.</typeparam>
public class BaseComputeNStepReturns<T> : IComputeNStepReturns<T>
    {
        /// <summary>
        /// Computes n-step returns for a batch of transitions using SIMD operations and parallel processing.
        /// </summary>
        /// <param name="transitions">The list of transitions to process.</param>
        /// <param name="opts">The DQN agent options containing parameters like n-step return and discount factor.</param>
        /// <param name="device">The device to use for tensor operations.</param>
        /// <returns>A tensor containing the computed n-step returns.</returns>
        public Tensor ComputeNStepReturns(IList<TransitionInMemory<T>> transitions, DQNAgentOptions opts, Device device)
        {
            int batchSize = transitions.Count;
            float[] returnsArray = new float[batchSize];

            Parallel.For(0, batchSize, i =>
            {
                TransitionInMemory<T> currentTransition = transitions[i];
                float nStepReturn = 0;
                float discount = 1;
                float[] rewards = new float[Vector<float>.Count];

                for (int j = 0; j < opts.NStepReturn; j += Vector<float>.Count)
                {
                    int remainingSteps = Math.Min(Vector<float>.Count, opts.NStepReturn - j);

                    for (int k = 0; k < remainingSteps; k++)
                    {
                        if (currentTransition != null)
                        {
                            rewards[k] = currentTransition.reward;
                            currentTransition = currentTransition.nextTransition;
                        }
                        else
                        {
                            rewards[k] = 0;
                        }
                    }

                    Vector<float> rewardsVector = new Vector<float>(rewards);
                    Vector<float> discountVector = new Vector<float>(discount);
                    nStepReturn += Vector.Dot(rewardsVector, discountVector);

                    discount *= (float)Math.Pow(opts.GAMMA, remainingSteps);

                    if (currentTransition == null)
                        break;
                }

                returnsArray[i] = nStepReturn;
            });

            return torch.tensor(returnsArray, device: device);
        }



        //When profiling this was a bottleneck. The Parallel + SIMD version is 50-100 times faster than single threaded one and 1.5-2 times faster than just multi-threaded one:
        /*
         public Tensor ComputeNStepReturns(IList<TransitionInMemory<T>> transitions, DQNAgentOptions opts, Device device)
    {
        int batchSize = transitions.Count;
        float[] returnsArray = new float[batchSize];

        Parallel.For(0, batchSize, i =>
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

            returnsArray[i] = nStepReturn;
        });

        return torch.tensor(returnsArray, device: device);
    }
         * */
    }

    public class BaseComputeExpectedStateActionValues<T> : IComputeExpectedStateActionValues<T>
    {
        BaseComputeNStepReturns<T> computeNStepReturns = new BaseComputeNStepReturns<T>();

        public Tensor ComputeExpectedStateActionValues(Tensor nextStateValues, Tensor rewardBatch, Tensor nonFinalMask, DQNAgentOptions opts, IList<TransitionInMemory<T>> transitions, int[] ActionCount, Device device)
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
                Tensor nStepRewards = computeNStepReturns.ComputeNStepReturns(transitions, opts, device);

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