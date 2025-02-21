using System.Numerics;
using RLMatrix.Common;

namespace RLMatrix;

public class BaseQValuesComputer : IQValuesComputer
{
    public Tensor ComputeQValues(Tensor states, TensorModule policyNet)
    {
        var res = policyNet.forward(states);
        return res;
    }
}

public class BaseStateActionValuesExtractor : IStateActionValuesExtractor
{
    public Tensor ExtractStateActionValues(Tensor qValues, Tensor actions)
    {
        var expandedActionBatch = actions.unsqueeze(2);
        var res = qValues.gather(2, expandedActionBatch).squeeze(2);
        return res;
    }
}

//TODO: This takes somewhat long to execute?
public class BaseComputeNextStateValues : INextStateValuesComputer
{
    public Tensor ComputeNextStateValues(Tensor nonFinalNextStates, TensorModule targetNet, TensorModule policyNet, DQNAgentOptions opts, int[] discreteActionDimensions, Device device)
    {
        Tensor nextStateValues;
        using (torch.no_grad())
        {
            if (nonFinalNextStates.shape[0] > 0)
            {
                //code smell? should be bool instead of volatile dependency? 
                if (opts.UseDoubleDQN)
                {
                    // Use policyNet to select the best action for each next state based on the current policy
                    var nextActions = policyNet.forward(nonFinalNextStates).max(2).indexes;
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
                nextStateValues = torch.zeros(new long[] { opts.BatchSize, discreteActionDimensions.Length}, device: device);
            }
        }
        return nextStateValues;
    }
}

/// <summary>
///     Provides an optimized implementation for computing n-step returns using SIMD and parallelization.
/// </summary>
/// <typeparam name="TState">The type of state in the transitions.</typeparam>
public class BaseLookAheadStepsComputer<TState> : ILookAheadStepsComputer<TState>
    where TState : notnull
{
    /// <summary>
    ///     Computes n-step returns (look-ahead steps) for a batch of transitions using SIMD operations and parallel processing.
    /// </summary>
    /// <param name="transitions">The list of transitions to process.</param>
    /// <param name="opts">The DQN agent options containing parameters like n-step return and discount factor.</param>
    /// <param name="device">The device to use for tensor operations.</param>
    /// <returns>A tensor containing the computed n-step returns.</returns>
    public Tensor ComputeLookAheadSteps(IList<MemoryTransition<TState>> transitions, DQNAgentOptions opts, Device device)
    {
        var batchSize = transitions.Count;
        var returnsArray = new float[batchSize];

        Parallel.For(0, batchSize, i =>
        {
            var currentTransition = transitions[i];
            float nStepReturn = 0;
            float discount = 1;
            var rewards = new float[Vector<float>.Count];

            for (var j = 0; j < opts.LookAheadSteps; j += Vector<float>.Count)
            {
                var remainingSteps = Math.Min(Vector<float>.Count, opts.LookAheadSteps - j);

                for (var k = 0; k < remainingSteps; k++)
                {
                    if (currentTransition != null)
                    {
                        rewards[k] = currentTransition.Reward;
                        currentTransition = currentTransition.NextTransition;
                    }
                    else
                    {
                        rewards[k] = 0;
                    }
                }

                var rewardsVector = new Vector<float>(rewards);
                var discountVector = new Vector<float>(discount);
                nStepReturn += Vector.Dot(rewardsVector, discountVector);

                discount *= (float)Math.Pow(opts.Gamma, remainingSteps);

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

public class BaseExpectedStateActionValuesComputer<TState> : IExpectedStateActionValuesComputer<TState>
    where TState : notnull
{
    private readonly BaseLookAheadStepsComputer<TState> _lookAheadStepsComputer = new();

    public Tensor ComputeExpectedStateActionValues(Tensor nextStateValues, Tensor rewardBatch, Tensor nonFinalMask, DQNAgentOptions opts, IList<MemoryTransition<TState>> transitions, int[] discreteActions, Device device)
    {
        var maskedNextStateValues = torch.zeros(new long[] { opts.BatchSize, discreteActions.Length }, device: device);
        maskedNextStateValues.masked_scatter_(nonFinalMask.unsqueeze(1), nextStateValues);
        if (opts.LookAheadSteps <= 1)
        {
            var res = (maskedNextStateValues * opts.Gamma) + rewardBatch.unsqueeze(1);
            return res;
        }
        
        // Compute n-step returns
        var nStepRewards = _lookAheadStepsComputer.ComputeLookAheadSteps(transitions, opts, device);

        return maskedNextStateValues * Math.Pow(opts.Gamma, opts.LookAheadSteps) + nStepRewards.unsqueeze(1);
    }
}

public class BaseLossComputer : ILossComputer
{
    public Tensor ComputeLoss(Tensor stateActionValues, Tensor expectedStateActionValues)
    {
        var criterion = new SmoothL1Loss();
        var loss = criterion.forward(stateActionValues, expectedStateActionValues);
        return loss;
    }
}