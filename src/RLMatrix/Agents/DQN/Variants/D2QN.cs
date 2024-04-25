using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace RLMatrix
{
    public class D2QNAgent<T> : DQNAgent<T>
    {
        public D2QNAgent(DQNAgentOptions opts, List<IEnvironment<T>> env, IDQNNetProvider<T> netProvider = null)
            : base(opts, env, netProvider)
        {
        }

        public override void OptimizeModel()
        {
             if (myReplayBuffer.Length < myOptions.BatchSize)
                return;

            ReadOnlySpan<TransitionInMemory<T>> transitions = myReplayBuffer.Sample();

            (T[] batchStates, int[][] batchMultiActions, float[] batchRewards, T?[] batchNextStates, bool[] nonFinalMaskArray) = (null, null, null, null, null); //ExtractBatchData(transitions);

            Tensor nonFinalMask = tensor(batchNextStates.Select(s => s != null).ToArray()).to(myDevice);
            Tensor stateBatch = stack(batchStates.Select(s => StateToTensor(s)).ToArray()).to(myDevice);

            Tensor[] nonFinalNextStatesArray = batchNextStates.Where(s => s != null).Select(s => StateToTensor(s)).ToArray();
            Tensor nonFinalNextStates;
            if (nonFinalNextStatesArray.Length > 0)
            {
                nonFinalNextStates = stack(nonFinalNextStatesArray).to(myDevice);
            }
            else
            {
                // If all next states are terminal, we still need to create a dummy tensor for nonFinalNextStates
                // to prevent errors but it won't be used because nonFinalMask will filter them out.
                nonFinalNextStates = zeros(new long[] { 1, stateBatch.shape[1] }).to(myDevice); // Adjust the shape as necessary
            }

            Tensor actionBatch = stack(batchMultiActions.Select(a => tensor(a).to(torch.int64)).ToArray()).to(myDevice);
            Tensor rewardBatch = stack(batchRewards.Select(r => tensor(r)).ToArray()).to(myDevice);

            Tensor qValuesAllHeads = myPolicyNet.forward(stateBatch);

            Tensor expandedActionBatch = actionBatch.unsqueeze(2); // Expand to [batchSize, numHeads, 1]

            Tensor stateActionValues = qValuesAllHeads.gather(2, expandedActionBatch).squeeze(2).to(myDevice); // [batchSize, numHeads]

            Tensor nextStateValues;
            using (no_grad())
            {
                if (nonFinalNextStatesArray.Length > 0)
                {
                    // Use myPolicyNet to select the best action for each next state based on the current policy
                    Tensor nextActions = myPolicyNet.forward(nonFinalNextStates).max(2).indexes;

                    // Evaluate the selected actions' Q-values using myTargetNet
                    nextStateValues = myTargetNet.forward(nonFinalNextStates).gather(2, nextActions.unsqueeze(-1)).squeeze(-1);
                }
                else
                {
                    // If all states are terminal, set nextStateValues as zeros
                    nextStateValues = zeros(new long[] { myOptions.BatchSize, myEnvironments[0].actionSize.Count() }).to(myDevice);
                }
            }


            Tensor maskedNextStateValues = zeros(new long[] { myOptions.BatchSize, myEnvironments[0].actionSize.Count() }).to(myDevice);
            maskedNextStateValues.masked_scatter_(nonFinalMask.unsqueeze(1), nextStateValues);

            Tensor expectedStateActionValues = (maskedNextStateValues * myOptions.GAMMA) + rewardBatch.unsqueeze(1);

            // Compute Huber loss
            SmoothL1Loss criterion = torch.nn.SmoothL1Loss();
            Tensor loss = criterion.forward(stateActionValues, expectedStateActionValues);

            // Optimize the model
            myOptimizer.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_value_(myPolicyNet.parameters(), 100);
            myOptimizer.step();

        }
    }
}
