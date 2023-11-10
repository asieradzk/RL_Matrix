using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace RLMatrix.Agents.DQN.Variants
{
    public class D2QNBatching<T> : DQNAgent<T>
    {
        public D2QNBatching(DQNAgentOptions opts, List<IEnvironment<T>> env, IDQNNetProvider<T> netProvider = null)
           : base(opts, env, netProvider)
        {

           
        }



        public void ImplantMemory(ReplayMemory<T> memory)
        {
            myReplayBuffer = memory;
        }

        public override void OptimizeModel()
        {
            if (myReplayBuffer.Length < myOptions.BatchSize)
                return;

            List<Transition<T>> transitions = myReplayBuffer.Sample();

            List<T> batchStates = transitions.Select(t => t.state).ToList();
            List<int[]> batchActions = transitions.Select(t => t.discreteActions).ToList();
            List<float> batchRewards = transitions.Select(t => t.reward).ToList();
            List<T> batchNextStates = transitions.Select(t => t.nextState).ToList();

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
                return;
            }

            Tensor actionBatch = stack(batchActions.Select(a => tensor(a).to(torch.int64)).ToArray()).to(myDevice);

            Tensor rewardBatch = stack(batchRewards.Select(r => tensor(r)).ToArray()).to(myDevice);

            Tensor qValuesAllHeads = myPolicyNet.forward(stateBatch); // [batchSize, numHeads, numActions]
            Tensor expandedActionBatch = actionBatch.unsqueeze(2); // Expand to [batchSize, numHeads, 1]

            Tensor stateActionValues = qValuesAllHeads.gather(2, expandedActionBatch).squeeze(2).to(myDevice);  // [batchSize, numHeads]

            Tensor targetNextStateValues;

            using (no_grad())
            {
                Tensor nextQValuesAllHeads = myPolicyNet.forward(nonFinalNextStates);  // [batchSize, numHeads, numActions]
                var nextActions = nextQValuesAllHeads.argmax(2);  // [batchSize, numHeads]

                Tensor targetNextQValuesAllHeads = myTargetNet.forward(nonFinalNextStates);  // [batchSize, numHeads, numActions]
                Tensor expandedNextActions = nextActions.unsqueeze(2);  // [batchSize, numHeads, 1]
                targetNextStateValues = targetNextQValuesAllHeads.gather(2, expandedNextActions).squeeze(2);  // [batchSize, numHeads]
            }
            Tensor maskedTargetNextStateValues = zeros(new long[] { myOptions.BatchSize, myEnvironments[0].actionSize.Count() }).to(myDevice);
            maskedTargetNextStateValues.masked_scatter_(nonFinalMask.unsqueeze(1), targetNextStateValues);


            Tensor expectedStateActionValues = (maskedTargetNextStateValues * myOptions.GAMMA) + rewardBatch.unsqueeze(1);

            SmoothL1Loss criterion = torch.nn.SmoothL1Loss();
            Tensor loss = criterion.forward(stateActionValues, expectedStateActionValues);

            myOptimizer.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_value_(myPolicyNet.parameters(), 100);
            myOptimizer.step();
            loss.print();
        }
    }
}
