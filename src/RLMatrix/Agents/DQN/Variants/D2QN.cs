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
        public D2QNAgent(DQNAgentOptions opts, IEnvironment<T> env, IDQNNetProvider<T> netProvider = null)
            : base(opts, env, netProvider)
        {
        }

        public override void OptimizeModel()
        {
            if (myReplayBuffer.Length < myOptions.BatchSize)
                return;

            List<Transition<T>> transitions = myReplayBuffer.Sample();

            List<T> batchStates = transitions.Select(t => t.state).ToList();
            List<int> batchActions = transitions.Select(t => t.action).ToList();
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

            Tensor actionBatch = stack(batchActions.Select(a => tensor(new int[] { a }).to(torch.int64)).ToArray()).to(myDevice);
            Tensor rewardBatch = stack(batchRewards.Select(r => tensor(r)).ToArray()).to(myDevice);

            Tensor stateActionValues = myPolicyNet.forward(stateBatch).gather(1, actionBatch).to(myDevice);

            Tensor nextStateValues = zeros(new long[] { myOptions.BatchSize }).to(myDevice);
            using (no_grad())
            {
                var nextActions = myPolicyNet.forward(nonFinalNextStates).argmax(1);
                nextStateValues.masked_scatter_(nonFinalMask, myTargetNet.forward(nonFinalNextStates).gather(1, nextActions.unsqueeze(-1)).squeeze(-1));
            }

            Tensor expectedStateActionValues = (nextStateValues.detach() * myOptions.GAMMA) + rewardBatch;

            SmoothL1Loss criterion = torch.nn.SmoothL1Loss();
            Tensor loss = criterion.forward(stateActionValues, expectedStateActionValues.unsqueeze(1));

            myOptimizer.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_value_(myPolicyNet.parameters(), 100);
            myOptimizer.step();
        }
    }
}
