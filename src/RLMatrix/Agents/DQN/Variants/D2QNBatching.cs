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
        public D2QNBatching(DQNAgentOptions opts, IEnvironment<T> env, IDQNNetProvider<T> netProvider = null)
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


            //TODO: Transition/s is a ridiculous name from pytorch DQN example, should this be changed?
            List<Transition<T>> transitions = myReplayBuffer.Sample();

            

            // Transpose the batch (convert batch of Transitions to Transition of batches)
            List<T> batchStates = transitions.Select(t => t.state).ToList();

            List<int> batchActions = transitions.Select(t => t.action).ToList();
            List<float> batchRewards = transitions.Select(t => t.reward).ToList();
            List<T> batchNextStates = transitions.Select(t => t.nextState).ToList();
            // Compute a mask of non-final states and concatenate the batch elements
            Tensor nonFinalMask = tensor(batchNextStates.Select(s => s != null).ToArray()).to(myDevice);
            Tensor stateBatch = stack(batchStates.Select(s => StateToTensor(s)).ToArray()).to(myDevice);

            //This clumsy part is to account for situation where batch is picked where each episode has only 1 step
            //Why 1 step episode is a problem anyway? It should still have Q value for the action taken
            Tensor[] nonFinalNextStatesArray = batchNextStates.Where(s => s != null).Select(s => StateToTensor(s)).ToArray();
            Tensor nonFinalNextStates;
            if (nonFinalNextStatesArray.Length > 0)
            {
                nonFinalNextStates = stack(nonFinalNextStatesArray).to(myDevice);
                // Continue with the rest of your code
            }
            else
            {
                return;
                // Handle the case when all states are terminal
            }

            Tensor actionBatch = stack(batchActions.Select(a => tensor(new int[] { a }).to(torch.int64)).ToArray()).to(myDevice);
            Tensor rewardBatch = stack(batchRewards.Select(r => tensor(r)).ToArray()).to(myDevice);
            // Compute Q(s_t, a)
            Tensor stateActionValues = myPolicyNet.forward(stateBatch).gather(1, actionBatch).to(myDevice);


            // Compute V(s_{t+1}) for all next states.
            Tensor nextStateValues = zeros(new long[] { myOptions.BatchSize }).to(myDevice);
            using (no_grad())
            {
                nextStateValues.masked_scatter_(nonFinalMask, myTargetNet.forward(nonFinalNextStates).max(1).values);

            }
            // Compute the expected Q values
            Tensor expectedStateActionValues = (nextStateValues.detach() * myOptions.GAMMA) + rewardBatch;

            // Compute Huber loss
            SmoothL1Loss criterion = torch.nn.SmoothL1Loss();
            Tensor loss = criterion.forward(stateActionValues, expectedStateActionValues.unsqueeze(1));

            // Optimize the model
            myOptimizer.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_value_(myPolicyNet.parameters(), 100);
            myOptimizer.step();

            loss.print();
        }
    }
}
