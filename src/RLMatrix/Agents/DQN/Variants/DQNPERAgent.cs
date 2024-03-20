using OneOf;
using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


namespace RLMatrix
{

    /// <summary>
    /// Represents a Deep Q-Learning agent.
    /// </summary>
    /// <typeparam name="T">The type of the state representation, either float[] or float[,].</typeparam>
    public class DQNAgentPER<T> : DQNAgent<T>
    {
        public DQNAgentPER(DQNAgentOptions opts, List<IEnvironment<T>> envs, IDQNNetProvider<T> netProvider = null)
            : base(opts, envs, netProvider)
        {
            myReplayBuffer = new PrioritizedReplayMemory<T>(myOptions.MemorySize, myOptions.BatchSize);
        }

        public override void OptimizeModel()
        {
            if (myReplayBuffer.Length < myOptions.BatchSize)
                return;

            List<Transition<T>> transitions = myReplayBuffer.Sample();
            List<int> sampledIndices = null;

            if (myReplayBuffer is PrioritizedReplayMemory<T> prioritizedReplayBuffer)
            {
                sampledIndices = prioritizedReplayBuffer.GetSampledIndices();
            }

            List<T> batchStates = transitions.Select(t => t.state).ToList();
            List<int[]> batchMultiActions = transitions.Select(t => t.discreteActions).ToList();
            List<float> batchRewards = transitions.Select(t => t.reward).ToList();
            List<T> batchNextStates = transitions.Select(t => t.nextState).ToList();

            Tensor nonFinalMask = CreateNonFinalMask(batchNextStates);
            Tensor nextStateValues = ComputeNextStateValues(batchNextStates, nonFinalMask);
            Tensor stateBatch = CreateStateBatch(batchStates);
            Tensor actionBatch = CreateActionBatch(batchMultiActions);
            Tensor rewardBatch = CreateRewardBatch(batchRewards);

            Tensor qValuesAllHeads = ComputeQValues(stateBatch);
            Tensor stateActionValues = ExtractStateActionValues(qValuesAllHeads, actionBatch);

            
            Tensor expectedStateActionValues = ComputeExpectedStateActionValues(nextStateValues, rewardBatch, nonFinalMask);

            Tensor loss = ComputeLoss(stateActionValues, expectedStateActionValues);

            UpdateModel(loss);

            if (sampledIndices != null)
            {
                UpdatePrioritizedReplayMemory(stateActionValues, expectedStateActionValues.detach(), sampledIndices);
            }
        }

        protected virtual Tensor CreateNonFinalMask(List<T> batchNextStates)
        {
            return tensor(batchNextStates.Select(s => s != null).ToArray()).to(myDevice);
        }

        protected virtual Tensor CreateStateBatch(List<T> batchStates)
        {
            return stack(batchStates.Select(s => StateToTensor(s)).ToArray()).to(myDevice);
        }

        protected virtual Tensor CreateActionBatch(List<int[]> batchMultiActions)
        {
            return stack(batchMultiActions.Select(a => tensor(a).to(torch.int64)).ToArray()).to(myDevice);
        }

        protected virtual Tensor CreateRewardBatch(List<float> batchRewards)
        {
            return stack(batchRewards.Select(r => tensor(r)).ToArray()).to(myDevice);
        }
        protected virtual Tensor CreateRewardBatch(long batchSize)
        {
            return ones(new long[] { batchSize }).mul(1f).to(myDevice); // [batch_size]
        }

        protected virtual Tensor ComputeQValues(Tensor stateBatch)
        {
            return myPolicyNet.forward(stateBatch);
        }

        protected virtual Tensor ExtractStateActionValues(Tensor qValuesAllHeads, Tensor actionBatch)
        {
            Tensor expandedActionBatch = actionBatch.unsqueeze(2);
            var res = qValuesAllHeads.gather(2, expandedActionBatch).squeeze(2).to(myDevice);
            return res;
            
        }

        protected virtual Tensor ComputeNextStateValues(List<T> batchNextStates, Tensor nonFinalMask)
        {
            Tensor[] nonFinalNextStatesArray = batchNextStates.Where(s => s != null).Select(s => StateToTensor(s)).ToArray();

            if (nonFinalNextStatesArray.Length > 0)
            {
                Tensor nonFinalNextStates = stack(nonFinalNextStatesArray).to(myDevice);

                using (no_grad())
                {
                    // Use the policy network (value net) to select the best action for each next state
                    Tensor nextActions = myPolicyNet.forward(nonFinalNextStates).max(2).indexes;
                    // Use the target network to evaluate the selected actions
                    return myTargetNet.forward(nonFinalNextStates).gather(2, nextActions.unsqueeze(-1)).squeeze(-1);
                }
            }
            else
            {
                //All steps are final likely this means all episodes are 1-step long.
                return zeros(new long[] { myOptions.BatchSize, myEnvironments[0].actionSize.Count() }).to(myDevice);
            }
        }

        protected virtual Tensor ComputeExpectedStateActionValues(Tensor nextStateValues, Tensor rewardBatch, Tensor nonFinalMask)
        {
            Tensor maskedNextStateValues = zeros(new long[] { myOptions.BatchSize, myEnvironments[0].actionSize.Count() }).to(myDevice);
            maskedNextStateValues.masked_scatter_(nonFinalMask.unsqueeze(1), nextStateValues);
            return (maskedNextStateValues * myOptions.GAMMA) + rewardBatch.unsqueeze(1);

        }

        protected virtual Tensor ComputeLoss(Tensor stateActionValues, Tensor expectedStateActionValues)
        {
            SmoothL1Loss criterion = torch.nn.SmoothL1Loss();
            return criterion.forward(stateActionValues, expectedStateActionValues);
        }

        protected virtual void UpdateModel(Tensor loss)
        {
            myOptimizer.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_norm_(myPolicyNet.parameters(), 100);
            myOptimizer.step();
        }

        protected virtual void UpdatePrioritizedReplayMemory(Tensor stateActionValues, Tensor detachedExpectedStateActionValues, List<int> sampledIndices)
        {
            Tensor tdErrors = (stateActionValues - detachedExpectedStateActionValues).abs();
            float[] errors = ExtractTensorData(tdErrors);

            for (int i = 0; i < sampledIndices.Count; i++)
            {
                ((PrioritizedReplayMemory<T>)myReplayBuffer).UpdatePriority(sampledIndices[i], errors[i] + 0.001f);
            }
        }

        internal float[] ExtractTensorData(Tensor tensor)
        {
            // Ensure the tensor is on CPU memory to access its data directly
            tensor = tensor.cpu();

            // TorchSharp data can be accessed directly if it's a scalar or via array for multidimensional
            float[] data = new float[tensor.NumberOfElements];
            tensor.data<float>().CopyTo(data, 0);
            return data;
        }
    }
}