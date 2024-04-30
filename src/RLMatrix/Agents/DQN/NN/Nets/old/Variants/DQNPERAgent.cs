using OneOf;
using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace RLMatrix
{
    /// <summary>
    /// Represents a Deep Q-Learning agent with Prioritized Experience Replay (PER).
    /// </summary>
    /// <typeparam name="T">The type of the state representation, either float[] or float[,].</typeparam>
    public class DQNAgentPER<T> : DQNAgent<T>
    {
        public DQNAgentPER(DQNAgentOptions opts, List<IEnvironment<T>> envs, IDQNNetProvider<T> netProvider = null)
            : base(opts, envs, netProvider)
        {
            myReplayBuffer = new PrioritizedReplayMemory<T>(myOptions.MemorySize, myOptions.BatchSize);
        }

        public override unsafe void OptimizeModel()
        {
            if (myReplayBuffer.Length < myOptions.BatchSize)
                return;

            ReadOnlySpan<TransitionInMemory<T>> transitions = myReplayBuffer.Sample();
            ReadOnlySpan<int> sampledIndices = null;

            if (myReplayBuffer is PrioritizedReplayMemory<T> prioritizedReplayBuffer)
            {
                sampledIndices = prioritizedReplayBuffer.GetSampledIndices();
            }

            CreateTensorsFromTransitions(ref transitions, out Tensor nonFinalMask, out Tensor stateBatch, out Tensor nonFinalNextStates, out Tensor actionBatch, out Tensor rewardBatch);

            Tensor qValuesAllHeads = ComputeQValues(stateBatch);
            Tensor stateActionValues = ExtractStateActionValues(qValuesAllHeads, actionBatch);

            Tensor nextStateValues;
            using (no_grad())
            {
                if (nonFinalNextStates.shape[0] > 0)
                {
                    nextStateValues = myTargetNet.forward(nonFinalNextStates).max(2).values;
                }
                else
                {
                    nextStateValues = zeros(new long[] { myOptions.BatchSize, myEnvironments[0].actionSize.Count() }).to(myDevice);
                }
            }

            Tensor expectedStateActionValues = ComputeExpectedStateActionValues(nextStateValues, rewardBatch, nonFinalMask);
            Tensor loss = ComputeLoss(stateActionValues, expectedStateActionValues);
            UpdateModel(loss);

            if (sampledIndices != null)
            {
                UpdatePrioritizedReplayMemory(stateActionValues, expectedStateActionValues.detach(), sampledIndices.ToArray().ToList());
            }
        }

        public void CreateTensorsFromTransitions(ref ReadOnlySpan<TransitionInMemory<T>> transitions, out Tensor nonFinalMask, out Tensor stateBatch, out Tensor nonFinalNextStates, out Tensor actionBatch, out Tensor rewardBatch)
        {
            int length = transitions.Length;
            var fixedActionSize = myEnvironments[0].actionSize.Length;

            bool[] nonFinalMaskArray = new bool[length];
            float[] batchRewards = new float[length];
            int[] flatMultiActions = new int[length * fixedActionSize];
            T[] batchStates = new T[length];
            T?[] batchNextStates = new T?[length];

            int flatMultiActionsIndex = 0;
            int nonFinalNextStatesCount = 0;

            for (int i = 0; i < length; i++)
            {
                var transition = transitions[i];
                nonFinalMaskArray[i] = transition.nextState != null;
                batchRewards[i] = transition.reward;
                Array.Copy(transition.discreteActions, 0, flatMultiActions, flatMultiActionsIndex, transition.discreteActions.Length);
                flatMultiActionsIndex += transition.discreteActions.Length;

                batchStates[i] = transition.state;
                batchNextStates[i] = transition.nextState;

                if (transition.nextState != null)
                {
                    nonFinalNextStatesCount++;
                }
            }

            stateBatch = StateToTensor(batchStates, myDevice);
            nonFinalMask = torch.tensor(nonFinalMaskArray, device: myDevice);
            rewardBatch = torch.tensor(batchRewards, device: myDevice);
            actionBatch = torch.tensor(flatMultiActions, new long[] { length, fixedActionSize }, torch.int64).to(myDevice);

            if (nonFinalNextStatesCount > 0)
            {
                T[] nonFinalNextStatesArray = new T[nonFinalNextStatesCount];
                int index = 0;
                for (int i = 0; i < length; i++)
                {
                    if (batchNextStates[i] is not null)
                    {
                        nonFinalNextStatesArray[index++] = batchNextStates[i];
                    }
                }
                nonFinalNextStates = StateToTensor(nonFinalNextStatesArray, myDevice);
            }
            else
            {
                nonFinalNextStates = torch.zeros(new long[] { 1, stateBatch.shape[1] }, device: myDevice);
            }
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
            float[] ExtractTensorData(Tensor tensor)
            {
                tensor = tensor.cpu();

                float[] data = new float[tensor.NumberOfElements];
                tensor.data<float>().CopyTo(data, 0);
                return data;
            }
        }

       
    }
}