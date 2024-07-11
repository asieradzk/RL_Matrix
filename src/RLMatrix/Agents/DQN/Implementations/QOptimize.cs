using RLMatrix.Agents.Common;
using RLMatrix.Dashboard;
using RLMatrix.Memories;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim.lr_scheduler;

namespace RLMatrix.Agents.DQN.Domain
{
    public class QOptimize<T> : IOptimize<T>
    {
        public Module<Tensor, Tensor> PolicyNet;
        public Module<Tensor, Tensor> TargetNet;
        public OptimizerHelper Optimizer;
        public LRScheduler LRScheduler;
        public IGAIL<T>? myGAIL;
        public IComputeQValues QValuesCalculator;
        public IExtractStateActionValues StateActionValueExtractor;
        public IComputeNextStateValues NextStateValuesCalculator;
        public IComputeExpectedStateActionValues<T> ExpectedStateActionValuesCalculator;
        public IComputeLoss LossCalculator;
        public DQNAgentOptions myOptions;
        public Device myDevice;
        public int[] ActionSizes;

        public QOptimize(
         Module<Tensor, Tensor> policyNet,
         Module<Tensor, Tensor> targetNet,
         OptimizerHelper optimizer,
         IComputeQValues qvaluesCalculator,
         IExtractStateActionValues stateActionValueExtractor,
         IComputeNextStateValues nextStateValuesCalculator,
         IComputeExpectedStateActionValues<T> expectedStateActionValuesCalculator,
         IComputeLoss lossCalculator,
         DQNAgentOptions options,
         Device device,
         int[] actionSizes,
         LRScheduler lrScheduler = null,
         IGAIL<T> GAIL = null)
        {
            PolicyNet = policyNet;
            TargetNet = targetNet;
            Optimizer = optimizer;
            QValuesCalculator = qvaluesCalculator;
            StateActionValueExtractor = stateActionValueExtractor;
            NextStateValuesCalculator = nextStateValuesCalculator;
            ExpectedStateActionValuesCalculator = expectedStateActionValuesCalculator;
            LossCalculator = lossCalculator;
            myOptions = options;
            myDevice = device;
            ActionSizes = actionSizes;
            LRScheduler = lrScheduler;
            myGAIL = GAIL;

        }

        public void UpdateOptimizers(LRScheduler scheduler)
        {
            Optimizer = torch.optim.Adam(PolicyNet.parameters(), lr: myOptions.LR, amsgrad: true);
            scheduler ??= new optim.lr_scheduler.impl.CyclicLR(Optimizer, myOptions.LR * 0.5f, myOptions.LR * 2f, step_size_up: 500, step_size_down: 2000, cycle_momentum: false);
            LRScheduler = scheduler;
        
        }

        public void Optimize(IMemory<T> ReplayBuffer)
        {
            using (var OptimizeScope = torch.NewDisposeScope())
            {
                if (myGAIL != null && ReplayBuffer.Length > 0)
                {
                    myGAIL.OptimiseDiscriminator(ReplayBuffer);
                }

                if (ReplayBuffer.Length < myOptions.BatchSize)
                    return;

                var transitions = ReplayBuffer.Sample(myOptions.BatchSize);
                Utilities<T>.CreateTensorsFromTransitions(ref myDevice, transitions, out Tensor nonFinalMask, out Tensor stateBatch, out Tensor nonFinalNextStates, out Tensor actionBatch, out Tensor rewardBatch);
                if (myGAIL != null)
                {
                    myGAIL.OptimiseDiscriminator(ReplayBuffer);
                    rewardBatch = myGAIL.AugmentRewardBatch(stateBatch, actionBatch, rewardBatch);
                }
                Tensor qValuesAllHeads = QValuesCalculator.ComputeQValues(stateBatch, PolicyNet);
                Tensor stateActionValues = StateActionValueExtractor.ExtractStateActionValues(qValuesAllHeads, actionBatch);
                Tensor nextStateValues = NextStateValuesCalculator.ComputeNextStateValues(nonFinalNextStates, TargetNet, PolicyNet, myOptions, ActionSizes, myDevice);
                Tensor expectedStateActionValues = ExpectedStateActionValuesCalculator.ComputeExpectedStateActionValues(nextStateValues, rewardBatch, nonFinalMask, myOptions, transitions, ActionSizes, myDevice);

                Tensor loss = LossCalculator.ComputeLoss(stateActionValues, expectedStateActionValues);
                UpdateModel(loss);
                LRScheduler.step();

                DashboardProvider.Instance.UpdateLoss((double)loss.item<float>());
                DashboardProvider.Instance.UpdateLearningRate(LRScheduler.get_last_lr().FirstOrDefault());

                if (ReplayBuffer is PrioritizedReplayMemory<T> prioritizedReplayBuffer)
                {
                    var sampledIncides = prioritizedReplayBuffer.GetSampledIndices();
                    UpdatePrioritizedReplayMemory(prioritizedReplayBuffer, stateActionValues, expectedStateActionValues.detach(), sampledIncides);
                }
            }


        }
        protected virtual void UpdatePrioritizedReplayMemory(PrioritizedReplayMemory<T> prioritizedReplayBuffer, Tensor stateActionValues, Tensor detachedExpectedStateActionValues, Span<int> sampledIndices)
        {
            Tensor tdErrors = (stateActionValues - detachedExpectedStateActionValues).abs();
            float[] errors = Utilities<T>.ExtractTensorData(tdErrors);

            for (int i = 0; i < sampledIndices.Length; i++)
            {
                (prioritizedReplayBuffer).UpdatePriority(sampledIndices[i], errors[i] + 0.001f);
            }
            
        }

        int updateCounter = 0;
        private void UpdateModel(Tensor loss)
        {
            Optimizer.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_value_(PolicyNet.parameters(), 100);
            Optimizer.step();
            updateCounter++;
            if (updateCounter >= myOptions.SoftUpdateInterval)
            {
                SoftUpdateTargetNetwork();
                updateCounter = 0;
            }
        }

        /// <summary>
        /// Performs an optimized soft update of the target network parameters using vectorized operations.
        /// </summary>
        private void SoftUpdateTargetNetwork()
        {
            using (torch.no_grad())
            {
                var targetParams = TargetNet.parameters().ToList();
                var policyParams = PolicyNet.parameters().ToList();

                // Combine all parameters into single tensors
                var targetTensor = torch.cat(targetParams.Select(p => p.flatten()).ToList());
                var policyTensor = torch.cat(policyParams.Select(p => p.flatten()).ToList());

                // Perform the soft update in a single operation
                targetTensor.mul_(1 - myOptions.TAU).add_(policyTensor, alpha: myOptions.TAU);

                // Update the target network parameters
                int index = 0;
                foreach (var param in targetParams)
                {
                    var flatSize = (int)param.numel();
                    param.copy_(targetTensor.narrow(0, index, flatSize).view_as(param));
                    index += flatSize;
                }
            }
        }

    }
}