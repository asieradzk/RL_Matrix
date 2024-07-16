using RLMatrix.Agents.Common;
using RLMatrix.Agents.DQN.Domain;
using RLMatrix.Dashboard;
using RLMatrix.Memories;
using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;

namespace RLMatrix.Agents.PPO.Implementations
{
    public class PPOOptimize<T> : IOptimize<T>
    {
        public PPOActorNet actorNet { get; set; }
        public PPOCriticNet criticNet { get; set; }
        public OptimizerHelper actorOptimizer { get; set; }
        public OptimizerHelper criticOptimizer { get; set; }
        public PPOAgentOptions myOptions { get; set; }
        public Device myDevice { get; set; }
        public int[] ActionSizes { get; set; }
        (float min, float max)[] continuousActionBounds { get; set; }
        LRScheduler actorLrScheduler { get; set; }
        LRScheduler criticLrScheduler { get; set; }
        public IGAIL<T>? myGAIL { get; set; }

        public PPOOptimize(PPOActorNet actorNet, PPOCriticNet criticNet, OptimizerHelper actorOptimizer, OptimizerHelper criticOptimizer,
            PPOAgentOptions options, Device myDevice, int[] ActionSizes, (float min, float max)[] continuousActionBounds, LRScheduler actorLrScheduler, LRScheduler criticLrScheduler, IGAIL<T>? myGAIL)
        {
            this.actorNet = actorNet;
            this.criticNet = criticNet;
            this.actorOptimizer = actorOptimizer;
            this.criticOptimizer = criticOptimizer;
            this.myOptions = options;
            this.myDevice = myDevice;
            this.ActionSizes = ActionSizes;
            this.myGAIL = myGAIL;
            this.actorLrScheduler = actorLrScheduler;
            this.criticLrScheduler = criticLrScheduler;
            this.continuousActionBounds = continuousActionBounds;
        }

        #region utils
        public static unsafe void CreateTensorsFromTransitions(Device device, IList<TransitionInMemory<T>> transitions, out Tensor stateBatch, out Tensor actionBatch)
        {
            int length = transitions.Count;
            var fixedDiscreteActionSize = transitions[0].discreteActions.Length;
            var fixedContinuousActionSize = 0;

            // Pre-allocate arrays based on the known batch size
            transitions.UnpackTransitionInMemory(out T[] batchStates, out int[,] batchDiscreteActions, out float[,] batchContinuousActions);
            stateBatch = Utilities<T>.StateBatchToTensor(batchStates, device);
            using (var discreteActionBatch = torch.tensor(batchDiscreteActions, torch.int64, device: device))
            using (var continuousActionBatch = torch.tensor(batchContinuousActions, device: device))
            {
                actionBatch = torch.cat(new Tensor[] { discreteActionBatch, continuousActionBatch }, dim: 1);
            }
        }
        (Tensor, Tensor) DiscountedRewardsAndAdvantages(IList<TransitionInMemory<T>> transitions, Tensor values)
        {
            var batchSize = transitions.Count();

            if (batchSize == 1)
            {
                return (torch.tensor(transitions.First().reward).to(myDevice), torch.tensor(transitions.First().reward).to(myDevice));
            }

            float[] discountedRewards = new float[batchSize];
            float[] advantages = new float[batchSize];
            float[] valueArray = Utilities<T>.ExtractTensorData(values);

            var lastTransitions = transitions.ToArray()
                .Where(t => t.nextTransition == null)
                .Select(t => t)
                .ToList();

            foreach (var lastTransition in lastTransitions)
            {
                double discountedReward = 0;
                float runningAdd = 0;
                var currentTransition = lastTransition;
                int transitionIndex = transitions.IndexOf(currentTransition);

                while (currentTransition != null)
                {
                    discountedReward = currentTransition.reward + myOptions.Gamma * discountedReward;
                    discountedRewards[transitionIndex] = (float)discountedReward;

                    float nextValue = currentTransition.nextTransition != null ? valueArray[transitionIndex + 1] : 0f;
                    float tdError = currentTransition.reward + myOptions.Gamma * nextValue - valueArray[transitionIndex];
                    runningAdd = tdError + myOptions.Gamma * myOptions.GaeLambda * runningAdd;
                    advantages[transitionIndex] = runningAdd;

                    currentTransition = currentTransition.previousTransition;
                    if (currentTransition != null)
                        transitionIndex = transitions.IndexOf(currentTransition);
                }
            }

            Tensor discountedRewardsTensor = torch.tensor(discountedRewards).to(myDevice);
            Tensor advantagesTensor = torch.tensor(advantages).to(myDevice);
            advantagesTensor = (advantagesTensor - advantagesTensor.mean()) / (advantagesTensor.std() + 1e-10);


            return (discountedRewardsTensor, advantagesTensor);
        }

        #region rnn
        private static List<List<TransitionInMemory<T>>> PadTransitions(IList<TransitionInMemory<T>> transitions, Device device, out Tensor mask)
        {
            int length = transitions.Count;
            var fixedDiscreteActionSize = transitions[0].discreteActions.Length;
            var fixedContinuousActionSize = transitions[0].continuousActions.Length;
            var firstTransitions = transitions.ToArray()
                .Where(t => t.previousTransition == null)
                .Select(t => t)
                .ToList();

            Dictionary<TransitionInMemory<T>, int> sequenceLengths = new Dictionary<TransitionInMemory<T>, int>();
            foreach (var transition in firstTransitions)
            {
                int sequenceLength = CalculateSequenceLength(transition);
                sequenceLengths.Add(transition, sequenceLength);
            }

            var longestSequence = sequenceLengths.Values.Max();

            List<List<TransitionInMemory<T>>> paddedSequences = new List<List<TransitionInMemory<T>>>();
            foreach (var transition in firstTransitions)
            {
                List<TransitionInMemory<T>> paddedSequence = new List<TransitionInMemory<T>>();
                AddInitialSequence(transition, paddedSequence);
                PadSequence(paddedSequence[paddedSequence.Count - 1], longestSequence, sequenceLengths[transition], paddedSequence);
                paddedSequences.Add(paddedSequence);
            }


            #region mask
            mask = CreateMask(paddedSequences, sequenceLengths, device);
            return paddedSequences;

            Tensor CreateMask(List<List<TransitionInMemory<T>>> paddedSequences, Dictionary<TransitionInMemory<T>, int> sequenceLengths, Device device)
            {
                List<Tensor> maskList = new List<Tensor>();
                foreach (var sequence in paddedSequences)
                {
                    var originalLength = sequenceLengths[sequence[0]];
                    var paddedLength = sequence.Count;
                    bool[] maskData = Enumerable.Repeat(true, originalLength)
                                                .Concat(Enumerable.Repeat(false, paddedLength - originalLength))
                                                .ToArray();
                    maskList.Add(torch.tensor(maskData)); //.unsqueeze
                }

                var mask = torch.cat(maskList.ToArray(), dim: 0).to(device);
                return mask;

            }
            #endregion


            int CalculateSequenceLength(TransitionInMemory<T> transition)
            {
                if (transition.nextTransition == null)
                    return 1;

                return 1 + CalculateSequenceLength(transition.nextTransition);
            }

            void AddInitialSequence(TransitionInMemory<T> transition, List<TransitionInMemory<T>> paddedSequence)
            {
                while (transition != null)
                {
                    paddedSequence.Add(transition);
                    transition = transition.nextTransition;
                }
            }

            static TransitionInMemory<T> PadSequence(TransitionInMemory<T> transition, int targetLength, int currentLength, List<TransitionInMemory<T>> paddedSequence)
            {

                if (currentLength >= targetLength)
                    return transition;

                var paddedTransition = new TransitionInMemory<T>(
                    transition.state,
                    transition.discreteActions,
                    transition.continuousActions,
                    0f,
                    transition.state,
                    null,
                    transition
                );

                transition.nextTransition = paddedTransition;
                paddedSequence.Add(paddedTransition);

                var result = PadSequence(paddedTransition, targetLength, currentLength + 1, paddedSequence);
                return result;
            }
        }

        //TODO: probablt overkill with masks. Maybe need to check later where its redundant.
        void OptimizeModelRNN(IMemory<T> myReplayBuffer)
        {
            using (var scope = torch.NewDisposeScope())
            {
                var paddedTransitions = PadTransitions(myReplayBuffer.SampleEntireMemory(), myDevice, out var mask);

                List<Tensor> stateBatches = new List<Tensor>();
                List<Tensor> actionBatches = new List<Tensor>();

                foreach (var sequence in paddedTransitions)
                {
                    CreateTensorsFromTransitions(myDevice, sequence, out var stateBatchEpisode, out var actionBatchEpisode);
                    stateBatches.Add(stateBatchEpisode);
                    actionBatches.Add(actionBatchEpisode);
                }

                Tensor stateBatch = torch.stack(stateBatches.ToArray(), dim: 0);
                Tensor actionBatch = torch.cat(actionBatches.ToArray(), dim: 0);

                Tensor policyOld = actorNet.get_log_prob(stateBatch, actionBatch, ActionSizes.Count(), continuousActionBounds.Count()).detach();
                
                Tensor valueOld = criticNet.forward(stateBatch).detach().squeeze(1);

                //Tensor maskedPolicyOld = torch.masked_select(policyOld, mask.to_type(ScalarType.Bool));
                Tensor maskedPolicyOld = MaskedSelectBatch(policyOld, mask.to_type(ScalarType.Bool));
                Tensor maskedValueOld = torch.masked_select(valueOld, mask.to_type(ScalarType.Bool));

                List<Tensor> discountedRewardsList = new List<Tensor>();
                List<Tensor> advantagesList = new List<Tensor>();

                List<TransitionInMemory<T>> paddedTransitionsSummed = paddedTransitions.SelectMany(t => t).ToList();
                var (maskedDiscountedRewards, maskedAdvantages) = DiscountedRewardsAndAdvantages(paddedTransitionsSummed, valueOld);
                maskedDiscountedRewards = torch.masked_select(maskedDiscountedRewards, mask.to_type(ScalarType.Bool));
                maskedAdvantages = torch.masked_select(maskedAdvantages, mask.to_type(ScalarType.Bool));
                maskedAdvantages = ReshapeAdvantages(maskedAdvantages, maskedPolicyOld);

                for (int i = 0; i < myOptions.PPOEpochs; i++)
                {
                    using (var actorScope = torch.NewDisposeScope())
                    {
                        (Tensor policy, Tensor entropy) = actorNet.get_log_prob_entropy(stateBatch, actionBatch, ActionSizes.Count(), continuousActionBounds.Count());
                        //policy = torch.masked_select(policy, mask.to_type(ScalarType.Bool));
                        policy = MaskedSelectBatch(policy, mask.to_type(ScalarType.Bool));
                        entropy = torch.masked_select(entropy, mask.to_type(ScalarType.Bool));
                        Tensor ratios = torch.exp(policy - maskedPolicyOld);
                        Tensor surr1 = ratios * maskedAdvantages;
                        Tensor surr2 = torch.clamp(ratios, 1.0 - myOptions.ClipEpsilon, 1.0 + myOptions.ClipEpsilon) * maskedAdvantages;
                        // Select the non-masked surrogate values
                        Tensor surr = torch.min(surr1, surr2);

                        // Select the non-masked entropy values
                        Tensor maskedEntropy = entropy;

                        // Calculate the mean of the non-masked surrogate and entropy values
                        Tensor actorLoss = -surr.mean() - myOptions.EntropyCoefficient * maskedEntropy.mean();
                        actorOptimizer.zero_grad();
                        actorLoss.backward();
                        torch.nn.utils.clip_grad_norm_(actorNet.parameters(), myOptions.ClipGradNorm);
                        actorOptimizer.step();
                        if(i == 0)
                        {

                            DashboardProvider.Instance.UpdateEntropy((double)maskedEntropy.mean().item<float>());
                            DashboardProvider.Instance.UpdateActorLoss((double)actorLoss.item<float>());
                            DashboardProvider.Instance.UpdateActorLearningRate(actorLrScheduler.get_last_lr().FirstOrDefault());
                        }
                        
                    }

                    using (var criticScope = torch.NewDisposeScope())
                    {
                        Tensor values = criticNet.forward(stateBatch).squeeze(1);
                        values = torch.masked_select(values, mask.to_type(ScalarType.Bool));
                        Tensor valueClipped = maskedValueOld + torch.clamp(values - maskedValueOld, -myOptions.VClipRange, myOptions.VClipRange);
                        Tensor valueLoss1 = torch.pow(values - maskedDiscountedRewards, 2);
                        Tensor valueLoss2 = torch.pow(valueClipped - maskedDiscountedRewards, 2);

                        // Select the non-masked loss values
                        Tensor valueLoss = torch.max(valueLoss1, valueLoss2);

                        // Calculate the mean of the non-masked loss values
                        Tensor criticLoss = myOptions.CValue * valueLoss.mean();

                        criticOptimizer.zero_grad();
                        criticLoss.backward();
                        torch.nn.utils.clip_grad_norm_(criticNet.parameters(), myOptions.ClipGradNorm);
                        criticOptimizer.step();
                        if (i == 0)
                        {
                            DashboardProvider.Instance.UpdateCriticLoss((double)criticLoss.item<float>());
                            DashboardProvider.Instance.UpdateCriticLearningRate(criticLrScheduler.get_last_lr().FirstOrDefault());
                        }
                    }
                }
            }

           
            myReplayBuffer.ClearMemory();
        }


        //TODO: move to utils?
        private Tensor MaskedSelectBatch(Tensor tensor, Tensor mask)
        {
            if (tensor.shape[0] != mask.shape[0])
            {
                throw new ArgumentException("Tensor and mask must have the same batch size (first dimension)");
            }

            // Ensure mask is boolean
            mask = mask.to_type(ScalarType.Bool);

            // Reshape tensor to [batch_size, -1]
            long flattenedSize = tensor.shape.Skip(1).Aggregate(1L, (a, b) => a * b);
            var reshaped = tensor.view(tensor.shape[0], flattenedSize);

            // Apply mask
            var maskedFlat = torch.masked_select(reshaped, mask.unsqueeze(-1));

            // Reshape back to original shape minus the masked-out batch elements
            long newBatchSize = mask.sum().item<long>();
            var newShape = new long[] { newBatchSize }.Concat(tensor.shape.Skip(1)).ToArray();
            return maskedFlat.view(newShape);
        }

        private Tensor ReshapeAdvantages(Tensor advantages, Tensor policyTensor)
        {
            if (policyTensor.dim() == 1)
                return advantages;

            // Ensure advantages is 2D
            if (advantages.dim() == 1)
                advantages = advantages.unsqueeze(1);

            // Expand advantages to match policy shape in first two dimensions
            return advantages.expand(policyTensor.shape[0], policyTensor.shape[1]);
        }
        #endregion

        #endregion

        private static torch.nn.utils.rnn.PackedSequence CreatePackedSequence(IList<TransitionInMemory<T>> transitions, Device device)
        {
            var firstTransitions = transitions.ToArray()
                .Where(t => t.previousTransition == null)
                .Select(t => t)
                .ToList();

            List<torch.Tensor> sequenceTensors = new List<torch.Tensor>();

            foreach (var transition in firstTransitions)
            {
                List<T> sequenceStates = new List<T>();
                var currentTransition = transition;

                while (currentTransition != null)
                {
                    sequenceStates.Add(currentTransition.state);
                    currentTransition = currentTransition.nextTransition;
                }

                var sequenceTensor = Utilities<T>.StateBatchToTensor(sequenceStates.ToArray(), device);
                sequenceTensors.Add(sequenceTensor);
            }

            return torch.nn.utils.rnn.pack_sequence(sequenceTensors, false);
        }


        //This could potentially have better performance but due to some error doesnt want to learn :) 
        void OptimizeRNNPacked(IMemory<T> replayBuffer)
        {
            
            using (var scope = torch.NewDisposeScope())
            {
                var transitions = replayBuffer.SampleEntireMemory();
                CreateTensorsFromTransitions(myDevice, transitions, out var stateBatch, out var actionBatch);
                var packedTransition = CreatePackedSequence(transitions, myDevice);

                Tensor policyOld = actorNet.get_log_prob(packedTransition, actionBatch, ActionSizes.Count(), continuousActionBounds.Count()).detach();
                Tensor valueOld = criticNet.forward(packedTransition).detach().squeeze(1);

                (var discountedRewards, var advantages) = DiscountedRewardsAndAdvantages(transitions, valueOld);

                if (policyOld.dim() > 1)
                {
                    advantages = advantages.unsqueeze(1);
                }

                for (int i = 0; i < myOptions.PPOEpochs; i++)
                {
                    using (var actorScope = torch.NewDisposeScope())
                    {

                        (Tensor policy, Tensor entropy) = actorNet.get_log_prob_entropy(packedTransition, actionBatch, ActionSizes.Count(), continuousActionBounds.Count());
                        Tensor ratios = torch.exp(policy - policyOld);
                        Tensor surr1 = ratios * advantages;
                        Tensor surr2 = torch.clamp(ratios, 1.0 - myOptions.ClipEpsilon, 1.0 + myOptions.ClipEpsilon) * advantages;
                        Tensor actorLoss = -torch.min(surr1, surr2).mean() - myOptions.EntropyCoefficient * entropy.mean();
                        actorOptimizer.zero_grad();
                        actorLoss.backward();
                        torch.nn.utils.clip_grad_norm_(actorNet.parameters(), myOptions.ClipGradNorm);
                        actorOptimizer.step();
                    }

                    using (var criticScope = torch.NewDisposeScope())
                    {
                        Tensor values = criticNet.forward(packedTransition).squeeze(1);
                        Tensor valueClipped = valueOld + torch.clamp(values - valueOld, -myOptions.VClipRange, myOptions.VClipRange);
                        Tensor valueLoss1 = torch.pow(values - discountedRewards, 2);
                        Tensor valueLoss2 = torch.pow(valueClipped - discountedRewards, 2);
                        Tensor criticLoss = myOptions.CValue * torch.max(valueLoss1, valueLoss2).mean();

                        criticOptimizer.zero_grad();
                        criticLoss.backward();
                        torch.nn.utils.clip_grad_norm_(criticNet.parameters(), myOptions.ClipGradNorm);
                        criticOptimizer.step();
                    }
                }
            }

            replayBuffer.ClearMemory();
        }


       public void Optimize(IMemory<T> replayBuffer)
        {
            if (myOptions.UseRNN && myOptions.BatchSize > 1)
            {
                // throw new ArgumentException("Batch size larger than 1 is not yet supported with RNN");
            }

            if (myGAIL != null && replayBuffer.Length > 0)
            {
                myGAIL.OptimiseDiscriminator(replayBuffer);
            }

            if (replayBuffer.NumEpisodes < myOptions.BatchSize)
                return;

            if (myOptions.UseRNN)
            {
                // (padding + masking)
                OptimizeModelRNN(replayBuffer);

                actorLrScheduler.step();
                criticLrScheduler.step();

                //(no masking version?)
                // OptimizeRNNPacked(replayBuffer); //broken
                return;
            }

            using (var scope = torch.NewDisposeScope())
            {
                var transitions = replayBuffer.SampleEntireMemory();
                CreateTensorsFromTransitions(myDevice, transitions, out var stateBatch, out var actionBatch);

                using (var policyOld = actorNet.get_log_prob(stateBatch, actionBatch, ActionSizes.Count(), continuousActionBounds.Count()).detach())
                using (var valueOld = criticNet.forward(stateBatch).detach())
                {
                    (var discountedRewards, var advantages) = DiscountedRewardsAndAdvantages(transitions, valueOld);

                    if (policyOld.dim() > 1)
                    {
                        advantages = advantages.unsqueeze(1);
                    }

                    for (int i = 0; i < myOptions.PPOEpochs; i++)
                    {
                        using (var actorScope = torch.NewDisposeScope())
                        {
                            (Tensor policy, Tensor entropy) = actorNet.get_log_prob_entropy(stateBatch, actionBatch, ActionSizes.Count(), continuousActionBounds.Count());
                            using (var ratios = torch.exp(policy - policyOld))
                            using (var surr1 = ratios * advantages)
                            using (var surr2 = torch.clamp(ratios, 1.0 - myOptions.ClipEpsilon, 1.0 + myOptions.ClipEpsilon) * advantages)
                            {
                                var actorLoss = -torch.min(surr1, surr2).mean() - myOptions.EntropyCoefficient * entropy.mean();
                                //actorLoss.print();
                                actorOptimizer.zero_grad();
                                actorLoss.backward();
                                torch.nn.utils.clip_grad_norm_(actorNet.parameters(), myOptions.ClipGradNorm);
                                actorOptimizer.step();


                                if(i == 0)
                                {
                                    Tensor klDivergence = (policyOld.exp() * (policyOld - policy)).mean();
                                    DashboardProvider.Instance.UpdateKLDivergence((double)klDivergence.item<float>());

                                   // DashboardProvider.Instance.UpdateKLDivergence((double)klDivergence.item<float>());


                                    DashboardProvider.Instance.UpdateKLDivergence((double)klDivergence.item<float>());
                                    DashboardProvider.Instance.UpdateEntropy((double)entropy.mean().item<float>());
                                    DashboardProvider.Instance.UpdateActorLoss((double)actorLoss.item<float>());
                                    DashboardProvider.Instance.UpdateActorLearningRate(actorLrScheduler.get_last_lr().FirstOrDefault());
                                }                                
                            }
                        }

                        using (var criticScope = torch.NewDisposeScope())
                        {
                            using (var values = criticNet.forward(stateBatch))
                            using (var valueClipped = valueOld + torch.clamp(values - valueOld, -myOptions.VClipRange, myOptions.VClipRange))
                            using (var valueLoss1 = torch.pow(values - discountedRewards, 2))
                            using (var valueLoss2 = torch.pow(valueClipped - discountedRewards, 2))
                            {
                                var criticLoss = myOptions.CValue * torch.max(valueLoss1, valueLoss2).mean();
                                criticOptimizer.zero_grad();
                                criticLoss.backward();
                                torch.nn.utils.clip_grad_norm_(criticNet.parameters(), myOptions.ClipGradNorm);
                                criticOptimizer.step();

                                if (i == 0)
                                {
                                    DashboardProvider.Instance.UpdateCriticLoss((double)criticLoss.item<float>());
                                    DashboardProvider.Instance.UpdateCriticLearningRate(actorLrScheduler.get_last_lr().FirstOrDefault());
                                }
                            }
                        }
                    }
                }
            }

            // TODO: Didn't check that default scheduler doesn't degrade training :)
            actorLrScheduler.step();
            criticLrScheduler.step();

            replayBuffer.ClearMemory();
        }

        public void UpdateOptimizers(LRScheduler scheduler)
        {
            //TODO: SEIROUS violation of DRY. Default Optimizer implementation should be moved to some kind of provider
            actorOptimizer = torch.optim.Adam(actorNet.parameters(), myOptions.LR, amsgrad: true);
            scheduler ??= new optim.lr_scheduler.impl.CyclicLR(actorOptimizer, myOptions.LR * 0.5f, myOptions.LR * 2f, step_size_up: 10, step_size_down: 10, cycle_momentum: false);
            actorLrScheduler = scheduler;

            criticOptimizer = torch.optim.Adam(criticNet.parameters(), myOptions.LR, amsgrad: true);
            scheduler ??= new optim.lr_scheduler.impl.CyclicLR(criticOptimizer, myOptions.LR * 0.5f, myOptions.LR * 2f, step_size_up: 10, step_size_down: 10, cycle_momentum: false);
            criticLrScheduler = scheduler;
        }
    }
}


    

