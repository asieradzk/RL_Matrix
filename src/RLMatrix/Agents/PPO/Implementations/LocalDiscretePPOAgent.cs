using OneOf;
using RLMatrix.Agents.Common;
using RLMatrix.Agents.DQN.Domain;
using RLMatrix.Memories;
using Tensorboard;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;

namespace RLMatrix.Agents.PPO.Implementations
{
    public class LocalDiscretePPOAgent<T> : IDiscreteProxy<T>
    {
        private readonly IDiscretePPOAgent<T> _agent;
        bool useRnn = false;
        private Dictionary<Guid, Tensor?> memoriesStore = new();

        //TODO: Composer param
        public LocalDiscretePPOAgent(PPOAgentOptions options, int[] ActionSizes, OneOf<int, (int, int)> StateSizes /*, IDiscretePPOAgentCOmposer<T> agentComposer = null*/)
        {
            _agent = PPOAgentFactory<T>.ComposeDiscretePPOAgent(options, ActionSizes, StateSizes);
            useRnn = options.UseRNN;           
        }

        public ValueTask OptimizeModelAsync()
        {
            _agent.OptimizeModel();
            return ValueTask.CompletedTask;
        }
       
        public ValueTask<Dictionary<Guid, int[]>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos)
        {
            

            if (useRnn)
            {
                Dictionary<Guid, int[]> actionDict = new Dictionary<Guid, int[]>();
                if (memoriesStore.Count == 0)
                {
                    foreach (var stateInfo in stateInfos)
                    {
                        memoriesStore[stateInfo.environmentId] = null;
                    }
                }

                (T state, Tensor? memoryState)[] statesWithMemory = stateInfos.Select(info => (info.state, memoriesStore[info.environmentId])).ToArray();

                (int[] actions, Tensor? memoryState)[] actionsWithMemory = _agent.SelectActionsRecurrent(statesWithMemory, isTraining: true);

                for (int i = 0; i < stateInfos.Count; i++)
                {
                    Guid environmentId = stateInfos[i].environmentId;
                    int[] action = actionsWithMemory[i].actions;
                    memoriesStore[environmentId] = actionsWithMemory[i].memoryState;
                    actionDict[environmentId] = action;
                }

                return ValueTask.FromResult(actionDict);
            }else
            {

                // Extract the states from the stateInfos list
                T[] states = stateInfos.Select(info => info.state).ToArray();

                // Select actions for the batch of states
                int[][] actions = _agent.SelectActions(states, isTraining: true);

                // Create a dictionary to map environment IDs to their corresponding actions
                Dictionary<Guid, int[]> actionDict = new Dictionary<Guid, int[]>();

                // Iterate over the stateInfos and populate the actionDict
                for (int i = 0; i < stateInfos.Count; i++)
                {
                    Guid environmentId = stateInfos[i].environmentId;
                    int[] action = actions[i];
                    actionDict[environmentId] = action;
                }

                return ValueTask.FromResult(actionDict);
            }


        }

        public ValueTask UploadTransitionsAsync(IEnumerable<TransitionPortable<T>> transitions)
        {
            _agent.AddTransition(transitions);
            return ValueTask.CompletedTask;
        }
    }

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
                Tensor valueOld = criticNet.forward(stateBatch).detach();
                Tensor maskedPolicyOld = policyOld * mask;
                Tensor maskedValueOld = valueOld * mask;

                List<Tensor> discountedRewardsList = new List<Tensor>();
                List<Tensor> advantagesList = new List<Tensor>();

                List<TransitionInMemory<T>> paddedTransitionsSummed = paddedTransitions.SelectMany(t => t).ToList();
                var (maskedDiscountedRewards, maskedAdvantages) = DiscountedRewardsAndAdvantages(paddedTransitionsSummed, maskedValueOld);
                maskedDiscountedRewards *= mask;
                maskedAdvantages *= mask;


                for (int i = 0; i < myOptions.PPOEpochs; i++)
                {
                    using (var actorScope = torch.NewDisposeScope())
                    {
                        (Tensor policy, Tensor entropy) = actorNet.get_log_prob_entropy(stateBatch, actionBatch, ActionSizes.Count(), continuousActionBounds.Count());
                        policy *= mask;  // Apply mask to policy
                                         //  entropy *= mask;  // Apply mask to entropy

                        Tensor ratios = torch.exp(policy - maskedPolicyOld);
                        Tensor surr1 = ratios * maskedAdvantages;
                        Tensor surr2 = torch.clamp(ratios, 1.0 - myOptions.ClipEpsilon, 1.0 + myOptions.ClipEpsilon) * maskedAdvantages;

                        // Select the non-masked surrogate values
                        Tensor surr = torch.masked_select(torch.min(surr1, surr2), mask.to_type(ScalarType.Bool));

                        // Select the non-masked entropy values
                        Tensor maskedEntropy = torch.masked_select(entropy, mask.to_type(ScalarType.Bool));

                        // Calculate the mean of the non-masked surrogate and entropy values
                        Tensor actorLoss = -surr.mean() - myOptions.EntropyCoefficient * maskedEntropy.mean();

                        actorOptimizer.zero_grad();
                        actorLoss.backward();
                        torch.nn.utils.clip_grad_norm_(actorNet.parameters(), myOptions.ClipGradNorm);
                        actorOptimizer.step();
                    }

                    using (var criticScope = torch.NewDisposeScope())
                    {
                        Tensor values = criticNet.forward(stateBatch);
                        values *= mask;  // Apply mask to values
                        Tensor valueClipped = maskedValueOld + torch.clamp(values - maskedValueOld, -myOptions.VClipRange, myOptions.VClipRange);
                        Tensor valueLoss1 = torch.pow(values - maskedDiscountedRewards, 2);
                        Tensor valueLoss2 = torch.pow(valueClipped - maskedDiscountedRewards, 2);

                        // Select the non-masked loss values
                        Tensor valueLoss = torch.masked_select(torch.max(valueLoss1, valueLoss2), mask.to_type(ScalarType.Bool));

                        // Calculate the mean of the non-masked loss values
                        Tensor criticLoss = myOptions.CValue * valueLoss.mean();

                        criticOptimizer.zero_grad();
                        criticLoss.backward();
                        torch.nn.utils.clip_grad_norm_(criticNet.parameters(), myOptions.ClipGradNorm);
                        criticOptimizer.step();
                    }
                }
            }

            myReplayBuffer.ClearMemory();
        }


        #endregion

        #endregion



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
                OptimizeModelRNN(replayBuffer);
                return;
            }



            using (var scope = torch.NewDisposeScope())
            {
                var transitions = replayBuffer.SampleEntireMemory();
                CreateTensorsFromTransitions(myDevice, transitions, out var stateBatch, out var actionBatch);

                Tensor policyOld = actorNet.get_log_prob(stateBatch, actionBatch, ActionSizes.Count(), continuousActionBounds.Count()).detach();
                Tensor valueOld = criticNet.forward(stateBatch).detach();

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
                        Tensor values = criticNet.forward(stateBatch);
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
            //TODO: Didnt check that default scheduler doesn't degrade training :)
            actorLrScheduler.step();
            criticLrScheduler.step();

            replayBuffer.ClearMemory();
        }
    }
}


    

