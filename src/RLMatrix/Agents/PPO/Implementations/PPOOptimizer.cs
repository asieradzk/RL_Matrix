using RLMatrix.Common;

namespace RLMatrix;

public class PPOOptimizer<TState> : IOptimizer<TState>
    where TState : notnull
{
    public PPOOptimizer(PPOActorNet actorNet, PPOCriticNet criticNet, OptimizerHelper actorOptimizerHelper, OptimizerHelper criticOptimizerHelper, 
        PPOAgentOptions options, Device device, int[] discreteActionDimensions, ContinuousActionDimensions[] continuousActionDimensions, 
        LRScheduler actorScheduler, LRScheduler criticScheduler, IGAIL<TState>? gail)
    {
        ActorNet = actorNet;
        CriticNet = criticNet;
        ActorOptimizerHelper = actorOptimizerHelper;
        CriticOptimizerHelper = criticOptimizerHelper;
        Options = options;
        Device = device;
        DiscreteActionDimensions = discreteActionDimensions;
        ContinuousActionDimensions = continuousActionDimensions;
        ActorScheduler = actorScheduler;
        CriticScheduler = criticScheduler;
        GAIL = gail;
    }

    public PPOActorNet ActorNet { get; }
    
    public PPOCriticNet CriticNet { get; }
    
    public OptimizerHelper ActorOptimizerHelper { get; private set; }
    
    public OptimizerHelper CriticOptimizerHelper { get; private set; }
    
    public PPOAgentOptions Options { get; }
    
    public Device Device { get; }
    
    public int[] DiscreteActionDimensions { get; }
    
    public ContinuousActionDimensions[] ContinuousActionDimensions { get; }
    
    public LRScheduler ActorScheduler { get; private set; }
    
    public LRScheduler CriticScheduler { get; private set; }
    
    public IGAIL<TState>? GAIL { get; }
    
    public async ValueTask OptimizeAsync(IMemory<TState> replayBuffer)
    {
        if (Options is { UseRNN: true, BatchSize: > 1 })
        {
            // TODO: remove this block if it is now supported?
            // throw new ArgumentException("Batch size larger than 1 is not yet supported with RNN");
        }

        if (GAIL != null && replayBuffer.Length > 0)
        {
            GAIL.OptimiseDiscriminator(replayBuffer);
        }

        if (replayBuffer.EpisodeCount < Options.BatchSize)
            return;

        if (Options.UseRNN)
        {
            // (padding + masking)
            await OptimizeModelRNNAsync(replayBuffer);

            ActorScheduler.step();
            CriticScheduler.step();

            //(no masking version?)
            // OptimizeRNNPacked(replayBuffer); // TODO: broken
            return;
        }

        using (torch.NewDisposeScope())
        {
            var transitions = replayBuffer.SampleEntireMemory();
            CreateTensorsFromTransitions(Device, transitions, out var stateBatch, out var actionBatch);

            using (var policyOld = ActorNet.get_log_prob(stateBatch, actionBatch, DiscreteActionDimensions.Length, ContinuousActionDimensions.Length).detach())
            using (var valueOld = CriticNet.forward(stateBatch).detach())
            {
                var (discountedRewards, advantages) = GetDiscountedRewardsAndAdvantages(transitions, valueOld);

                if (policyOld.dim() > 1)
                {
                    advantages = advantages.unsqueeze(1);
                }

                var dashboard = await DashboardProvider.Instance.GetDashboardAsync();
                for (var i = 0; i < Options.PPOEpochCount; i++)
                {
                    using (torch.NewDisposeScope())
                    {
                        var (policy, entropy) = ActorNet.get_log_prob_entropy(stateBatch, actionBatch, DiscreteActionDimensions.Length, ContinuousActionDimensions.Length);

                        using var ratios = torch.exp(policy - policyOld);
                        using var surr1 = ratios * advantages;
                        using var surr2 = torch.clamp(ratios, 1.0 - Options.EpsilonClippingFactor, 1.0 + Options.EpsilonClippingFactor) * advantages;
                        
                        var actorLoss = -torch.min(surr1, surr2).mean() - Options.EntropyCoefficient * entropy.mean();
                        //actorLoss.print();
                        ActorOptimizerHelper.zero_grad();
                        actorLoss.backward();
                        torch.nn.utils.clip_grad_norm_(ActorNet.parameters(), Options.ClipGradientNorm);
                        ActorOptimizerHelper.step();

                        if (i == 0)
                        {
                            var klDivergence = (policyOld.exp() * (policyOld - policy)).mean();

                            dashboard.UpdateKLDivergence(klDivergence.item<float>());
                            dashboard.UpdateKLDivergence(klDivergence.item<float>());
                            dashboard.UpdateEntropy(entropy.mean().item<float>());
                            dashboard.UpdateActorLoss(actorLoss.item<float>());
                            dashboard.UpdateActorLearningRate((float) ActorScheduler.get_last_lr().FirstOrDefault());
                        }        
                    }

                    using (torch.NewDisposeScope())
                    {
                        using var values = CriticNet.forward(stateBatch);
                        using var valueClipped = valueOld + torch.clamp(values - valueOld, -Options.ValueLossClipRange, Options.ValueLossClipRange);
                        using var valueLoss1 = torch.pow(values - discountedRewards, 2);
                        using var valueLoss2 = torch.pow(valueClipped - discountedRewards, 2);
                        
                        var criticLoss = Options.ValueLossCoefficient * torch.max(valueLoss1, valueLoss2).mean();
                        CriticOptimizerHelper.zero_grad();
                        criticLoss.backward();
                        torch.nn.utils.clip_grad_norm_(CriticNet.parameters(), Options.ClipGradientNorm);
                        CriticOptimizerHelper.step();

                        if (i == 0)
                        {
                            dashboard.UpdateCriticLoss(criticLoss.item<float>());
                            dashboard.UpdateCriticLearningRate((float)ActorScheduler.get_last_lr().FirstOrDefault());
                        }
                    }
                }
            }
        }

        // TODO: Didn't check that default scheduler doesn't degrade training :)
        ActorScheduler.step();
        CriticScheduler.step();

        replayBuffer.ClearMemory();
    }

    // TODO: This method is odd. `scheduler` is re-assigned and/or checked for null multiple times despite a guarantee that it's not null. Maybe a bug?
    public ValueTask UpdateOptimizersAsync(LRScheduler? scheduler)
    {
        //TODO: SEIROUS violation of DRY. Default Optimizer implementation should be moved to some kind of provider
        ActorOptimizerHelper = torch.optim.Adam(ActorNet.parameters(), Options.LearningRate, amsgrad: true);
        scheduler ??= new CyclicLR(ActorOptimizerHelper, Options.LearningRate * 0.5f, Options.LearningRate * 2f, step_size_up: 10, step_size_down: 10, cycle_momentum: false);
        ActorScheduler = scheduler;

        CriticOptimizerHelper = torch.optim.Adam(CriticNet.parameters(), Options.LearningRate, amsgrad: true);
        scheduler ??= new CyclicLR(CriticOptimizerHelper, Options.LearningRate * 0.5f, Options.LearningRate * 2f, step_size_up: 10, step_size_down: 10, cycle_momentum: false);
        CriticScheduler = scheduler;
        return new();
    }
    
    public static void CreateTensorsFromTransitions(Device device, IEnumerable<MemoryTransition<TState>> transitions, out Tensor stateBatch, out Tensor actionBatch)
    {
        //var length = transitions.Count;
        //var fixedDiscreteActionSize = transitions[0].discreteActions.Length;
        //var fixedContinuousActionSize = 0;

        // Pre-allocate arrays based on the known batch size
        transitions.UnpackMemoryTransition(out var batchStates, out var batchDiscreteActions, out var batchContinuousActions);
        stateBatch = Utilities<TState>.StateBatchToTensor(batchStates, device);
        
        using var discreteActionBatch = torch.tensor(batchDiscreteActions, torch.int64, device: device);
        using var continuousActionBatch = torch.tensor(batchContinuousActions, device: device);
        
        actionBatch = torch.cat([discreteActionBatch, continuousActionBatch], dim: 1);
    }
    
    private (Tensor, Tensor) GetDiscountedRewardsAndAdvantages(IList<MemoryTransition<TState>> transitions, Tensor values)
    {
        var batchSize = transitions.Count;
        if (batchSize == 1)
        {
            return (torch.tensor(transitions.First().Reward).to(Device), torch.tensor(transitions.First().Reward).to(Device));
        }

        var discountedRewards = new float[batchSize];
        var advantages = new float[batchSize];
        var valueArray = Utilities<TState>.ExtractTensorData(values);

        var lastTransitions = transitions.ToArray()
            .Where(t => t.NextTransition == null)
            .Select(t => t)
            .ToList();

        foreach (var lastTransition in lastTransitions)
        {
            var discountedReward = 0d;
            var runningAdd = 0f;
            var currentTransition = lastTransition;
            var transitionIndex = transitions.IndexOf(currentTransition);

            while (currentTransition != null)
            {
                discountedReward = currentTransition.Reward + Options.Gamma * discountedReward;
                discountedRewards[transitionIndex] = (float)discountedReward;

                var nextValue = currentTransition.NextTransition != null ? valueArray[transitionIndex + 1] : 0f;
                var tdError = currentTransition.Reward + Options.Gamma * nextValue - valueArray[transitionIndex];
                runningAdd = tdError + Options.Gamma * Options.GAELambda * runningAdd;
                advantages[transitionIndex] = runningAdd;

                currentTransition = currentTransition.PreviousTransition;
                if (currentTransition != null)
                    transitionIndex = transitions.IndexOf(currentTransition);
            }
        }

        var discountedRewardsTensor = torch.tensor(discountedRewards).to(Device);
        var advantagesTensor = torch.tensor(advantages).to(Device);
        advantagesTensor = (advantagesTensor - advantagesTensor.mean()) / (advantagesTensor.std() + 1e-10);

        return (discountedRewardsTensor, advantagesTensor);
    }
    
    // TODO: claimed to be "broken" per Optimize()
    private void OptimizeRNNPacked(IMemory<TState> replayBuffer)
    {
        using (torch.NewDisposeScope())
        {
            var transitions = replayBuffer.SampleEntireMemory();
            CreateTensorsFromTransitions(Device, transitions, out var stateBatch, out var actionBatch);
            var packedTransition = CreatePackedSequence(transitions, Device);

            var policyOld = ActorNet.get_log_prob(packedTransition, actionBatch, DiscreteActionDimensions.Length, ContinuousActionDimensions.Length).detach();
            var valueOld = CriticNet.forward(packedTransition).detach().squeeze(1);

            (var discountedRewards, var advantages) = GetDiscountedRewardsAndAdvantages(transitions, valueOld);

            if (policyOld.dim() > 1)
            {
                advantages = advantages.unsqueeze(1);
            }

            for (var i = 0; i < Options.PPOEpochCount; i++)
            {
                using (torch.NewDisposeScope())
                {
                    var (policy, entropy) = ActorNet.get_log_prob_entropy(packedTransition, actionBatch, DiscreteActionDimensions.Length, ContinuousActionDimensions.Length);
                    var ratios = torch.exp(policy - policyOld);
                    var surr1 = ratios * advantages;
                    var surr2 = torch.clamp(ratios, 1.0 - Options.EpsilonClippingFactor, 1.0 + Options.EpsilonClippingFactor) * advantages;
                    var actorLoss = -torch.min(surr1, surr2).mean() - Options.EntropyCoefficient * entropy.mean();
                    ActorOptimizerHelper.zero_grad();
                    actorLoss.backward();
                    torch.nn.utils.clip_grad_norm_(ActorNet.parameters(), Options.ClipGradientNorm);
                    ActorOptimizerHelper.step();
                }

                using (torch.NewDisposeScope())
                {
                    var values = CriticNet.forward(packedTransition).squeeze(1);
                    var valueClipped = valueOld + torch.clamp(values - valueOld, -Options.ValueLossClipRange, Options.ValueLossClipRange);
                    var valueLoss1 = torch.pow(values - discountedRewards, 2);
                    var valueLoss2 = torch.pow(valueClipped - discountedRewards, 2);
                    var criticLoss = Options.ValueLossCoefficient * torch.max(valueLoss1, valueLoss2).mean();

                    CriticOptimizerHelper.zero_grad();
                    criticLoss.backward();
                    torch.nn.utils.clip_grad_norm_(CriticNet.parameters(), Options.ClipGradientNorm);
                    CriticOptimizerHelper.step();
                }
            }
        }

        replayBuffer.ClearMemory();
    }
    
    private async ValueTask OptimizeModelRNNAsync(IMemory<TState> myReplayBuffer)
    {
        using (torch.NewDisposeScope())
        {
            var paddedTransitions = PadTransitions(myReplayBuffer.SampleEntireMemory(), Device, out var mask);

            var stateBatches = new List<Tensor>();
            var actionBatches = new List<Tensor>();

            foreach (var sequence in paddedTransitions)
            {
                CreateTensorsFromTransitions(Device, sequence, out var stateBatchEpisode, out var actionBatchEpisode);
                stateBatches.Add(stateBatchEpisode);
                actionBatches.Add(actionBatchEpisode);
            }

            var stateBatch = torch.stack(stateBatches.ToArray(), dim: 0);
            var actionBatch = torch.cat(actionBatches.ToArray(), dim: 0);

            var oldPolicy = ActorNet.get_log_prob(stateBatch, actionBatch, DiscreteActionDimensions.Length, ContinuousActionDimensions.Length).detach();
                
            var oldValue = CriticNet.forward(stateBatch).detach().squeeze(1);

            //Tensor maskedPolicyOld = torch.masked_select(policyOld, mask.to_type(ScalarType.Bool));
            var maskedOldPolicy = MaskedSelectBatch(oldPolicy, mask.to_type(ScalarType.Bool));
            var maskedOldValue = torch.masked_select(oldValue, mask.to_type(ScalarType.Bool));

            //var discountedRewardsList = new List<Tensor>(); TODO: these two are unused
            //var advantagesList = new List<Tensor>();

            var paddedTransitionsSummed = paddedTransitions.SelectMany(t => t).ToList();
            var (maskedDiscountedRewards, maskedAdvantages) = GetDiscountedRewardsAndAdvantages(paddedTransitionsSummed, oldValue);
            maskedDiscountedRewards = torch.masked_select(maskedDiscountedRewards, mask.to_type(ScalarType.Bool));
            maskedAdvantages = torch.masked_select(maskedAdvantages, mask.to_type(ScalarType.Bool));
            maskedAdvantages = ReshapeAdvantages(maskedAdvantages, maskedOldPolicy);

            var dashboard = await DashboardProvider.Instance.GetDashboardAsync();
            for (var i = 0; i < Options.PPOEpochCount; i++)
            {
                using (torch.NewDisposeScope())
                {
                    var (policy, entropy) = ActorNet.get_log_prob_entropy(stateBatch, actionBatch, DiscreteActionDimensions.Length, ContinuousActionDimensions.Length);
                    //policy = torch.masked_select(policy, mask.to_type(ScalarType.Bool));
                    policy = MaskedSelectBatch(policy, mask.to_type(ScalarType.Bool));
                    entropy = torch.masked_select(entropy, mask.to_type(ScalarType.Bool));
                    
                    var ratios = torch.exp(policy - maskedOldPolicy);
                    var surr1 = ratios * maskedAdvantages;
                    var surr2 = torch.clamp(ratios, 1.0 - Options.EpsilonClippingFactor, 1.0 + Options.EpsilonClippingFactor) * maskedAdvantages;
                    // Select the non-masked surrogate values
                    var surr = torch.min(surr1, surr2);

                    // Select the non-masked entropy values
                    var maskedEntropy = entropy;

                    // Calculate the mean of the non-masked surrogate and entropy values
                    var actorLoss = -surr.mean() - Options.EntropyCoefficient * maskedEntropy.mean();
                    ActorOptimizerHelper.zero_grad();
                    actorLoss.backward();
                    torch.nn.utils.clip_grad_norm_(ActorNet.parameters(), Options.ClipGradientNorm);
                    ActorOptimizerHelper.step();
                    
                    if(i == 0)
                    {

                        dashboard.UpdateEntropy(maskedEntropy.mean().item<float>());
                        dashboard.UpdateActorLoss(actorLoss.item<float>());
                        dashboard.UpdateActorLearningRate((float)ActorScheduler.get_last_lr().FirstOrDefault());
                    }
                }

                using (torch.NewDisposeScope())
                {
                    var values = CriticNet.forward(stateBatch).squeeze(1);
                    values = torch.masked_select(values, mask.to_type(ScalarType.Bool));
                    
                    var valueClipped = maskedOldValue + torch.clamp(values - maskedOldValue, -Options.ValueLossClipRange, Options.ValueLossClipRange);
                    var valueLoss1 = torch.pow(values - maskedDiscountedRewards, 2);
                    var valueLoss2 = torch.pow(valueClipped - maskedDiscountedRewards, 2);

                    // Select the non-masked loss values
                    var valueLoss = torch.max(valueLoss1, valueLoss2);

                    // Calculate the mean of the non-masked loss values
                    var criticLoss = Options.ValueLossCoefficient * valueLoss.mean();

                    CriticOptimizerHelper.zero_grad();
                    criticLoss.backward();
                    torch.nn.utils.clip_grad_norm_(CriticNet.parameters(), Options.ClipGradientNorm);
                    CriticOptimizerHelper.step();
                    
                    if (i == 0)
                    {
                        dashboard.UpdateCriticLoss(criticLoss.item<float>());
                        dashboard.UpdateCriticLearningRate((float)CriticScheduler.get_last_lr().FirstOrDefault());
                    }
                }
            }
        }
           
        myReplayBuffer.ClearMemory();
    }
    
    //TODO: move to utils?
    private static Tensor MaskedSelectBatch(Tensor tensor, Tensor mask)
    {
        if (tensor.shape[0] != mask.shape[0])
        {
            throw new ArgumentException("Tensor and mask must have the same batch size (first dimension)");
        }

        // Ensure mask is boolean
        mask = mask.to_type(ScalarType.Bool);

        // Reshape tensor to [batch_size, -1]
        var flattenedSize = tensor.shape.Skip(1).Aggregate(1L, (a, b) => a * b);
        var reshaped = tensor.view(tensor.shape[0], flattenedSize);

        // Apply mask
        var maskedFlat = torch.masked_select(reshaped, mask.unsqueeze(-1));

        // Reshape back to original shape minus the masked-out batch elements
        var newBatchSize = mask.sum().item<long>();
        var newShape = new[] { newBatchSize }.Concat(tensor.shape.Skip(1)).ToArray();
        return maskedFlat.view(newShape);
    }

    private static Tensor ReshapeAdvantages(Tensor advantages, Tensor policyTensor)
    {
        if (policyTensor.dim() == 1)
            return advantages;

        // Ensure advantages is 2D
        if (advantages.dim() == 1)
            advantages = advantages.unsqueeze(1);

        // Expand advantages to match policy shape in first two dimensions
        return advantages.expand(policyTensor.shape[0], policyTensor.shape[1]);
    }
    
    private static List<List<MemoryTransition<TState>>> PadTransitions(IList<MemoryTransition<TState>> transitions, Device device, out Tensor mask)
    {
        //var length = transitions.Count;
        //var fixedDiscreteActionSize = transitions[0].discreteActions.Length;
        //var fixedContinuousActionSize = transitions[0].continuousActions.Length;
        var firstTransitions = transitions.ToArray()
            .Where(t => t.PreviousTransition == null)
            .Select(t => t)
            .ToList();

        var sequenceLengths = new Dictionary<MemoryTransition<TState>, int>();
        foreach (var transition in firstTransitions)
        {
            var sequenceLength = CalculateSequenceLength(transition);
            sequenceLengths.Add(transition, sequenceLength);
        }

        var longestSequence = sequenceLengths.Values.Max();

        var paddedSequences = new List<List<MemoryTransition<TState>>>();
        foreach (var transition in firstTransitions)
        {
            var paddedSequence = new List<MemoryTransition<TState>>();
            AddInitialSequence(transition, paddedSequence);
            PadSequence(paddedSequence[paddedSequence.Count - 1], longestSequence, sequenceLengths[transition], paddedSequence);
            paddedSequences.Add(paddedSequence);
        }

        mask = CreateMask(paddedSequences, sequenceLengths, device);
        return paddedSequences;

        static Tensor CreateMask(List<List<MemoryTransition<TState>>> paddedSequences, Dictionary<MemoryTransition<TState>, int> sequenceLengths, Device device)
        {
            var maskList = new List<Tensor>();
            foreach (var sequence in paddedSequences)
            {
                var originalLength = sequenceLengths[sequence[0]];
                var paddedLength = sequence.Count;
                var maskData = Enumerable.Repeat(true, originalLength)
                    .Concat(Enumerable.Repeat(false, paddedLength - originalLength))
                    .ToArray();
                
                maskList.Add(torch.tensor(maskData)); //.unsqueeze
            }

            var mask = torch.cat(maskList.ToArray(), dim: 0).to(device);
            return mask;
        }

        static int CalculateSequenceLength(MemoryTransition<TState> transition)
        {
            if (transition.NextTransition == null)
                return 1;

            return 1 + CalculateSequenceLength(transition.NextTransition);
        }

        static void AddInitialSequence(MemoryTransition<TState>? transition, List<MemoryTransition<TState>> paddedSequence)
        {
            while (transition != null)
            {
                paddedSequence.Add(transition);
                transition = transition.NextTransition;
            }
        }

        static MemoryTransition<TState> PadSequence(MemoryTransition<TState> transition, int targetLength, int currentLength, List<MemoryTransition<TState>> paddedSequence)
        {
            if (currentLength >= targetLength)
                return transition;

            var paddedTransition = new MemoryTransition<TState>(transition.State, transition.Actions, 0f, transition.NextState, null, transition);
            transition.NextTransition = paddedTransition;
            paddedSequence.Add(paddedTransition);

            var result = PadSequence(paddedTransition, targetLength, currentLength + 1, paddedSequence);
            return result;
        }
    }
    
    private static PackedSequence CreatePackedSequence(IList<MemoryTransition<TState>> transitions, Device device)
    {
        var firstTransitions = transitions.ToArray()
            .Where(t => t.PreviousTransition == null)
            .Select(t => t)
            .ToList();

        var sequenceTensors = new List<Tensor>();

        foreach (var transition in firstTransitions)
        {
            var sequenceStates = new List<TState>();
            var currentTransition = transition;

            while (currentTransition != null)
            {
                sequenceStates.Add(currentTransition._state);
                currentTransition = currentTransition.NextTransition;
            }

            var sequenceTensor = Utilities<TState>.StateBatchToTensor(sequenceStates.ToArray(), device);
            sequenceTensors.Add(sequenceTensor);
        }

        return torch.nn.utils.rnn.pack_sequence(sequenceTensors, false);
    }
}