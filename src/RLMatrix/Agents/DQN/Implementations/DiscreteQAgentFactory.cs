using RLMatrix.Common;

namespace RLMatrix;

public static class DiscreteQAgentFactory<TState>
    where TState : notnull
{
    public static ComposableQDiscreteAgent<TState> ComposeQAgent(DQNAgentOptions options, int[] actionDimensions, StateDimensions stateDimensions, IDQNNetProvider? netProvider = null, LRScheduler? lrScheduler = null, IGAIL<TState>? gail = null)
    {
        //Uses pattern matching from options to create a neural net for algorithm selected from permutations
        //Here we also initialise device and optimizer
        netProvider ??= GetNetProviderFromOptions(options);
        var device = torch.device(torch.cuda.is_available() ? "cuda" : "cpu");
        var policyNet = netProvider.CreateCriticNet<TState>(new DiscreteEnvironmentSizeDTO(actionDimensions, stateDimensions), options.UseNoisyLayers, options.NoisyLayersScale, options.NumberOfAtoms).to(device);
        var targetNet = netProvider.CreateCriticNet<TState>(new DiscreteEnvironmentSizeDTO(actionDimensions, stateDimensions), options.UseNoisyLayers, options.NoisyLayersScale, options.NumberOfAtoms).to(device);
        var optimizer = torch.optim.Adam(policyNet.parameters(), options.LearningRate);
        lrScheduler ??= new CyclicLR(optimizer, options.LearningRate * 0.5f, options.LearningRate * 2f, step_size_up: 500, step_size_down: 2000, cycle_momentum: false);

        //We need to discriminate between tensor functions for categorical and non-categorical DQN
        IComputeQValues qValuesCalculator = options.UseCategoricalDQN
            ? new CategoricalComputeQValues(actionDimensions, options.NumberOfAtoms)
            : new BaseComputeQValues();

        IExtractStateActionValues qValuesExtractor = options.UseCategoricalDQN
            ? new CategoricalExtractStateActionValues(options.NumberOfAtoms)
            : new BaseExtractStateActionValues();

        IComputeNextStateValues nextStateValueCalculator = options.UseCategoricalDQN
            ? new C51ComputeNextStateValues(options.NumberOfAtoms)
            : new BaseComputeNextStateValues();

        IExpectedStateActionValuesComputer<TState> expectedStateActionValuesComputerCalculator = options.UseCategoricalDQN
            ? new CategoricalExpectedStateActionValuesComputer<TState>(options.ValueDistributionMin, options.ValueDistributionMax, options.NumberOfAtoms, device/*, support: GetSupport(options.NumAtoms, options.VMin, options.VMax, device)*/) // TODO: support went unused
            : new BaseExpectedStateActionValuesComputer<TState>();

        ILossComputer lossComputerCalculator = options.UseCategoricalDQN
            ? new CategoricalLossComputer()
            : new BaseLossComputer();

        var qOptimizer = new QOptimizer<TState>(policyNet, targetNet, optimizer, qValuesCalculator, qValuesExtractor, nextStateValueCalculator, expectedStateActionValuesComputerCalculator, lossComputerCalculator, options, device, actionDimensions, lrScheduler, gail); //TODO: Null GAIL
            
        //If noisy layers are present will be cached here for noise resetting
        List<NoisyLinear> noisyLayers = new();
        if (options.UseNoisyLayers)
        {
            noisyLayers.AddRange(from module in policyNet.modules()
                where module is NoisyLinear
                select (NoisyLinear)module);
        }

        //Caching support tensor for categorical DQN
        Tensor? support = null;
        if (options.UseCategoricalDQN)
        {
            support = GetSupport(options.NumberOfAtoms, options.ValueDistributionMin, options.ValueDistributionMax, device);
        }

        //composition of the DQN agent takes place here
        var agent = new ComposableQDiscreteAgent<TState>(policyNet, targetNet, qOptimizer, GetMemoryFromOptions(options), actionDimensions,
            () => noisyLayers.ForEach(module => module.ResetNoise()), options, device, support,// optimizer,
            (states, agent, isTraining) =>
            {
                using var disposeScope = torch.NewDisposeScope();
                return GetActionSelectFuncFromOptions(options)(states, agent, isTraining);
            });

        return agent;
    }

    private static IDQNNetProvider GetNetProviderFromOptions(DQNAgentOptions options)
    {
        return new DQNNetProvider(options.Width, options.Depth, options.UseDuelingDQN, options.UseCategoricalDQN);
    }

    private static IMemory<TState> GetMemoryFromOptions(DQNAgentOptions options)
    {
        return options.UsePrioritizedExperienceReplay
            ? new PrioritizedReplayMemory<TState>(options.MemorySize)
            : new ReplayMemory<TState>(options.MemorySize);
    }

    #region actionSelection

    private static Func<TState[], ComposableQDiscreteAgent<TState>, bool, RLActions[]> GetActionSelectFuncFromOptions(DQNAgentOptions opts)
    {
        //We fetch correct action selection algorithm based on permutation of options
        if(opts.UseBatchedInputProcessing && opts.UseBoltzmannExploration)
            return (opts.UseNoisyLayers, opts.UseCategoricalDQN) switch
            {
                (false, false) => VanillaActionSelectionBatchedBoltzmann,
                (true, false) => NoisyActionSelectionBatchedBoltzmann,
                (false, true) => CategoricalActionSelectionBatchedBoltzmann,
                (true, true) => CategoricalNoisyActionSelectionBatchedBoltzmann,
            };


        if (opts.UseBatchedInputProcessing)
            return (opts.UseNoisyLayers, opts.UseCategoricalDQN) switch
            {
                (false, false) => VanillaActionSelectionBatched,
                (true, false) => NoisyActionSelectionBatched,
                (false, true) => CategoricalActionSelectionBatched,
                (true, true) => CategoricalNoisyActionSelectionBatched,
            };

        return (opts.UseNoisyLayers, opts.UseCategoricalDQN) switch
        {
            (false, false) => VanillaActionSelection,
            (true, false) => NoisyActionSelection,
            (false, true) => CategoricalActionSelection,
            (true, true) => CategoricalNoisyActionSelection,
        };
    }

    #region UnbatchedActionSelection
    private static RLActions[] VanillaActionSelection(TState[] states, ComposableQDiscreteAgent<TState> agent, bool isTraining)
    {
        var actions = new RLActions[states.Length];
        var epsThreshold = agent.Options.EpsilonMin + (agent.Options.EpsilonStart - agent.Options.EpsilonMin) *
            Math.Exp(-1.0 * agent.EpisodeCount / agent.Options.EpsilonDecay);

        for (var i = 0; i < states.Length; i++)
        {
            if (agent.Random.NextDouble() > epsThreshold || !isTraining)
            {
                actions[i] = ActionsFromState(states[i], agent.PolicyNet, agent.DiscreteActionDimensions, agent.Device);
            }
            else
            {
                actions[i] = RandomActions(agent.DiscreteActionDimensions, agent.Random);
            }
        }

        return actions;
    }

    private static RLActions[] NoisyActionSelection(TState[] states, ComposableQDiscreteAgent<TState> agent, bool isTraining)
    {
        var actions = new RLActions[states.Length];

        if (isTraining)
        {
            agent.PolicyNet.train();
        }
        else
        {
            agent.PolicyNet.eval();
        }

        for (var i = 0; i < states.Length; i++)
        {
            if (isTraining)
            {
                agent.ResetNoisyLayers();
            }
            
            actions[i] = ActionsFromState(states[i], agent.PolicyNet, agent.DiscreteActionDimensions, agent.Device);
        }

        return actions;
    }

    private static RLActions[] CategoricalActionSelection(TState[] states, ComposableQDiscreteAgent<TState> agent, bool isTraining)
    {
        var actions = new RLActions[states.Length];
        for (var i = 0; i < states.Length; i++)
        {
            actions[i] = CategoricalActionsFromState(states[i], agent.PolicyNet, agent.DiscreteActionDimensions, agent.Options.NumberOfAtoms, agent.Device, agent.Support!);
        }

        return actions;
    }

    private static RLActions[] CategoricalNoisyActionSelection(TState[] states, ComposableQDiscreteAgent<TState> agent, bool isTraining)
    {
        var actions = new RLActions[states.Length];

        if (isTraining)
        {
            agent.PolicyNet.train();
        }
        else
        {
            agent.PolicyNet.eval();
        }

        for (var i = 0; i < states.Length; i++)
        {
            if (isTraining)
            {
                agent.ResetNoisyLayers();
            }
            
            actions[i] = CategoricalActionsFromState(states[i], agent.PolicyNet, agent.DiscreteActionDimensions, agent.Options.NumberOfAtoms, agent.Device, agent.Support!);
        }

        return actions;
    }
    #endregion

    #region BatchedActionSelection
    private static RLActions[] VanillaActionSelectionBatched(TState[] states, ComposableQDiscreteAgent<TState> agent, bool isTraining)
    {
        var epsThreshold = agent.Options.EpsilonMin + (agent.Options.EpsilonStart - agent.Options.EpsilonMin) *
            Math.Exp(-1.0 * agent.EpisodeCount / agent.Options.EpsilonDecay);

        var stateTensor = Utilities<TState>.StateBatchToTensor(states, agent.Device);
        Tensor qValuesAllHeads;

        using (torch.no_grad())
        {
            qValuesAllHeads = agent.PolicyNet.forward(stateTensor).view(states.Length, agent.DiscreteActionDimensions.Length, agent.DiscreteActionDimensions[0]);
        }

        var actions = new RLActions[states.Length];
        for (var i = 0; i < states.Length; i++)
        {
            if (agent.Random.NextDouble() > epsThreshold || !isTraining)
            {
                var bestActions = qValuesAllHeads[i].argmax(dim: -1).squeeze().to(ScalarType.Int32);
                actions[i] = RLActions.Discrete(bestActions.data<int>().ToArray());
            }
            else
            {
                actions[i] = RandomActions(agent.DiscreteActionDimensions, agent.Random);
            }
        }

        return actions;
    }

    private static RLActions[] NoisyActionSelectionBatched(TState[] states, ComposableQDiscreteAgent<TState> agent, bool isTraining)
    {
        if (isTraining)
        {
            agent.PolicyNet.train();
            agent.ResetNoisyLayers();
        }
        else
        {
            agent.PolicyNet.eval();
        }

        var stateTensor = Utilities<TState>.StateBatchToTensor(states, agent.Device);
        Tensor qValuesAllHeads;

        using (torch.no_grad())
        {
            qValuesAllHeads = agent.PolicyNet.forward(stateTensor).view(states.Length, agent.DiscreteActionDimensions.Length, agent.DiscreteActionDimensions[0]);
        }

        var actions = new RLActions[states.Length];
        for (var i = 0; i < states.Length; i++)
        {
            var bestActions = qValuesAllHeads[i].argmax(dim: -1).squeeze().to(ScalarType.Int32);
            actions[i] = RLActions.Discrete(bestActions.data<int>().ToArray());
        }

        return actions;
    }

    private static RLActions[] CategoricalActionSelectionBatched(TState[] states, ComposableQDiscreteAgent<TState> agent, bool isTraining)
    {
        var stateTensor = Utilities<TState>.StateBatchToTensor(states, agent.Device);
        Tensor qValuesAllHeads;

        using (torch.no_grad())
        {
            qValuesAllHeads = agent.PolicyNet.forward(stateTensor).view(states.Length, agent.DiscreteActionDimensions.Length, agent.DiscreteActionDimensions[0], agent.Options.NumberOfAtoms);
        }

        var expectedQValues = (qValuesAllHeads * agent.Support!).sum(dim: -1);

        var actions = new RLActions[states.Length];
        for (var i = 0; i < states.Length; i++)
        {
            var bestActions = expectedQValues[i].argmax(dim: -1).squeeze().to(ScalarType.Int32);
            actions[i] = RLActions.Discrete(bestActions.data<int>().ToArray());
        }

        return actions;
    }

    private static RLActions[] CategoricalNoisyActionSelectionBatched(TState[] states, ComposableQDiscreteAgent<TState> agent, bool isTraining)
    {
        if (isTraining)
        {
            agent.PolicyNet.train();
            agent.ResetNoisyLayers();
        }
        else
        {
            agent.PolicyNet.eval();
        }

        var stateTensor = Utilities<TState>.StateBatchToTensor(states, agent.Device);
        Tensor qValuesAllHeads;

        using (torch.no_grad())
        {
            qValuesAllHeads = agent.PolicyNet.forward(stateTensor).view(states.Length, agent.DiscreteActionDimensions.Length, agent.DiscreteActionDimensions[0], agent.Options.NumberOfAtoms);
        }

        var expectedQValues = (qValuesAllHeads * agent.Support!).sum(dim: -1);

        var actions = new RLActions[states.Length];
        for (var i = 0; i < states.Length; i++)
        {
            var bestActions = expectedQValues[i].argmax(dim: -1).squeeze().to(ScalarType.Int32);
            actions[i] = RLActions.Discrete(bestActions.data<int>().ToArray());
        }

        return actions;
    }
    #endregion

    #region BatchedBoltzmannActionSelection
    private static RLActions[] VanillaActionSelectionBatchedBoltzmann(TState[] states, ComposableQDiscreteAgent<TState> agent, bool isTraining)
    {
        var epsThreshold = agent.Options.EpsilonMin + (agent.Options.EpsilonStart - agent.Options.EpsilonMin) *
            Math.Exp(-1.0 * agent.EpisodeCount / agent.Options.EpsilonDecay);

        var stateTensor = Utilities<TState>.StateBatchToTensor(states, agent.Device);
        Tensor qValuesAllHeads;

        using (torch.no_grad())
        {
            qValuesAllHeads = agent.PolicyNet.forward(stateTensor).view(states.Length, agent.DiscreteActionDimensions.Length, agent.DiscreteActionDimensions[0]);
        }

        qValuesAllHeads = torch.where(qValuesAllHeads.isnan(), torch.zeros_like(qValuesAllHeads), qValuesAllHeads);
        const float temperature = 1.0f;
        var actions = new RLActions[states.Length];
        for (var i = 0; i < states.Length; i++)
        {
            var currentActions = new int[agent.DiscreteActionDimensions.Length];
            for (var j = 0; j < agent.DiscreteActionDimensions.Length; j++)
            {
                if (agent.Random.NextDouble() > epsThreshold || !isTraining)
                {
                    var scaledQValues = qValuesAllHeads[i, j] / temperature;
                    var maxQValue = scaledQValues.max();
                    var actionProbabilities = torch.softmax(scaledQValues - maxQValue, dim: -1);
                    actionProbabilities = torch.clamp(actionProbabilities, 1e-10f, 1.0f);
                    actionProbabilities /= actionProbabilities.sum();
                    var sampledAction = torch.multinomial(actionProbabilities, num_samples: 1, replacement: true);
                    currentActions[j] = (int)sampledAction.item<long>();
                }
                else
                {
                    currentActions[j] = agent.Random.Next(0, agent.DiscreteActionDimensions[j]);
                }
            }
            
            actions[i] = RLActions.Discrete(currentActions);
        }

        return actions;
    }

    private static RLActions[] NoisyActionSelectionBatchedBoltzmann(TState[] states, ComposableQDiscreteAgent<TState> agent, bool isTraining)
    {
        if (isTraining)
        {
            agent.PolicyNet.train();
            agent.ResetNoisyLayers();
        }
        else
        {
            agent.PolicyNet.eval();
        }

        var stateTensor = Utilities<TState>.StateBatchToTensor(states, agent.Device);
        Tensor qValuesAllHeads;

        using (torch.no_grad())
        {
            qValuesAllHeads = agent.PolicyNet.forward(stateTensor).view(states.Length, agent.DiscreteActionDimensions.Length, agent.DiscreteActionDimensions[0]);
        }

        qValuesAllHeads = torch.where(qValuesAllHeads.isnan(), torch.zeros_like(qValuesAllHeads), qValuesAllHeads);
        const float temperature = 1.0f;
        
        var actions = new RLActions[states.Length];
        for (var i = 0; i < states.Length; i++)
        {
            var currentActions = new int[agent.DiscreteActionDimensions.Length];
            for (var j = 0; j < agent.DiscreteActionDimensions.Length; j++)
            {
                var scaledQValues = qValuesAllHeads[i, j] / temperature;
                var maxQValue = scaledQValues.max();
                var actionProbabilities = torch.softmax(scaledQValues - maxQValue, dim: -1);
                actionProbabilities = torch.clamp(actionProbabilities, 1e-10f, 1.0f);
                actionProbabilities /= actionProbabilities.sum();
                var sampledAction = torch.multinomial(actionProbabilities, num_samples: 1, replacement: true);
                currentActions[j] = (int)sampledAction.item<long>();
            }
            
            actions[i] = RLActions.Discrete(currentActions);
        }

        return actions;
    }

    private static RLActions[] CategoricalActionSelectionBatchedBoltzmann(TState[] states, ComposableQDiscreteAgent<TState> agent, bool isTraining)
    {
        var stateTensor = Utilities<TState>.StateBatchToTensor(states, agent.Device);
        Tensor qValuesAllHeads;

        using (torch.no_grad())
        {
            qValuesAllHeads = agent.PolicyNet.forward(stateTensor).view(states.Length, agent.DiscreteActionDimensions.Length, agent.DiscreteActionDimensions[0], agent.Options.NumberOfAtoms);
        }

        var expectedQValues = (qValuesAllHeads * agent.Support!).sum(dim: -1);
        expectedQValues = torch.where(expectedQValues.isnan(), torch.zeros_like(expectedQValues), expectedQValues);

        const float temperature = 1.0f;
        var actions = new RLActions[states.Length];
        for (var i = 0; i < states.Length; i++)
        {
            var currentActions = new int[agent.DiscreteActionDimensions.Length];
            for (var j = 0; j < agent.DiscreteActionDimensions.Length; j++)
            {
                var scaledQValues = expectedQValues[i, j] / temperature;
                var maxQValue = scaledQValues.max();
                var actionProbabilities = torch.softmax(scaledQValues - maxQValue, dim: -1);
                actionProbabilities = torch.clamp(actionProbabilities, 1e-10f, 1.0f);
                actionProbabilities /= actionProbabilities.sum();
                var sampledAction = torch.multinomial(actionProbabilities, num_samples: 1, replacement: true);
                currentActions[j] = (int)sampledAction.item<long>();
            }
            
            actions[i] = RLActions.Discrete(currentActions);
        }

        return actions;
    }

    private static RLActions[] CategoricalNoisyActionSelectionBatchedBoltzmann(TState[] states, ComposableQDiscreteAgent<TState> agent, bool isTraining)
    {
        if (isTraining)
        {
            agent.PolicyNet.train();
            agent.ResetNoisyLayers();
        }
        else
        {
            agent.PolicyNet.eval();
        }

        var stateTensor = Utilities<TState>.StateBatchToTensor(states, agent.Device);
        Tensor qValuesAllHeads;
        using (torch.no_grad())
        {
            qValuesAllHeads = agent.PolicyNet.forward(stateTensor).view(states.Length, agent.DiscreteActionDimensions.Length, agent.DiscreteActionDimensions[0], agent.Options.NumberOfAtoms);
        }

        var expectedQValues = (qValuesAllHeads * agent.Support!).sum(dim: -1);
        expectedQValues = torch.where(expectedQValues.isnan(), torch.zeros_like(expectedQValues), expectedQValues);

        const float temperature = 1.0f;
        var actions = new RLActions[states.Length];
        for (var i = 0; i < states.Length; i++)
        {
            var currentActions = new int[agent.DiscreteActionDimensions.Length];
            for (var j = 0; j < agent.DiscreteActionDimensions.Length; j++)
            {
                var scaledQValues = expectedQValues[i, j] / temperature;
                var maxQValue = scaledQValues.max();
                var actionProbabilities = torch.softmax(scaledQValues - maxQValue, dim: -1);
                actionProbabilities = torch.clamp(actionProbabilities, 1e-10f, 1.0f);
                actionProbabilities /= actionProbabilities.sum();
                var sampledAction = torch.multinomial(actionProbabilities, num_samples: 1, replacement: true);
                currentActions[j] = (int)sampledAction.item<long>();
            }
            
            actions[i] = RLActions.Discrete(currentActions);
        }

        return actions;
    }
    #endregion

    private static RLActions ActionsFromState(TState state, TensorModule policyNet, int[] discreteActionDimensions, Device device)
    {
        using (torch.no_grad())
        {
            var stateTensor = Utilities<TState>.StateToTensor(state, device);
            var qValuesAllHeads = policyNet.forward(stateTensor).view(1, discreteActionDimensions.Length, discreteActionDimensions[0]);
            var bestActions = qValuesAllHeads.argmax(dim: -1).squeeze().to(ScalarType.Int32);
            var result = bestActions.data<int>().ToArray();
            return RLActions.Discrete(result);
        }
    }

    private static RLActions CategoricalActionsFromState(TState state, TensorModule policyNet, int[] discreteActionDimensions, int numAtoms, Device device, Tensor support)
    {
        using (torch.no_grad())
        {
            var stateTensor = Utilities<TState>.StateToTensor(state, device);
            var qValuesAllHeads = policyNet.forward(stateTensor).view(1, discreteActionDimensions.Length, discreteActionDimensions[0], numAtoms);
            var expectedQValues = (qValuesAllHeads * support).sum(dim: -1);
            var bestActions = expectedQValues.argmax(dim: -1).squeeze().to(ScalarType.Int32);
            return RLActions.Discrete(bestActions.data<int>().ToArray());
        }
    }

    private static RLActions RandomActions(int[] actionSizes, Random random)
    {
        return RLActions.Discrete(actionSizes.Select(size => random.Next(0, size)).ToArray());
    }

    private static Tensor GetSupport(int numAtoms, float vMin, float vMax, Device device)
    {
        //var deltaZ = (vMax - vMin) / (numAtoms - 1); TODO: unused?
        return torch.linspace(vMin, vMax, steps: numAtoms, device: device);
    }
    #endregion
}