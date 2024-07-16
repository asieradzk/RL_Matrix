using OneOf;
using RLMatrix.Agents.Common;
using RLMatrix.Agents.DQN.Domain;
using RLMatrix.Agents.DQN.Implementations.C51;
using RLMatrix.Memories;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim.lr_scheduler;

namespace RLMatrix
{
    public static class DiscreteQAgentFactory<T>
    {
        public static ComposableQDiscreteAgent<T> ComposeQAgent(DQNAgentOptions options, int[] ActionSizes, OneOf<int, (int, int)> StateSizes, IDQNNetProvider<T> netProvider = null, LRScheduler lrScheduler = null, IGAIL<T> gail = null)
        {
            //Uses pattern matching from options to create a neural net for algorithm selected from permutations
            //Here we also initialise device and optimizer
            netProvider ??= GetNetProviderFromOptions(options);
            var device = torch.device(torch.cuda.is_available() ? "cuda" : "cpu");
            var policyNet = netProvider.CreateCriticNet(new EnvSizeDTO<T> { actionSize = ActionSizes, stateSize = StateSizes }, options.NoisyLayers, options.NoisyLayersScale, options.NumAtoms).to(device);
            var targetNet = netProvider.CreateCriticNet(new EnvSizeDTO<T> { actionSize = ActionSizes, stateSize = StateSizes }, options.NoisyLayers, options.NoisyLayersScale, options.NumAtoms).to(device);
            var optimizer = optim.Adam(policyNet.parameters(), options.LR);
            lrScheduler ??= new optim.lr_scheduler.impl.CyclicLR(optimizer, options.LR * 0.5f, options.LR * 2f, step_size_up: 500, step_size_down: 2000, cycle_momentum: false);


            //We need to discriminate between tensor functions for categorical and non-categorical DQN
            IComputeQValues qValuesCalculator = options.CategoricalDQN
                ? new CategoricalComputeQValues(ActionSizes, options.NumAtoms)
                : new BaseComputeQValues();

            IExtractStateActionValues qValuesExtractor = options.CategoricalDQN
                ? new CategoricalExtractStateActionValues(options.NumAtoms)
                : new BaseExtractStateActionValues();

            IComputeNextStateValues nextStateValueCalculator = options.CategoricalDQN
                ? new C51ComputeNextStateValues(options.NumAtoms)
                : new BaseComputeNextStateValues();

            IComputeExpectedStateActionValues<T> expectedStateActionValuesCalculator = options.CategoricalDQN
                ? new CategoricalComputeExpectedStateActionValues<T>(options.VMin, options.VMax, options.NumAtoms, device, support: GetSupport(options.NumAtoms, options.VMin, options.VMax, device)) //TODO: a bit weird, should I just pass options?
                : new BaseComputeExpectedStateActionValues<T>();

            IComputeLoss lossCalculator = options.CategoricalDQN
                ? new CategoricalComputeLoss()
                : new BaseComputeLoss();

            var Optimizer = new QOptimize<T>(policyNet, targetNet, optimizer, qValuesCalculator, qValuesExtractor, nextStateValueCalculator, expectedStateActionValuesCalculator, lossCalculator, options, device, ActionSizes, lrScheduler, gail); //TODO: Null GAIL
            
            //If noisy layers are present will be cached here for noise resetting
            List<NoisyLinear> noisyLayers = new();
            if (options.NoisyLayers)
            {
                noisyLayers.AddRange(from module in policyNet.modules()
                          where module is NoisyLinear
                          select (NoisyLinear)module);
            }

            //Caching support tensor for categorical DQN
            Tensor support = null;
            if (options.CategoricalDQN)
            {
                support = GetSupport(options.NumAtoms, options.VMin, options.VMax, device);
            }

            //composition of the DQN agent takes place here
            var Agent = new ComposableQDiscreteAgent<T>
            {
                Options = options,
                policyNet = policyNet,
                targetNet = targetNet,
                optimizer = optimizer,
                Optimizer = Optimizer,
                Device = device,
                Memory = GetMemoryFromOptions(options),
                ActionSizes = ActionSizes,
                ResetNoisyLayers = () => noisyLayers.ForEach(module => module.ResetNoise()),
                SelectActionsFunc = (states, agent, isTraining) =>
                {
                    using (var disposeScope = torch.NewDisposeScope())
                    {
                        return GetActionSelectFuncFromOptions(options)(states, agent, isTraining);
                    }
                },
                support = support,
            };


            return Agent;

        }

        private static IDQNNetProvider<T> GetNetProviderFromOptions(DQNAgentOptions options)
        {
            return new DQNNetProvider<T>(options.Width, options.Depth, options.DuelingDQN, options.CategoricalDQN);
        }

        private static IMemory<T> GetMemoryFromOptions(DQNAgentOptions options)
        {
            IMemory<T> memory;
            switch(options.PrioritizedExperienceReplay)
            {                 case true:
                               memory = new PrioritizedReplayMemory<T>(options.MemorySize);
                               break;
                           case false:
                               memory = new ReplayMemory<T>(options.MemorySize);
                               break;
                       }
            return memory;
        }

        #region actionSelection

        private static Func<T[], ComposableQDiscreteAgent<T>, bool, int[][]> GetActionSelectFuncFromOptions(DQNAgentOptions opts)
        {
            //We fetch correct action selection algorithm based on permutation of options
            if(opts.BatchedInputProcessing && opts.BoltzmannExploration)
                return (opts.NoisyLayers, opts.CategoricalDQN) switch
                {
                    (false, false) => VanillaActionSelectionBatchedBoltzmann,
                    (true, false) => NoisyActionSelectionBatchedBoltzmann,
                    (false, true) => CategoricalActionSelectionBatchedBoltzmann,
                    (true, true) => CategoricalNoisyActionSelectionBatchedBoltzmann,
                };


            if (opts.BatchedInputProcessing)
            return (opts.NoisyLayers, opts.CategoricalDQN) switch
            {
                (false, false) => VanillaActionSelectionBatched,
                (true, false) => NoisyActionSelectionBatched,
                (false, true) => CategoricalActionSelectionBatched,
                (true, true) => CategoricalNoisyActionSelectionBatched,
            };

            return (opts.NoisyLayers, opts.CategoricalDQN) switch
            {
                (false, false) => VanillaActionSelection,
                (true, false) => NoisyActionSelection,
                (false, true) => CategoricalActionSelection,
                (true, true) => CategoricalNoisyActionSelection,
            };
        }


        #region UnbatchedActionSelection
        private static int[][] VanillaActionSelection(T[] states, ComposableQDiscreteAgent<T> agent, bool isTraining)
        {
            int[][] actions = new int[states.Length][];
            double epsThreshold = agent.Options.EPS_END + (agent.Options.EPS_START - agent.Options.EPS_END) *
                Math.Exp(-1.0 * agent.episodeCount / agent.Options.EPS_DECAY);

            for (int i = 0; i < states.Length; i++)
            {
                if (agent.Random.NextDouble() > epsThreshold || !isTraining)
                {
                    actions[i] = ActionsFromState(states[i], agent.policyNet, agent.ActionSizes, agent.Device);
                }
                else
                {
                    actions[i] = RandomActions(agent.ActionSizes, agent.Random);
                }
            }

            return actions;
        }

        private static int[][] NoisyActionSelection(T[] states, ComposableQDiscreteAgent<T> agent, bool isTraining)
        {
            int[][] result = new int[states.Length][];

            if (isTraining)
            {
                agent.policyNet.train();
            }
            else
            {
                agent.policyNet.eval();
            }

            for (int i = 0; i < states.Length; i++)
            {
                if (isTraining)
                {
                    agent.ResetNoisyLayers();
                }
                result[i] = ActionsFromState(states[i], agent.policyNet, agent.ActionSizes, agent.Device);
            }

            return result;
        }

        private static int[][] CategoricalActionSelection(T[] states, ComposableQDiscreteAgent<T> agent, bool isTraining)
        {
            int[][] result = new int[states.Length][];

            for (int i = 0; i < states.Length; i++)
            {
                result[i] = CategoricalActionsFromState(states[i], agent.policyNet, agent.ActionSizes, agent.Options.NumAtoms, agent.Device, agent.support);
            }

            return result;
        }

        private static int[][] CategoricalNoisyActionSelection(T[] states, ComposableQDiscreteAgent<T> agent, bool isTraining)
        {
            int[][] result = new int[states.Length][];

            if (isTraining)
            {
                agent.policyNet.train();
            }
            else
            {
                agent.policyNet.eval();
            }

            for (int i = 0; i < states.Length; i++)
            {
                if (isTraining)
                {
                    agent.ResetNoisyLayers();
                }
                result[i] = CategoricalActionsFromState(states[i], agent.policyNet, agent.ActionSizes, agent.Options.NumAtoms, agent.Device, agent.support);
            }

            return result;
        }
        #endregion

        #region BatchedActionSelection
        private static int[][] VanillaActionSelectionBatched(T[] states, ComposableQDiscreteAgent<T> agent, bool isTraining)
        {
            double epsThreshold = agent.Options.EPS_END + (agent.Options.EPS_START - agent.Options.EPS_END) *
                Math.Exp(-1.0 * agent.episodeCount / agent.Options.EPS_DECAY);

            Tensor stateTensor = Utilities<T>.StateBatchToTensor(states, agent.Device);
            Tensor qValuesAllHeads;

            using (torch.no_grad())
            {
                qValuesAllHeads = agent.policyNet.forward(stateTensor).view(states.Length, agent.ActionSizes.Length, agent.ActionSizes[0]);
            }

            int[][] actions = new int[states.Length][];

            for (int i = 0; i < states.Length; i++)
            {
                if (agent.Random.NextDouble() > epsThreshold || !isTraining)
                {
                    Tensor bestActions = qValuesAllHeads[i].argmax(dim: -1).squeeze().to(ScalarType.Int32);
                    actions[i] = bestActions.data<int>().ToArray();
                }
                else
                {
                    actions[i] = RandomActions(agent.ActionSizes, agent.Random);
                }
            }

            return actions;
        }

        private static int[][] NoisyActionSelectionBatched(T[] states, ComposableQDiscreteAgent<T> agent, bool isTraining)
        {
            if (isTraining)
            {
                agent.policyNet.train();
                agent.ResetNoisyLayers();
            }
            else
            {
                agent.policyNet.eval();
            }

            Tensor stateTensor = Utilities<T>.StateBatchToTensor(states, agent.Device);
            Tensor qValuesAllHeads;

            using (torch.no_grad())
            {
                qValuesAllHeads = agent.policyNet.forward(stateTensor).view(states.Length, agent.ActionSizes.Length, agent.ActionSizes[0]);
            }

            int[][] actions = new int[states.Length][];

            for (int i = 0; i < states.Length; i++)
            {
                Tensor bestActions = qValuesAllHeads[i].argmax(dim: -1).squeeze().to(ScalarType.Int32);
                actions[i] = bestActions.data<int>().ToArray();
            }

            return actions;
        }

        private static int[][] CategoricalActionSelectionBatched(T[] states, ComposableQDiscreteAgent<T> agent, bool isTraining)
        {
            Tensor stateTensor = Utilities<T>.StateBatchToTensor(states, agent.Device);
            Tensor qValuesAllHeads;

            using (torch.no_grad())
            {
                qValuesAllHeads = agent.policyNet.forward(stateTensor).view(states.Length, agent.ActionSizes.Length, agent.ActionSizes[0], agent.Options.NumAtoms);
            }

            Tensor expectedQValues = (qValuesAllHeads * agent.support).sum(dim: -1);

            int[][] actions = new int[states.Length][];

            for (int i = 0; i < states.Length; i++)
            {
                Tensor bestActions = expectedQValues[i].argmax(dim: -1).squeeze().to(ScalarType.Int32);
                actions[i] = bestActions.data<int>().ToArray();
            }

            return actions;
        }

        private static int[][] CategoricalNoisyActionSelectionBatched(T[] states, ComposableQDiscreteAgent<T> agent, bool isTraining)
        {
            if (isTraining)
            {
                agent.policyNet.train();
                agent.ResetNoisyLayers();
            }
            else
            {
                agent.policyNet.eval();
            }

            Tensor stateTensor = Utilities<T>.StateBatchToTensor(states, agent.Device);
            Tensor qValuesAllHeads;

            using (torch.no_grad())
            {
                qValuesAllHeads = agent.policyNet.forward(stateTensor).view(states.Length, agent.ActionSizes.Length, agent.ActionSizes[0], agent.Options.NumAtoms);
            }

            Tensor expectedQValues = (qValuesAllHeads * agent.support).sum(dim: -1);

            int[][] actions = new int[states.Length][];

            for (int i = 0; i < states.Length; i++)
            {
                Tensor bestActions = expectedQValues[i].argmax(dim: -1).squeeze().to(ScalarType.Int32);
                actions[i] = bestActions.data<int>().ToArray();
            }

            return actions;
        }
        #endregion

        #region BatchedBoltzmannActionSelection
        private static int[][] VanillaActionSelectionBatchedBoltzmann(T[] states, ComposableQDiscreteAgent<T> agent, bool isTraining)
        {
            double epsThreshold = agent.Options.EPS_END + (agent.Options.EPS_START - agent.Options.EPS_END) *
                Math.Exp(-1.0 * agent.episodeCount / agent.Options.EPS_DECAY);

            Tensor stateTensor = Utilities<T>.StateBatchToTensor(states, agent.Device);
            Tensor qValuesAllHeads;

            using (torch.no_grad())
            {
                qValuesAllHeads = agent.policyNet.forward(stateTensor).view(states.Length, agent.ActionSizes.Length, agent.ActionSizes[0]);
            }

            qValuesAllHeads = torch.where(qValuesAllHeads.isnan(), torch.zeros_like(qValuesAllHeads), qValuesAllHeads);
            float temperature = 1.0f;
            int[][] actions = new int[states.Length][];

            for (int i = 0; i < states.Length; i++)
            {
                actions[i] = new int[agent.ActionSizes.Length];

                for (int j = 0; j < agent.ActionSizes.Length; j++)
                {
                    if (agent.Random.NextDouble() > epsThreshold || !isTraining)
                    {
                        Tensor scaledQValues = qValuesAllHeads[i, j] / temperature;
                        Tensor maxQValue = scaledQValues.max();
                        Tensor actionProbs = torch.softmax(scaledQValues - maxQValue, dim: -1);
                        actionProbs = torch.clamp(actionProbs, 1e-10f, 1.0f);
                        actionProbs /= actionProbs.sum();
                        Tensor sampledAction = torch.multinomial(actionProbs, num_samples: 1, replacement: true);
                        actions[i][j] = (int)sampledAction.item<long>();
                    }
                    else
                    {
                        actions[i][j] = agent.Random.Next(0, agent.ActionSizes[j]);
                    }
                }
            }

            return actions;
        }

        private static int[][] NoisyActionSelectionBatchedBoltzmann(T[] states, ComposableQDiscreteAgent<T> agent, bool isTraining)
        {
            if (isTraining)
            {
                agent.policyNet.train();
                agent.ResetNoisyLayers();
            }
            else
            {
                agent.policyNet.eval();
            }

            Tensor stateTensor = Utilities<T>.StateBatchToTensor(states, agent.Device);
            Tensor qValuesAllHeads;

            using (torch.no_grad())
            {
                qValuesAllHeads = agent.policyNet.forward(stateTensor).view(states.Length, agent.ActionSizes.Length, agent.ActionSizes[0]);
            }

            qValuesAllHeads = torch.where(qValuesAllHeads.isnan(), torch.zeros_like(qValuesAllHeads), qValuesAllHeads);
            float temperature = 1.0f;
            int[][] actions = new int[states.Length][];

            for (int i = 0; i < states.Length; i++)
            {
                actions[i] = new int[agent.ActionSizes.Length];

                for (int j = 0; j < agent.ActionSizes.Length; j++)
                {
                    Tensor scaledQValues = qValuesAllHeads[i, j] / temperature;
                    Tensor maxQValue = scaledQValues.max();
                    Tensor actionProbs = torch.softmax(scaledQValues - maxQValue, dim: -1);
                    actionProbs = torch.clamp(actionProbs, 1e-10f, 1.0f);
                    actionProbs /= actionProbs.sum();
                    Tensor sampledAction = torch.multinomial(actionProbs, num_samples: 1, replacement: true);
                    actions[i][j] = (int)sampledAction.item<long>();
                }
            }

            return actions;
        }

        private static int[][] CategoricalActionSelectionBatchedBoltzmann(T[] states, ComposableQDiscreteAgent<T> agent, bool isTraining)
        {
            Tensor stateTensor = Utilities<T>.StateBatchToTensor(states, agent.Device);
            Tensor qValuesAllHeads;

            using (torch.no_grad())
            {
                qValuesAllHeads = agent.policyNet.forward(stateTensor).view(states.Length, agent.ActionSizes.Length, agent.ActionSizes[0], agent.Options.NumAtoms);
            }

            Tensor expectedQValues = (qValuesAllHeads * agent.support).sum(dim: -1);
            expectedQValues = torch.where(expectedQValues.isnan(), torch.zeros_like(expectedQValues), expectedQValues);

            float temperature = 1.0f;
            int[][] actions = new int[states.Length][];

            for (int i = 0; i < states.Length; i++)
            {
                actions[i] = new int[agent.ActionSizes.Length];

                for (int j = 0; j < agent.ActionSizes.Length; j++)
                {
                    Tensor scaledQValues = expectedQValues[i, j] / temperature;
                    Tensor maxQValue = scaledQValues.max();
                    Tensor actionProbs = torch.softmax(scaledQValues - maxQValue, dim: -1);
                    actionProbs = torch.clamp(actionProbs, 1e-10f, 1.0f);
                    actionProbs /= actionProbs.sum();
                    Tensor sampledAction = torch.multinomial(actionProbs, num_samples: 1, replacement: true);
                    actions[i][j] = (int)sampledAction.item<long>();
                }
            }

            return actions;
        }

        private static int[][] CategoricalNoisyActionSelectionBatchedBoltzmann(T[] states, ComposableQDiscreteAgent<T> agent, bool isTraining)
        {
            if (isTraining)
            {
                agent.policyNet.train();
                agent.ResetNoisyLayers();
            }
            else
            {
                agent.policyNet.eval();
            }

            Tensor stateTensor = Utilities<T>.StateBatchToTensor(states, agent.Device);
            Tensor qValuesAllHeads;
            using (torch.no_grad())
            {
                qValuesAllHeads = agent.policyNet.forward(stateTensor).view(states.Length, agent.ActionSizes.Length, agent.ActionSizes[0], agent.Options.NumAtoms);
            }

            Tensor expectedQValues = (qValuesAllHeads * agent.support).sum(dim: -1);
            expectedQValues = torch.where(expectedQValues.isnan(), torch.zeros_like(expectedQValues), expectedQValues);

            float temperature = 1.0f;
            int[][] actions = new int[states.Length][];

            for (int i = 0; i < states.Length; i++)
            {
                actions[i] = new int[agent.ActionSizes.Length];
                for (int j = 0; j < agent.ActionSizes.Length; j++)
                {
                    Tensor scaledQValues = expectedQValues[i, j] / temperature;
                    Tensor maxQValue = scaledQValues.max();
                    Tensor actionProbs = torch.softmax(scaledQValues - maxQValue, dim: -1);
                    actionProbs = torch.clamp(actionProbs, 1e-10f, 1.0f);
                    actionProbs /= actionProbs.sum();
                    Tensor sampledAction = torch.multinomial(actionProbs, num_samples: 1, replacement: true);
                    actions[i][j] = (int)sampledAction.item<long>();
                }
            }

            return actions;
        }
        #endregion


        private static int[] ActionsFromState(T state, Module<Tensor, Tensor> policyNet, int[] ActionSizes, Device device)
        {
            using (torch.no_grad())
            {
                Tensor stateTensor = Utilities<T>.StateToTensor(state, device);
                Tensor qValuesAllHeads = policyNet.forward(stateTensor).view(1, ActionSizes.Length, ActionSizes[0]);
                Tensor bestActions = qValuesAllHeads.argmax(dim: -1).squeeze().to(ScalarType.Int32);
                var result = bestActions.data<int>().ToArray();
                return result;
            }
        }

        private static int[] CategoricalActionsFromState(T state, Module<Tensor, Tensor> policyNet, int[] ActionSizes, int numAtoms, Device device, Tensor support)
        {
            using (torch.no_grad())
            {
                Tensor stateTensor = Utilities<T>.StateToTensor(state, device);
                Tensor qValuesAllHeads = policyNet.forward(stateTensor).view(1, ActionSizes.Length, ActionSizes[0], numAtoms);
                Tensor expectedQValues = (qValuesAllHeads * support).sum(dim: -1);
                Tensor bestActions = expectedQValues.argmax(dim: -1).squeeze().to(ScalarType.Int32);
                return bestActions.data<int>().ToArray();
            }
        }

        private static int[] RandomActions(int[] actionSizes, Random Random)
        {
          
            return actionSizes.Select(size => Random.Next(0, size)).ToArray();
        }

        private static Tensor GetSupport(int numAtoms, float vMin, float vMax, Device device)
        {
            float deltaZ = (vMax - vMin) / (numAtoms - 1);
            return torch.linspace(vMin, vMax, steps: numAtoms, device: device);
        }
        #endregion

    }
}
