using OneOf;
using RLMatrix.Agents.DQN.Domain;
using TorchSharp;
using static TorchSharp.torch;
namespace RLMatrix.Agents.PPO.Implementations
{
    public static class PPOAgentFactory<T>
    {
        public static IDiscretePPOAgent<T> ComposeDiscretePPOAgent(PPOAgentOptions options, int[] ActionSizes, OneOf<int, (int, int)> StateSizes, IPPONetProvider<T> netProvider = null, IGAIL<T> gail = null)
        {
            netProvider ??= new PPONetProviderBase<T>(options.Width, options.Depth, options.UseRNN);

            var device = torch.device(torch.cuda.is_available() ? "cuda" : "cpu");
            var envSizeDTO = new DiscreteEnvSizeDTO { actionSize = ActionSizes, stateSize = StateSizes };
            var actorNet = netProvider.CreateActorNet(envSizeDTO).to(device);
            var criticNet = netProvider.CreateCriticNet(envSizeDTO).to(device);
            var actorOptimizer = optim.Adam(actorNet.parameters(), lr: options.LR, amsgrad: true);
            var criticOptimizer = optim.Adam(criticNet.parameters(), lr: options.LR, amsgrad: true);

            //var actorLrScheduler = new optim.lr_scheduler.impl.StepLR(actorOptimizer, step_size: 1000, gamma: 0.9f);
            //var criticlLrScheduler = new optim.lr_scheduler.impl.StepLR(actorOptimizer, step_size: 1000, gamma: 0.9f);
             var actorLrScheduler = new optim.lr_scheduler.impl.CyclicLR(actorOptimizer, options.LR * 0.5f, options.LR * 2f, step_size_up: 10, step_size_down: 10, cycle_momentum: false);
            var criticlLrScheduler = new optim.lr_scheduler.impl.CyclicLR(criticOptimizer, options.LR * 0.5f, options.LR * 2f, step_size_up: 10, step_size_down: 10, cycle_momentum: false);
            var PPOOptimize = new PPOOptimize<T>(actorNet, criticNet, actorOptimizer, criticOptimizer, options, device, ActionSizes, new (float, float)[0], actorLrScheduler, criticlLrScheduler, gail);
            if (options.UseRNN)
            {
                var Agent = new DiscreteRecurrentPPOAgent<T>
                {
                    actorNet = actorNet,
                    criticNet = criticNet,
                    Optimizer = PPOOptimize,
                    Memory = new ReplayMemory<T>(options.MemorySize),
                    ActionSizes = ActionSizes,
                    Options = options,
                    Device = device,
                };
                return Agent;
            }
            else
            {
                var Agent = new DiscretePPOAgent<T>
                {
                    actorNet = actorNet,
                    criticNet = criticNet,
                    Optimizer = PPOOptimize,
                    Memory = new ReplayMemory<T>(options.MemorySize),
                    ActionSizes = ActionSizes,
                    Options = options,
                    Device = device,
                };
                return Agent;
            }
        }

        public static IContinuousPPOAgent<T> ComposeContinuousPPOAgent(PPOAgentOptions options, int[] DiscreteDimensions, OneOf<int, (int, int)> StateSizes, (float min, float max)[] ContinuousActionBounds, IPPONetProvider<T> netProvider = null, IGAIL<T> gail = null)
        {
            netProvider ??= new PPONetProviderBase<T>(options.Width, options.Depth, options.UseRNN);

            var device = torch.device(torch.cuda.is_available() ? "cuda" : "cpu");
            var envSizeDTO = new ContinuousEnvSizeDTO { actionSize = DiscreteDimensions, continuousActionBounds = ContinuousActionBounds, stateSize = StateSizes };
            var actorNet = netProvider.CreateActorNet(envSizeDTO).to(device);
            var criticNet = netProvider.CreateCriticNet(envSizeDTO).to(device);
            var actorOptimizer = optim.Adam(actorNet.parameters(), lr: options.LR, amsgrad: true);
            var criticOptimizer = optim.Adam(criticNet.parameters(), lr: options.LR, amsgrad: true);

            var actorLrScheduler = new optim.lr_scheduler.impl.CyclicLR(actorOptimizer, options.LR * 0.5f, options.LR * 2f, step_size_up: 10, step_size_down: 10, cycle_momentum: false);
            var criticlLrScheduler = new optim.lr_scheduler.impl.CyclicLR(criticOptimizer, options.LR * 0.5f, options.LR * 2f, step_size_up: 10, step_size_down: 10, cycle_momentum: false);
            var PPOOptimize = new PPOOptimize<T>(actorNet, criticNet, actorOptimizer, criticOptimizer, options, device, DiscreteDimensions, ContinuousActionBounds, actorLrScheduler, criticlLrScheduler, gail);

            if (options.UseRNN)
            {
                var Agent = new ContinuousRecurrentPPOAgent<T>
                {
                    actorNet = actorNet,
                    criticNet = criticNet,
                    Optimizer = PPOOptimize,
                    Memory = new ReplayMemory<T>(options.MemorySize),
                    DiscreteDimensions = DiscreteDimensions,
                    ContinuousActionBounds = ContinuousActionBounds,
                    Options = options,
                    Device = device,
                };
                return Agent;
            }
            else
            {
                var Agent = new ContinuousPPOAgent<T>
                {
                    actorNet = actorNet,
                    criticNet = criticNet,
                    Optimizer = PPOOptimize,
                    Memory = new ReplayMemory<T>(options.MemorySize),
                    DiscreteDimensions = DiscreteDimensions,
                    ContinuousActionBounds = ContinuousActionBounds,
                    Options = options,
                    Device = device,
                };
                return Agent;
            }
        }
    }
}