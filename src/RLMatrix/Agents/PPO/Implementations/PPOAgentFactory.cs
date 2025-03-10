using RLMatrix.Common;

namespace RLMatrix;

public static class PPOAgentFactory
{
    public static IDiscretePPOAgent<TState> ComposeDiscretePPOAgent<TState>(PPOAgentOptions options, int[] discreteActionDimensions, StateDimensions stateDimensions, IPPONetProvider? netProvider = null, IGAIL<TState>? gail = null)
        where TState : notnull
    {
        netProvider ??= new PPONetProviderBase<TState>(options.Width, options.Depth, options.UseRNN);

        var device = torch.device(torch.cuda.is_available() ? "cuda" : "cpu");
        var envSize = new DiscreteEnvironmentSizeDTO(discreteActionDimensions, stateDimensions);
        var actorNet = netProvider.CreateActorNet(envSize).to(device);
        var criticNet = netProvider.CreateCriticNet(envSize).to(device);
        var actorOptimizer = torch.optim.Adam(actorNet.parameters(), lr: options.LearningRate, amsgrad: true);
        var criticOptimizer = torch.optim.Adam(criticNet.parameters(), lr: options.LearningRate, amsgrad: true);

        //var actorLrScheduler = new optim.lr_scheduler.impl.StepLR(actorOptimizer, step_size: 1000, gamma: 0.9f);
        //var criticlLrScheduler = new optim.lr_scheduler.impl.StepLR(actorOptimizer, step_size: 1000, gamma: 0.9f);
        var actorLrScheduler = new CyclicLR(actorOptimizer, options.LearningRate * 0.5f, options.LearningRate * 2f, step_size_up: 10, step_size_down: 10, cycle_momentum: false);
        var criticLrScheduler = new CyclicLR(criticOptimizer, options.LearningRate * 0.5f, options.LearningRate * 2f, step_size_up: 10, step_size_down: 10, cycle_momentum: false);
        var ppoOptimizer = new PPOOptimizer<TState>(actorNet, criticNet, actorOptimizer, criticOptimizer, options, device, discreteActionDimensions, [], actorLrScheduler, criticLrScheduler, gail);
        
        if (options.UseRNN)
        {
            return new DiscreteRecurrentPPOAgent<TState>(
                actorNet, criticNet, ppoOptimizer, new ReplayMemory<TState>(options.MemorySize), discreteActionDimensions, options, device);
        }

        return new DiscretePPOAgent<TState>(
            actorNet, criticNet, ppoOptimizer, new ReplayMemory<TState>(options.MemorySize), discreteActionDimensions, options, device);
    }

    public static IContinuousPPOAgent<TState> ComposeContinuousPPOAgent<TState>(PPOAgentOptions options, int[] discreteActionDimensions, StateDimensions stateDimensions, ContinuousActionDimensions[] continuousActionDimensions, IPPONetProvider? netProvider = null, IGAIL<TState>? gail = null)
        where TState : notnull
    {
        netProvider ??= new PPONetProviderBase<TState>(options.Width, options.Depth, options.UseRNN);

        var device = torch.device(torch.cuda.is_available() ? "cuda" : "cpu");
        var envSize = new ContinuousEnvironmentSizeDTO(discreteActionDimensions, stateDimensions, continuousActionDimensions);
        var actorNet = netProvider.CreateActorNet(envSize).to(device);
        var criticNet = netProvider.CreateCriticNet(envSize).to(device);
        var actorOptimizer = torch.optim.Adam(actorNet.parameters(), lr: options.LearningRate, amsgrad: true);
        var criticOptimizer = torch.optim.Adam(criticNet.parameters(), lr: options.LearningRate, amsgrad: true);

        var actorLrScheduler = new CyclicLR(actorOptimizer, options.LearningRate * 0.5f, options.LearningRate * 2f, step_size_up: 10, step_size_down: 10, cycle_momentum: false);
        var criticLrScheduler = new CyclicLR(criticOptimizer, options.LearningRate * 0.5f, options.LearningRate * 2f, step_size_up: 10, step_size_down: 10, cycle_momentum: false);
        var ppoOptimizer = new PPOOptimizer<TState>(actorNet, criticNet, actorOptimizer, criticOptimizer, options, device, discreteActionDimensions, continuousActionDimensions, actorLrScheduler, criticLrScheduler, gail);

        if (options.UseRNN)
        {
            return new ContinuousRecurrentPPOAgent<TState>(
                actorNet, criticNet, ppoOptimizer, new ReplayMemory<TState>(options.MemorySize), discreteActionDimensions, continuousActionDimensions, options, device);
        }

        return new ContinuousPPOAgent<TState>(
            actorNet, criticNet, ppoOptimizer, new ReplayMemory<TState>(options.MemorySize), discreteActionDimensions, continuousActionDimensions, options, device);
    }
}