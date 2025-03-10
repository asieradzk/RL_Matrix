using RLMatrix.Common;

namespace RLMatrix;

public class ContinuousPPOAgent<TState> : IContinuousPPOAgent<TState>
    where TState : notnull
{
    public ContinuousPPOAgent(PPOActorNet actorNet, PPOCriticNet criticNet, IOptimizer<TState> optimizer, IMemory<TState> memory, 
        int[] discreteActionDimensions, ContinuousActionDimensions[] continuousActionDimensions, PPOAgentOptions options, Device device)
    {
        ActorNet = actorNet;
        CriticNet = criticNet;
        Optimizer = optimizer;
        Memory = memory;
        DiscreteActionDimensions = discreteActionDimensions;
        ContinuousActionDimensions = continuousActionDimensions;
        Options = options;
        Device = device;
    }

    public PPOActorNet ActorNet { get; }
    
    public PPOCriticNet CriticNet { get; }
    
    public IOptimizer<TState> Optimizer { get; }
    
    public IMemory<TState> Memory { get; }
    
    public int[] DiscreteActionDimensions { get; }
    
    public ContinuousActionDimensions[] ContinuousActionDimensions { get; }
    
    public PPOAgentOptions Options { get; }
    
    public Device Device { get; }

    public ValueTask AddTransitionsAsync(IEnumerable<Transition<TState>> transitions)
    {
        return Memory.PushAsync(Utilities<TState>.ConvertToMemoryTransitions(transitions));
    }

    public ValueTask OptimizeModelAsync()
    {
        return Optimizer.OptimizeAsync(Memory);
    }

    public virtual ValueTask<RLActions[]> SelectActionsAsync(TState[] states, bool isTraining)
    {
        using (torch.no_grad())
        {
            var stateTensor = Utilities<TState>.StateBatchToTensor(states, Device);
            var result = ActorNet.forward(stateTensor);
            var actions = new RLActions[states.Length];

            for (var i = 0; i < states.Length; i++)
            {
                if (isTraining)
                {
                    actions[i] = RLActions.Continuous(
                        PPOActionSelection.SelectDiscreteActionsFromProbabilities(result[i].unsqueeze(0), DiscreteActionDimensions),
                        PPOActionSelection.SampleContinuousActions(result[i].unsqueeze(0), DiscreteActionDimensions, ContinuousActionDimensions));
                }
                else
                {
                    actions[i] = RLActions.Continuous(
                        PPOActionSelection.SelectGreedyDiscreteActions(result[i].unsqueeze(0), DiscreteActionDimensions),
                        PPOActionSelection.SelectMeanContinuousActions(result[i].unsqueeze(0), DiscreteActionDimensions, ContinuousActionDimensions));
                }
            }

            return new(actions);
        }
    }

    /* TODO: unused
    private float Clamp(float value, float min, float max)
    {
        return value < min ? min : value > max ? max : value;
    }
    */

    // TODO: alternative non-batch?
    public ValueTask<RLActions[]> SelectActions2(TState[] states, bool isTraining)
    {
        var result = new RLActions[states.Length];

        for (var i = 0; i < states.Length; i++)
        {
            using (torch.no_grad())
            {
                var stateTensor = Utilities<TState>.StateToTensor(states[i], Device);
                var forwardResult = ActorNet.forward(stateTensor);

                if (isTraining)
                {
                    result[i] = RLActions.Continuous(
                        PPOActionSelection.SelectDiscreteActionsFromProbabilities(forwardResult, DiscreteActionDimensions),
                        PPOActionSelection.SampleContinuousActions(forwardResult, DiscreteActionDimensions, ContinuousActionDimensions));
                }
                else
                {
                    result[i] = RLActions.Continuous(
                        PPOActionSelection.SelectGreedyDiscreteActions(forwardResult, DiscreteActionDimensions),
                        PPOActionSelection.SelectMeanContinuousActions(forwardResult, DiscreteActionDimensions, ContinuousActionDimensions));
                }
            }
        }

        return new(result);
    }

    public virtual ValueTask<ActionsState[]> SelectActionsRecurrentAsync(RLMemoryState<TState>[] states, bool isTraining)
    {
        throw new NotSupportedException($"Using recurrent action selection with non recurrent agent, use {nameof(SelectActionsAsync)} instead.");
    }

    public ValueTask SaveAsync(string path)
    {
        var modelPath = path.EndsWith(Path.DirectorySeparatorChar.ToString()) ? path : path + Path.DirectorySeparatorChar;

        var actorNetPath = GetNextAvailableModelPath(modelPath, "modelActor");
        ActorNet.save(actorNetPath);

        var criticNetPath = GetNextAvailableModelPath(modelPath, "modelCritic");
        CriticNet.save(criticNetPath);
        return new();
    }

    // TODO: what to do when scheduler is null?
    public async ValueTask LoadAsync(string path, LRScheduler? scheduler = null)
    {
        var modelPath = path.EndsWith(Path.DirectorySeparatorChar.ToString()) ? path : path + Path.DirectorySeparatorChar;

        var actorNetPath = GetLatestModelPath(modelPath, "modelActor");
        ActorNet.load(actorNetPath, strict: true);

        var criticNetPath = GetLatestModelPath(modelPath, "modelCritic");
        CriticNet.load(criticNetPath, strict: true);

        await Optimizer.UpdateOptimizersAsync(scheduler!);
    }
    
    private string GetNextAvailableModelPath(string modelPath, string modelName)
    {
        var files = Directory.GetFiles(modelPath);

        var maxNumber = files
            .Where(file => file.Contains(modelName))
            .Select(file => Path.GetFileNameWithoutExtension(file).Split('_').LastOrDefault() ?? string.Empty)
            .Where(number => int.TryParse(number, out _))
            .DefaultIfEmpty("0")
            .Max(int.Parse);

        var nextModelPath = $"{modelPath}{modelName}_{maxNumber + 1}";
        return nextModelPath;
    }

    private string GetLatestModelPath(string modelPath, string modelName)
    {
        var files = Directory.GetFiles(modelPath);

        var maxNumber = files
            .Where(file => file.Contains(modelName))
            .Select(file => Path.GetFileNameWithoutExtension(file).Split('_').LastOrDefault() ?? string.Empty)
            .Where(number => int.TryParse(number, out _))
            .DefaultIfEmpty("0")
            .Max(int.Parse);

        var latestModelPath = $"{modelPath}{modelName}_{maxNumber}";

        return latestModelPath;
    }

    ValueTask ISavable.LoadAsync(string path) => LoadAsync(path);
}