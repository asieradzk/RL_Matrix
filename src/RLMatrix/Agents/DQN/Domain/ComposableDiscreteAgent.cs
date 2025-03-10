using RLMatrix.Common;

namespace RLMatrix;

public class ComposableQDiscreteAgent<TState> : IDiscreteAgent<TState>
    where TState : notnull
{
    // TODO: OptimizerHelper goes unused even in the original impl
    // private readonly OptimizerHelper _optimizerHelper;
    private readonly Func<TState[], ComposableQDiscreteAgent<TState>, bool, RLActions[]> _selectActionsFunc;

    public ComposableQDiscreteAgent(TensorModule policyNet, TensorModule targetNet, IOptimizer<TState> optimizer, 
        IMemory<TState> memory, int[] actionDimensions, Action resetNoisyLayers, DQNAgentOptions options, Device device, Tensor? support,
        /* OptimizerHelper optimizerHelper, */ Func<TState[], ComposableQDiscreteAgent<TState>, bool, RLActions[]> selectActionsFunc)
    {
        // _optimizerHelper = optimizerHelper;
        _selectActionsFunc = selectActionsFunc;
        
        PolicyNet = policyNet;
        TargetNet = targetNet;
        Optimizer = optimizer;
        Memory = memory;
        DiscreteActionDimensions = actionDimensions;
        ResetNoisyLayers = resetNoisyLayers;
        Options = options;
        Device = device;
        Support = support;
    }

    public TensorModule PolicyNet { get; }
    
    public TensorModule TargetNet { get; }
    
    public IOptimizer<TState> Optimizer { get; }
    
    public IMemory<TState> Memory { get; }
    
    public int[] DiscreteActionDimensions { get; }
    
    public Action ResetNoisyLayers { get; }
    
    public DQNAgentOptions Options { get; }
    
    public Device Device { get; }

    public Tensor? Support { get; }

    public ulong EpisodeCount { get; private set; }

    public Random Random { get; } = new();

    public async ValueTask AddTransitionsAsync(IEnumerable<Transition<TState>> transitions)
    {
        var transitionsInMemory = Utilities<TState>.ConvertToMemoryTransitions(transitions);

        await Memory.PushAsync(transitionsInMemory);
        EpisodeCount++;
    }

    public ValueTask OptimizeModelAsync()
    {
        return Optimizer.OptimizeAsync(Memory);
    }

    public ValueTask<RLActions[]> SelectActionsAsync(TState[] states, bool isTraining)
    {
        return new(_selectActionsFunc(states, this, isTraining).ToArray());
    }

    public ValueTask SaveAsync(string path)
    {
        var modelPath = path.EndsWith(Path.DirectorySeparatorChar.ToString()) ? path : path + Path.DirectorySeparatorChar;

        var policyNetPath = GetNextAvailableModelPath(modelPath, "modelPolicy");
        PolicyNet.save(policyNetPath);

        var targetNetPath = GetNextAvailableModelPath(modelPath, "modelTarget");
        TargetNet.save(targetNetPath);
        return new();
    }

    public ValueTask LoadAsync(string path, LRScheduler? scheduler = null)
    {
        var modelPath = path.EndsWith(Path.DirectorySeparatorChar.ToString()) ? path : path + Path.DirectorySeparatorChar;

        var policyNetPath = GetLatestModelPath(modelPath, "modelPolicy");
        var targetNetPath = GetLatestModelPath(modelPath, "modelTarget");
        
        PolicyNet.load(policyNetPath);
        TargetNet.load(targetNetPath);

        // TODO: what to do when scheduler is null?
        return Optimizer.UpdateOptimizersAsync(scheduler!);
    }
    
    private static string GetNextAvailableModelPath(string modelPath, string modelName)
    {
        var files = Directory.GetFiles(modelPath);

        var maxNumber = files
            .Where(file => file.Contains(modelName))
            .Select(file => Path.GetFileNameWithoutExtension(file).Split('_').LastOrDefault())
            .Where(number => int.TryParse(number, out _))
            .DefaultIfEmpty("0")
            .Max(number => int.Parse(number!));

        var nextModelPath = $"{modelPath}{modelName}_{maxNumber + 1}";

        return nextModelPath;
    }

    private static string GetLatestModelPath(string modelPath, string modelName)
    {
        var files = Directory.GetFiles(modelPath);

        var maxNumber = files
            .Where(file => file.Contains(modelName))
            .Select(file => Path.GetFileNameWithoutExtension(file).Split('_').LastOrDefault())
            .Where(number => int.TryParse(number, out _))
            .DefaultIfEmpty("0")
            .Max(number => int.Parse(number!));

        var latestModelPath = $"{modelPath}{modelName}_{maxNumber}";

        return latestModelPath;
    }

    ValueTask ISavable.LoadAsync(string path) => LoadAsync(path);
}

public interface IDiscreteQAgentFactory<TState>
    where TState : notnull
{
    ComposableQDiscreteAgent<TState> ComposeAgent(DQNAgentOptions options);
}