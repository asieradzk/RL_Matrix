using Newtonsoft.Json;
using RLMatrix.Agents.DQN.Domain;
using RLMatrix.Memories;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torchvision;

namespace RLMatrix
{
    public interface IDiscreteAgent<T> {
        public void Step(bool isTraining);

    }

    public interface IHasMemory<T>
    {
        public IMemory<T> Memory { get; set; }
        public void AddTransition(IEnumerable<TransitionPortable<T>> transitions);
    }
    public interface ISelectActions<T>
    {
        public int[][] SelectActions(T[] states, bool isTraining);
    }
    public interface IHasOptimizer<T>
    {
        public IOptimize<T> Optimizer { get; init; }
    }

    public interface IDiscreteAgentCore<T>
    {
        public int[] ActionSizes { get; init; }
        public int[][] SelectActions(T[] states, bool isTraining);
        public void OptimizeModel();
    }

    public class ComposableQDiscreteAgent<T> : IDiscreteAgentCore<T>, IHasMemory<T>, ISelectActions<T>, IHasOptimizer<T>
    {
        public required Module<Tensor, Tensor> policyNet { get; set; }
        public required Module<Tensor, Tensor> targetNet { get; set; }
        public required OptimizerHelper optimizer { private get; init;}
        public required IOptimize<T> Optimizer { get; init; }
        public required IMemory<T> Memory { get; set; }
        public required int[] ActionSizes { get; init; }
        public required Action ResetNoisyLayers { get; init; }
        public required DQNAgentOptions Options { get; init; }
        public required Device Device { get; init; }
        public Random Random = new Random();
        public Tensor? support { get; set; }
        public void AddTransition(IEnumerable<TransitionPortable<T>> transitions)
        {
            Memory.Push(transitions.ToTransitionInMemory());
            episodeCount++;
        }

        public void OptimizeModel()
        {
            Optimizer.Optimize(Memory);
        }
        //TODO: Episode count for random action selection. Maybe belongs somewhere deeper and can depend on iterations of "select action batch"
        public ulong episodeCount = 0;
        public required Func<T[], ComposableQDiscreteAgent<T>, bool, int[][]> SelectActionsFunc { private get; init; }


        public int[][] SelectActions(T[] states, bool isTraining)
        {
            return SelectActionsFunc(states, this, isTraining);
        }
    }



    public interface IDiscreteProxy<T>
    {
        public ValueTask<Dictionary<Guid, int[]>> SelectActionsBatchAsync(List<(Guid environmentId, T state)> stateInfos);
        public ValueTask UploadTransitionsAsync(IEnumerable<TransitionPortable<T>> transitions);
        public ValueTask OptimizeModelAsync();
    }

    public interface IDiscreteQAgentFactory<T>
    {
        public ComposableQDiscreteAgent<T> ComposeAgent(DQNAgentOptions options);
    }
}
