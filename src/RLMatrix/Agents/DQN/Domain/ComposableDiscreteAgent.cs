using Newtonsoft.Json;
using RLMatrix.Agents.Common;
using RLMatrix.Agents.DQN.Domain;
using RLMatrix.Memories;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim.lr_scheduler;
using static TorchSharp.torchvision;

namespace RLMatrix
{
    public interface IDiscreteAgent<T>
    {
        void Step(bool isTraining);
    }

    public class ComposableQDiscreteAgent<T> : IDiscreteAgentCore<T>, IHasMemory<T>, ISelectActions<T>, IHasOptimizer<T>, ISavable
    {
#if NET8_0_OR_GREATER
        public required Module<Tensor, Tensor> policyNet { get; set; }
        public required Module<Tensor, Tensor> targetNet { get; set; }
        public required OptimizerHelper optimizer { private get; init; }
        public required IOptimize<T> Optimizer { get; init; }
        public required IMemory<T> Memory { get; set; }
        public required int[] ActionSizes { get; init; }
        public required Action ResetNoisyLayers { get; init; }
        public required DQNAgentOptions Options { get; init; }
        public required Device Device { get; init; }
        public required Func<T[], ComposableQDiscreteAgent<T>, bool, int[][]> SelectActionsFunc { private get; init; }
#else
        public Module<Tensor, Tensor> policyNet { get; set; }
        public Module<Tensor, Tensor> targetNet { get; set; }
        public OptimizerHelper optimizer { private get; set; }
        public IOptimize<T> Optimizer { get; set; }
        public IMemory<T> Memory { get; set; }
        public int[] ActionSizes { get; set; }
        public Action ResetNoisyLayers { get; set; }
        public DQNAgentOptions Options { get; set; }
        public Device Device { get; set; }
        public Func<T[], ComposableQDiscreteAgent<T>, bool, int[][]> SelectActionsFunc { private get; set; }
#endif

        public Random Random = new Random();
        public Tensor? support { get; set; }
        public ulong episodeCount = 0;

        public void AddTransition(IEnumerable<TransitionPortable<T>> transitions)
        {
            var transitionsInMemory = transitions.ToTransitionInMemory();


            Memory.Push(transitionsInMemory);
            episodeCount++;
        }

        public void OptimizeModel()
        {
            Optimizer.Optimize(Memory);
        }

        public int[][] SelectActions(T[] states, bool isTraining)
        {
            return SelectActionsFunc(states, this, isTraining);
        }

        public void Save(string path)
        {
            var modelPath = path.EndsWith(Path.DirectorySeparatorChar.ToString()) ? path : path + Path.DirectorySeparatorChar;

            string policyNetPath = GetNextAvailableModelPath(modelPath, "modelPolicy");
            policyNet.save(policyNetPath);

            string targetNetPath = GetNextAvailableModelPath(modelPath, "modelTarget");
            targetNet.save(targetNetPath);
        }

        private string GetNextAvailableModelPath(string modelPath, string modelName)
        {
            var files = Directory.GetFiles(modelPath);

            int maxNumber = files
                .Where(file => file.Contains(modelName))
                .Select(file => Path.GetFileNameWithoutExtension(file).Split('_').LastOrDefault())
                .Where(number => int.TryParse(number, out _))
                .DefaultIfEmpty("0")
                .Max(number => int.Parse(number));

            string nextModelPath = $"{modelPath}{modelName}_{maxNumber + 1}";

            return nextModelPath;
        }

        public void Load(string path, LRScheduler scheduler = null)
        {
            var modelPath = path.EndsWith(Path.DirectorySeparatorChar.ToString()) ? path : path + Path.DirectorySeparatorChar;

            string policyNetPath = GetLatestModelPath(modelPath, "modelPolicy");
            string targetNetPath = GetLatestModelPath(modelPath, "modelTarget");
            policyNet.load(policyNetPath, true);

            targetNet.load(targetNetPath, true);

            Optimizer.UpdateOptimizers(scheduler);
        }

        private string GetLatestModelPath(string modelPath, string modelName)
        {
            var files = Directory.GetFiles(modelPath);

            int maxNumber = files
                .Where(file => file.Contains(modelName))
                .Select(file => Path.GetFileNameWithoutExtension(file).Split('_').LastOrDefault())
                .Where(number => int.TryParse(number, out _))
                .DefaultIfEmpty("0")
                .Max(number => int.Parse(number));

            string latestModelPath = $"{modelPath}{modelName}_{maxNumber}";

            return latestModelPath;
        }
    }

    public interface IDiscreteQAgentFactory<T>
    {
        ComposableQDiscreteAgent<T> ComposeAgent(DQNAgentOptions options);
    }
}