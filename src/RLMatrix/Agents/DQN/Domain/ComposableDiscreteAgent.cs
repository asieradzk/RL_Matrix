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

    //TODO: for removal
    public interface IDiscreteAgent<T> {
        public void Step(bool isTraining);

    }

    public class ComposableQDiscreteAgent<T> : IDiscreteAgentCore<T>, IHasMemory<T>, ISelectActions<T>, IHasOptimizer<T>, ISavable
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

        public void Save(string path)
        {
            // Check if path ends with "/", if not, append it
            var modelPath = path.EndsWith(@"\") ? path : path + @"\";

            // Save the policy network
            string policyNetPath = GetNextAvailableModelPath(modelPath, "modelPolicy");
            policyNet.save(policyNetPath);

            // Save the target network
            string targetNetPath = GetNextAvailableModelPath(modelPath, "modelTarget");
            targetNet.save(targetNetPath);
        }

        private string GetNextAvailableModelPath(string modelPath, string modelName)
        {
            // Read all files in the directory
            var files = Directory.GetFiles(modelPath);

            // Find the highest number of files with the same name
            int maxNumber = files
                .Where(file => file.Contains(modelName))
                .Select(file => Path.GetFileNameWithoutExtension(file).Split("_").LastOrDefault())
                .Where(number => int.TryParse(number, out _))
                .DefaultIfEmpty("0")
                .Max(number => int.Parse(number));

            // Append the next number to the model name
            string nextModelPath = $"{modelPath}{modelName}_{maxNumber + 1}";

            return nextModelPath;
        }

        public void Load(string path, LRScheduler scheduler = null)
        {
            // Check if path ends with "/", if not, append it
            var modelPath = path.EndsWith(@"\") ? path : path + @"\";

            // Load the policy network
            string policyNetPath = GetLatestModelPath(modelPath, "modelPolicy");
            string targetNetPath = GetLatestModelPath(modelPath, "modelTarget");
            policyNet.load(policyNetPath, true);

            // Load the target network
           
            targetNet.load(targetNetPath, true);

            Optimizer.UpdateOptimizers(scheduler);

        }

        private string GetLatestModelPath(string modelPath, string modelName)
        {
            // Read all files in the directory
            var files = Directory.GetFiles(modelPath);

            // Find the highest number of files with the same name
            int maxNumber = files
                .Where(file => file.Contains(modelName))
                .Select(file => Path.GetFileNameWithoutExtension(file).Split("_").LastOrDefault())
                .Where(number => int.TryParse(number, out _))
                .DefaultIfEmpty("0")
                .Max(number => int.Parse(number));

            // Get the latest model path
            string latestModelPath = $"{modelPath}{modelName}_{maxNumber}";

            return latestModelPath;
        }
    }

    public interface IDiscreteQAgentFactory<T>
    {
        public ComposableQDiscreteAgent<T> ComposeAgent(DQNAgentOptions options);
    }
}
