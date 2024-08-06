using RLMatrix.Agents.Common;
using RLMatrix.Memories;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace RLMatrix.Agents.PPO.Implementations
{
    public class ContinuousPPOAgent<T> : IContinuousPPOAgent<T>
    {
#if NET8_0_OR_GREATER
        public required PPOActorNet actorNet { get; set; }
        public required PPOCriticNet criticNet { get; set; }
        public required IOptimize<T> Optimizer { get; init; }
        public required IMemory<T> Memory { get; set; }
        public required int[] DiscreteDimensions { get; init; }
        public required (float min, float max)[] ContinuousActionBounds { get; init; }
        public required PPOAgentOptions Options { get; init; }
        public required Device Device { get; init; }
#else
        public PPOActorNet actorNet { get; set; }
        public PPOCriticNet criticNet { get; set; }
        public IOptimize<T> Optimizer { get; set; }
        public IMemory<T> Memory { get; set; }
        public int[] DiscreteDimensions { get; set; }
        public (float min, float max)[] ContinuousActionBounds { get; set; }
        public PPOAgentOptions Options { get; set; }
        public Device Device { get; set; }
#endif

        public void AddTransition(IEnumerable<TransitionPortable<T>> transitions)
        {
            Memory.Push(transitions.ToTransitionInMemory());
        }

        public void OptimizeModel()
        {
            Optimizer.Optimize(Memory);
        }

        public (int[] discreteActions, float[] continuousActions)[] SelectActions(T[] states, bool isTraining)
        {
            using (var scope = torch.no_grad())
            {
                Tensor stateTensor = Utilities<T>.StateBatchToTensor(states, Device);
                var result = actorNet.forward(stateTensor);
                var actions = new (int[] discreteActions, float[] continuousActions)[states.Length];

                for (int i = 0; i < states.Length; i++)
                {
                    if (isTraining)
                    {
                        actions[i].discreteActions = PPOActionSelection<T>.SelectDiscreteActionsFromProbs(result[i].unsqueeze(0), DiscreteDimensions);
                        actions[i].continuousActions = PPOActionSelection<T>.SampleContinuousActions(result[i].unsqueeze(0), DiscreteDimensions, ContinuousActionBounds);
                    }
                    else
                    {
                        actions[i].discreteActions = PPOActionSelection<T>.SelectGreedyDiscreteActions(result[i].unsqueeze(0), DiscreteDimensions);
                        actions[i].continuousActions = PPOActionSelection<T>.SelectMeanContinuousActions(result[i].unsqueeze(0), DiscreteDimensions, ContinuousActionBounds);
                    }
                }

                return actions;
            }
        }

        private float Clamp(float value, float min, float max)
        {
            return (value < min) ? min : (value > max) ? max : value;
        }

        public (int[] discreteActions, float[] continuousActions)[] SelectActions2(T[] states, bool isTraining)
        {
            var result = new (int[] discreteActions, float[] continuousActions)[states.Length];

            for (int i = 0; i < states.Length; i++)
            {
                using (var scope = torch.no_grad())
                {
                    Tensor stateTensor = Utilities<T>.StateToTensor(states[i], Device);
                    var forwardResult = actorNet.forward(stateTensor);

                    if (isTraining)
                    {
                        result[i].discreteActions = PPOActionSelection<T>.SelectDiscreteActionsFromProbs(forwardResult, DiscreteDimensions);
                        result[i].continuousActions = PPOActionSelection<T>.SampleContinuousActions(forwardResult, DiscreteDimensions, ContinuousActionBounds);
                    }
                    else
                    {
                        result[i].discreteActions = PPOActionSelection<T>.SelectGreedyDiscreteActions(forwardResult, DiscreteDimensions);
                        result[i].continuousActions = PPOActionSelection<T>.SelectMeanContinuousActions(forwardResult, DiscreteDimensions, ContinuousActionBounds);
                    }
                }
            }

            return result;
        }

        public virtual ((int[] discreteActions, float[] continuousActions) actions, Tensor? memoryState, Tensor? memoryState2)[] SelectActionsRecurrent((T state, Tensor? memoryState, Tensor? memoryState2)[] states, bool isTraining)
        {
            throw new Exception("Using recurrent action selection with non recurrent agent, use (int[] discreteActions, float[][] continuousActions) SelectActions(T[] states, bool isTraining) signature instead");
        }

        public void Save(string path)
        {
            var modelPath = path.EndsWith(Path.DirectorySeparatorChar.ToString()) ? path : path + Path.DirectorySeparatorChar;

            string actorNetPath = GetNextAvailableModelPath(modelPath, "modelActor");
            actorNet.save(actorNetPath);

            string criticNetPath = GetNextAvailableModelPath(modelPath, "modelCritic");
            criticNet.save(criticNetPath);
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

            string actorNetPath = GetLatestModelPath(modelPath, "modelActor");
            actorNet.load(actorNetPath, strict: true);

            string criticNetPath = GetLatestModelPath(modelPath, "modelCritic");
            criticNet.load(criticNetPath, strict: true);

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
}