﻿using RLMatrix.Agents.Common;
using RLMatrix.Memories;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;

namespace RLMatrix.Agents.PPO.Implementations
{
    public class ContinuousPPOAgent<T> : IContinuousPPOAgent<T>
    {
        public required PPOActorNet actorNet { get; set; }
        public required PPOCriticNet criticNet { get; set; }
        public required IOptimize<T> Optimizer { get; init; }
        public required IMemory<T> Memory { get; set; }
        public required int[] DiscreteDimensions { get; init; }
        public required (float min, float max)[] ContinuousActionBounds { get; init; }
        public required PPOAgentOptions Options { get; init; }
        public required Device Device { get; init; }

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
                    actions[i].discreteActions = new int[DiscreteDimensions.Length];
                    actions[i].continuousActions = new float[ContinuousActionBounds.Length];

                    int discreteIndex = 0;
                    int continuousIndex = 0;

                    for (int j = 0; j < DiscreteDimensions.Length + ContinuousActionBounds.Length; j++)
                    {
                        if (j < DiscreteDimensions.Length)
                        {
                            var actionProbs = result[i, j];
                            if (isTraining)
                            {
                                var actionSample = torch.multinomial(actionProbs, 1, true);
                                actions[i].discreteActions[discreteIndex] = (int)actionSample.item<long>();
                            }
                            else
                            {
                                var actionIndex = actionProbs.argmax();
                                actions[i].discreteActions[discreteIndex] = (int)actionIndex.item<long>();
                            }
                            discreteIndex++;
                        }
                        else
                        {
                            var (min, max) = ContinuousActionBounds[continuousIndex];
                            var mu = result[i, j, 0];
                            var sigma = result[i, j, 1].exp();

                            float action;
                            if (isTraining)
                            {
                                var distribution = torch.distributions.Normal(mu, sigma);
                                action = distribution.sample().item<float>();
                            }
                            else
                            {
                                action = mu.item<float>();
                            }

                            // Clamp the action to the specified bounds
                            action = Math.Clamp(action, min, max);
                            actions[i].continuousActions[continuousIndex] = action;
                            continuousIndex++;
                        }
                    }
                }

                return actions;
            }
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
                        // Discrete Actions
                        result[i].discreteActions = PPOActionSelection<T>.SelectDiscreteActionsFromProbs(forwardResult, DiscreteDimensions);
                        // Continuous Actions
                        result[i].continuousActions = PPOActionSelection<T>.SampleContinuousActions(forwardResult, DiscreteDimensions, ContinuousActionBounds);
                    }
                    else
                    {
                        // Discrete Actions
                        result[i].discreteActions = PPOActionSelection<T>.SelectGreedyDiscreteActions(forwardResult, DiscreteDimensions);
                        // Continuous Actions
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
            // Check if path ends with "/", if not, append it
            var modelPath = path.EndsWith("/") ? path : path + "/";

            // Save the policy network
            string actorNetPath = GetNextAvailableModelPath(modelPath, "modelActor");
            actorNet.save(actorNetPath);

            // Save the target network
            string criticNetPath = GetNextAvailableModelPath(modelPath, "modelCritic");
            criticNet.save(criticNetPath);
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
            var modelPath = path.EndsWith("/") ? path : path + "/";

            // Load the policy network
            string actorNetPath = GetLatestModelPath(modelPath, "modelActor");
            actorNet.load(actorNetPath, strict: true);

            // Load the target network
            string criticNetPath = GetLatestModelPath(modelPath, "modelCritic");
            criticNet.load(criticNetPath, strict: true);

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
}