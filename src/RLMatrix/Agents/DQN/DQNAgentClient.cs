using OneOf;
using RLMatrix.Memories;
using System.Collections.Concurrent;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


namespace RLMatrix
{

    /// <summary>
    /// Represents a Deep Q-Learning agent.
    /// </summary>
    /// <typeparam name="T">The type of the state representation, either float[] or float[,].</typeparam>
    public class DQNAsyncAgent<T> : IDiscreteAgent<T>
    {
        torch.Device myDevice;
        DQNAgentOptions myOptions;
        List<IEnvironment<T>> myEnvironments;
        DQNNET myPolicyNet;
        DQNNET myTargetNet;
        OptimizerHelper myOptimizer;
        IMemory<T> myReplayBuffer;
        int episodeCounter = 0;
        int softUpdateCounter = 0;

        //TODO: Can this be managed? Can we have some object encapsulating all progress to peek inside current agent?
        public List<double> episodeRewards = new();

        /// <summary>
        /// Initializes a new instance of the DQNAgent class.
        /// </summary>
        /// <param name="opts">The options for the agent.</param>
        /// <param name="env">The environment in which the agent operates.</param>
        /// <param name="netProvider">The network provider for the agent.</param>
        public DQNAsyncAgent(DQNAgentOptions opts, EnvSizeDTO<T> sizeDTO, IDQNNetProvider<T> netProvider = null)
        {
            //check if T is either float[] or float[,]
            if (typeof(T) != typeof(float[]) && typeof(T) != typeof(float[,]))
            {
                throw new System.ArgumentException("T must be either float[] or float[,]");
            }
            netProvider ??= new DQNNetProvider<T>(opts.Width, opts.Depth);
            myDevice = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine($"Running DQN on {myDevice.type.ToString()}");
            myOptions = opts;

            myPolicyNet = netProvider.CreateCriticNet(sizeDTO).to(myDevice);
            myTargetNet = netProvider.CreateCriticNet(sizeDTO).to(myDevice);
            myOptimizer = optim.Adam(myPolicyNet.parameters(), opts.LR);

            myReplayBuffer = new ReplayMemory<T>(myOptions.MemorySize, myOptions.BatchSize);
            if (myOptions.DisplayPlot != null)
            {
                myOptions.DisplayPlot.CreateOrUpdateChart(new List<double>());
            }

        }

        public void SaveAgent(string path, bool saveExperienceBuffer = true)
        {
            System.IO.Directory.CreateDirectory(path);

            myPolicyNet.save(path + "/policy.pt");
            myTargetNet.save(path + "/target.pt");

            if (!saveExperienceBuffer)
                return;

            throw new Exception("Not implemented");

          //  myReplayBuffer.Save(path + "/experienceBuffer.bin");
        }

        public void LoadAgent(string path)
        {
            myPolicyNet.load(path + "/policy.pt");
            myTargetNet.load(path + "/target.pt");

            myOptimizer = optim.Adam(myPolicyNet.parameters(), myOptions.LR);
            
            //check if contains experience buffer in the folder
            if (File.Exists(path + "/experienceBuffer.bin"))
            {
                throw new Exception("Not implemented");
               // myReplayBuffer.Load(path + "/experienceBuffer.bin");
            }
        }


        /// <summary>
        /// Selects an action based on the current state and the policy.
        /// </summary>
        /// <param name="state">The current state.</param>
        /// <param name="isTraining">A flag indicating whether the agent is in training mode.</param>
        /// <returns>The selected action.</returns>
        Random Random = new Random();
        public virtual int[] SelectAction(T state, bool isTraining = true)
        {
            double sample = Random.NextDouble(); // Use the already created Random object
            double epsThreshold = myOptions.EPS_END + (myOptions.EPS_START - myOptions.EPS_END) *
                                  Math.Exp(-1.0 * episodeCounter / myOptions.EPS_DECAY);

            if (sample > epsThreshold || !isTraining)
            {
                return ActionsFromState(state);
            }
            else
            {
                // Exploration: Select random actions for each action dimension.
                int[] randomActions = new int[myEnvironments[0].actionSize.Length];
                for (int i = 0; i < myEnvironments[0].actionSize.Length; i++)
                {
                    randomActions[i] = Random.Next(0, myEnvironments[0].actionSize[i]);
                }
                return randomActions;
            }
        }

        public virtual int[] ActionsFromState(T state)
        {
            using (torch.no_grad())
            {
                Tensor stateTensor = StateToTensor(state); // Shape: [state_dim]
                Tensor qValuesAllHeads = myPolicyNet.forward(stateTensor).view(1, myEnvironments[0].actionSize.Length, myEnvironments[0].actionSize[0]); // Shape: [1, num_heads, num_actions]
                Tensor bestActions = qValuesAllHeads.argmax(dim: -1).squeeze().to(ScalarType.Int32); // Shape: [num_heads]
                return bestActions.data<int>().ToArray();
            }
        }



        private (T[] batchStates, int[][] batchMultiActions, float[] batchRewards, T?[] batchNextStates, bool[] nonFinalMaskArray)
    ExtractBatchData(ReadOnlySpan<TransitionInMemory<T>> transitions)
        {
            T[] batchStates = new T[transitions.Length];
            int[][] batchMultiActions = new int[transitions.Length][];
            float[] batchRewards = new float[transitions.Length];
            T?[] batchNextStates = new T?[transitions.Length];
            bool[] nonFinalMaskArray = new bool[transitions.Length];

            for (int i = 0; i < transitions.Length; i++)
            {
                batchStates[i] = transitions[i].state;
                batchMultiActions[i] = transitions[i].discreteActions;
                batchRewards[i] = transitions[i].reward;
                batchNextStates[i] = transitions[i].nextState;
                nonFinalMaskArray[i] = batchNextStates[i] != null;
            }

            return (batchStates, batchMultiActions, batchRewards, batchNextStates, nonFinalMaskArray);
        }

        /// <summary>
        /// Optimizes the model instance based on the replay buffer.
        /// </summary>
        public virtual void OptimizeModel()
        {
            if (myReplayBuffer.Length < myOptions.BatchSize)
                return;

            ReadOnlySpan<TransitionInMemory<T>> transitions = myReplayBuffer.Sample();

            (T[] batchStates, int[][] batchMultiActions, float[] batchRewards, T?[] batchNextStates, bool[] nonFinalMaskArray) = ExtractBatchData(transitions);

            Tensor nonFinalMask = tensor(batchNextStates.Select(s => s != null).ToArray()).to(myDevice);
            Tensor stateBatch = stack(batchStates.Select(s => StateToTensor(s)).ToArray()).to(myDevice);

            Tensor[] nonFinalNextStatesArray = batchNextStates.Where(s => s != null).Select(s => StateToTensor(s)).ToArray();
            Tensor nonFinalNextStates;
            if (nonFinalNextStatesArray.Length > 0)
            {
                nonFinalNextStates = stack(nonFinalNextStatesArray).to(myDevice);
            }
            else
            {
                // If all next states are terminal, we still need to create a dummy tensor for nonFinalNextStates
                // to prevent errors but it won't be used because nonFinalMask will filter them out.
                nonFinalNextStates = zeros(new long[] { 1, stateBatch.shape[1] }).to(myDevice); // Adjust the shape as necessary
            }

            Tensor actionBatch = stack(batchMultiActions.Select(a => tensor(a).to(torch.int64)).ToArray()).to(myDevice);
            Tensor rewardBatch = stack(batchRewards.Select(r => tensor(r)).ToArray()).to(myDevice);
            Tensor qValuesAllHeads = myPolicyNet.forward(stateBatch);

            Tensor expandedActionBatch = actionBatch.unsqueeze(2); // Expand to [batchSize, numHeads, 1]

            Tensor stateActionValues = qValuesAllHeads.gather(2, expandedActionBatch).squeeze(2).to(myDevice); // [batchSize, numHeads]

            Tensor nextStateValues;
            using (no_grad())
            {
                if (nonFinalNextStatesArray.Length > 0)
                {
                    nextStateValues = myTargetNet.forward(nonFinalNextStates).max(2).values; // [batchSize, numHeads]
                }
                else
                {
                    // If all states are terminal, this tensor won't be used due to masking.
                    nextStateValues = zeros(new long[] { myOptions.BatchSize, myEnvironments[0].actionSize.Count() }).to(myDevice);
                }
            }

            Tensor maskedNextStateValues = zeros(new long[] { myOptions.BatchSize, myEnvironments[0].actionSize.Count() }).to(myDevice);
            maskedNextStateValues.masked_scatter_(nonFinalMask.unsqueeze(1), nextStateValues);

            Tensor expectedStateActionValues = (maskedNextStateValues * myOptions.GAMMA) + rewardBatch.unsqueeze(1);

            // Compute Huber loss
            SmoothL1Loss criterion = torch.nn.SmoothL1Loss();
            Tensor loss = criterion.forward(stateActionValues, expectedStateActionValues);

            // Optimize the model
            myOptimizer.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_value_(myPolicyNet.parameters(), 100);
            myOptimizer.step();
        }



        //this makes sure arrays are passed by value not by reference to prevent old memories being overwritten
        public T DeepCopy(T input)
        {
            if (!typeof(T).IsArray)
            {
                throw new InvalidOperationException("This method can only be used with arrays!");
            }

            // Handle nulls
            if (ReferenceEquals(input, null))
            {
                return default(T);
            }

            var rank = ((Array)(object)input).Rank;
            var lengths = new int[rank];
            for (int i = 0; i < rank; ++i)
                lengths[i] = ((Array)(object)input).GetLength(i);

            var clone = Array.CreateInstance(typeof(T).GetElementType(), lengths);

            Array.Copy((Array)(object)input, clone, ((Array)(object)input).Length);

            return (T)(object)clone;
        }
        //TODO: Removed method for inference, need a new one

        /// <summary>
        /// Updates the target network weights based on the policy network weights.
        /// </summary>
        public void SoftUpdateTargetNetwork()
{
    var targetNetStateDict = myTargetNet.state_dict();
    var policyNetStateDict = myPolicyNet.state_dict();

    foreach (var key in targetNetStateDict.Keys)
    {
        Tensor targetNetParam = targetNetStateDict[key];
        Tensor policyNetParam = policyNetStateDict[key];

        // Detach the target network parameter from the computational graph
        targetNetParam = targetNetParam.detach();

        // Update the target network parameter using a weighted average
        targetNetParam.mul_(1 - myOptions.TAU).add_(policyNetParam * myOptions.TAU);
    }

    // Load the updated state dictionary into the target network
    myTargetNet.load_state_dict(targetNetStateDict);
}

        /// <summary>
        /// Converts the state to a tensor representation for torchsharp. Only float[] and float[,] states are supported.
        /// </summary>
        /// <param name="state">The state to convert.</param>
        /// <returns>The state as a tensor.</returns>
        protected Tensor StateToTensor(T state)
        {
            switch (state)
            {
                case float[] stateArray:
                    return tensor(stateArray).to(myDevice);
                case float[,] stateMatrix:
                    return tensor(stateMatrix).to(myDevice);
                default:
                    throw new InvalidCastException("State must be either float[] or float[,]");
            }
        }

        public void Step(bool isTraining)
        {
            throw new NotImplementedException();
        }
    }
}