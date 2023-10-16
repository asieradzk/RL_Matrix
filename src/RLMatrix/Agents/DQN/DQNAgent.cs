using OneOf;
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
    public class DQNAgent<T>
    {
        protected torch.Device myDevice;
        protected DQNAgentOptions myOptions;
        protected IEnvironment<T> myEnvironment;
        protected DQNNET myPolicyNet;
        protected DQNNET myTargetNet;
        protected OptimizerHelper myOptimizer;
        protected ReplayMemory<T> myReplayBuffer;
        protected int episodeCounter = 0;

        //TODO: Can this be managed? Can we have some object encapsulating all progress to peek inside current agent?
        public List<double> episodeRewards = new();

        /// <summary>
        /// Initializes a new instance of the DQNAgent class.
        /// </summary>
        /// <param name="opts">The options for the agent.</param>
        /// <param name="env">The environment in which the agent operates.</param>
        /// <param name="netProvider">The network provider for the agent.</param>
        public DQNAgent(DQNAgentOptions opts, IEnvironment<T> env, IDQNNetProvider<T> netProvider = null)
        {
            

            //check if T is either float[] or float[,]
            if (typeof(T) != typeof(float[]) && typeof(T) != typeof(float[,]))
            {
                throw new System.ArgumentException("T must be either float[] or float[,]");
            }
            netProvider ??= new DQNNetProvider<T>(1024, 2);
            myDevice = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine($"Running DQN on {myDevice.type.ToString()}");
            myOptions = opts;
            myEnvironment = env;

            myPolicyNet = netProvider.CreateCriticNet(env).to(myDevice);
            myTargetNet = netProvider.CreateCriticNet(env).to(myDevice);
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

            myReplayBuffer.Save(path + "/experienceBuffer.bin");
        }

        public void LoadAgent(string path)
        {
            myPolicyNet.load(path + "/policy.pt");
            myTargetNet.load(path + "/target.pt");

            myOptimizer = optim.Adam(myPolicyNet.parameters(), myOptions.LR);
            
            //check if contains experience buffer in the folder
            if (File.Exists(path + "/experienceBuffer.bin"))
            {
                myReplayBuffer.Load(path + "/experienceBuffer.bin");
            }
        }


        /// <summary>
        /// Selects an action based on the current state and the policy.
        /// </summary>
        /// <param name="state">The current state.</param>
        /// <param name="isTraining">A flag indicating whether the agent is in training mode.</param>
        /// <returns>The selected action.</returns>
        public int[] SelectAction(T state, bool isTraining = true)
        {
            double sample = new Random().NextDouble();
            double epsThreshold = myOptions.EPS_END + (myOptions.EPS_START - myOptions.EPS_END) *
                                  Math.Exp(-1.0 * episodeCounter / myOptions.EPS_DECAY);

            using (torch.no_grad())
            {
                Tensor stateTensor = StateToTensor(state);
                int[] selectedActions = new int[myEnvironment.actionSize.Length];

                // Get action predictions from policy network only once
                Tensor predictedActions = myPolicyNet.forward(stateTensor);

                for (int i = 0; i < myEnvironment.actionSize.Length; i++)
                {
                    if (sample > epsThreshold || !isTraining)
                    {
                        // Select the action with the highest expected reward for each action dimension.
                        selectedActions[i] = (int)predictedActions.select(1, i).argmax().item<long>();
                    }
                    else
                    {
                        // For exploration, select a random action within the range of the action dimension.
                        selectedActions[i] = new Random().Next(0, myEnvironment.actionSize[i]);
                    }
                }

                return selectedActions;
            }
        }




        /// <summary>
        /// Optimizes the model instance based on the replay buffer.
        /// </summary>
        public virtual void OptimizeModel()
        {
            if (myReplayBuffer.Length < myOptions.BatchSize)
                return;

            List<Transition<T>> transitions = myReplayBuffer.Sample();

            List<T> batchStates = transitions.Select(t => t.state).ToList();
            List<int[]> batchMultiActions = transitions.Select(t => t.discreteActions).ToList();
            List<float> batchRewards = transitions.Select(t => t.reward).ToList();
            List<T> batchNextStates = transitions.Select(t => t.nextState).ToList();

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
                return; // All states are terminal
            }

            Tensor actionBatch = stack(batchMultiActions.Select(a => tensor(a).to(torch.int64)).ToArray()).to(myDevice);
            Tensor rewardBatch = stack(batchRewards.Select(r => tensor(r)).ToArray()).to(myDevice);

            Tensor qValuesAllHeads = myPolicyNet.forward(stateBatch);
            Tensor expandedActionBatch = actionBatch.unsqueeze(2); // Expand to [batchSize, numHeads, 1]

            Tensor stateActionValues = qValuesAllHeads.gather(2, expandedActionBatch).squeeze(2).to(myDevice); // [batchSize, numHeads]

            Tensor nextStateValues;
            using (no_grad())
            {
                nextStateValues = myTargetNet.forward(nonFinalNextStates).max(2).values;  // [batchSize, numHeads]
            }
            Tensor maskedNextStateValues = zeros(new long[] { myOptions.BatchSize, myEnvironment.actionSize.Count() }).to(myDevice);


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

        /// <summary>
        /// Trains the agent for one episode. This will reset environment and run it untill done flag is set to true. After that it will optimize the model.
        /// </summary>
        public void TrainEpisode()
        {
            episodeCounter++;
            // Initialize the environment and get its state
            myEnvironment.Reset();
            T state = DeepCopy(myEnvironment.GetCurrentState());
            float cumulativeReward = 0;
            for (int t = 0; ; t++)
            {



                // Select an action based on the policy
                var action = SelectAction(state);
                // Take a step using the selected action
                var reward = myEnvironment.Step(action);
                // Check if the episode is done
                var done = myEnvironment.isDone;


                T nextState;
                if (done)
                {
                    // If done, there is no next state
                    nextState = default;
                }
                else
                {
                    // If not done, get the next state
                    nextState = DeepCopy(myEnvironment.GetCurrentState());
                }

                if (state == null)
                    Console.WriteLine("state is null");

                // Store the transition in temporary memory
                myReplayBuffer.Push(new Transition<T>(state, action, null, reward, nextState));



                cumulativeReward += reward;
                // If not done, move to the next state
                if (!done)
                {
                    state = nextState;
                }
                // Perform one step of the optimization (on the policy network)
                OptimizeModel();

                // Soft update of the target network's weights
                // θ′ ← τ θ + (1 −τ )θ′
                SoftUpdateTargetNetwork();


                if (done)
                {
                    episodeRewards.Add(cumulativeReward);
                    if(myOptions.DisplayPlot != null)
                    {
                        myOptions.DisplayPlot.CreateOrUpdateChart(episodeRewards);
                    }
                    break;
                }
            }
        }

        /// <summary>
        /// Predicts the cumulative reward for one episode using the currently loaded model.
        /// </summary>
        /// <returns>The predicted cumulative reward.</returns>
        public float PredictEpisode()
        {
            // Initialize the environment and get its state
            myEnvironment.Reset();
            var state = myEnvironment.GetCurrentState();
            float cumulativeReward = 0;
            for (int t = 0; ; t++)
            {
                // Select an action based on the policy
                var action = SelectAction(state, isTraining: false);
                // Take a step using the selected action
                var reward = myEnvironment.Step(action);
                // Check if the episode is done
                var done = myEnvironment.isDone;

                T nextState;
                if (done)
                {
                    // If done, there is no next state
                    nextState = default;
                }
                else
                {
                    // If not done, get the next state
                    nextState = myEnvironment.GetCurrentState();
                }

                if (state == null)
                    Console.WriteLine("state is null");

                cumulativeReward += reward;
                // If not done, move to the next state
                if (!done)
                {
                    state = nextState;
                }

                if (done)
                {

                    // Optionally, you can still update your chart with the cumulative reward
                    return cumulativeReward;
                }
            }
        }

        /// <summary>
        /// Updates the target network weights based on the policy network weights.
        /// </summary>
        public void SoftUpdateTargetNetwork()
        {
            var targetNetStateDict = myTargetNet.state_dict();
            var policyNetStateDict = myPolicyNet.state_dict();
            var updatedStateDict = new Dictionary<string, Tensor>();

            foreach (var key in targetNetStateDict.Keys)
            {
                Tensor targetNetParam = targetNetStateDict[key];
                Tensor policyNetParam = policyNetStateDict[key];
                updatedStateDict[key] = (policyNetParam * myOptions.TAU + targetNetParam * (1 - myOptions.TAU)).cpu();
            }

            myTargetNet = myTargetNet.cpu();
            myTargetNet.load_state_dict(updatedStateDict);
            myTargetNet = myTargetNet.to(myDevice);
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


    }
}