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
        protected List<IEnvironment<T>> myEnvironments;
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
        public DQNAgent(DQNAgentOptions opts, List<IEnvironment<T>> envs, IDQNNetProvider<T> netProvider = null)
        {
            if (envs == null || envs.Count == 0 || envs[0] == null)
            {
                throw new System.ArgumentException("Envs must contain at least one environment");
            }

            //check if T is either float[] or float[,]
            if (typeof(T) != typeof(float[]) && typeof(T) != typeof(float[,]))
            {
                throw new System.ArgumentException("T must be either float[] or float[,]");
            }
            netProvider ??= new DQNNetProvider<T>(1024, 2);
            myDevice = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine($"Running DQN on {myDevice.type.ToString()}");
            myOptions = opts;
            myEnvironments = envs;

            myPolicyNet = netProvider.CreateCriticNet(envs[0]).to(myDevice);
            myTargetNet = netProvider.CreateCriticNet(envs[0]).to(myDevice);
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
                int[] selectedActions = new int[myEnvironments[0].actionSize.Length];

                // Get action predictions from policy network only once
                Tensor predictedActions = myPolicyNet.forward(stateTensor);

                for (int i = 0; i < myEnvironments[0].actionSize.Length; i++)
                {
                    if (sample > epsThreshold || !isTraining)
                    {
                        // Select the action with the highest expected reward for each action dimension.
                        selectedActions[i] = (int)predictedActions.select(1, i).argmax().item<long>();
                    }
                    else
                    {
                        // For exploration, select a random action within the range of the action dimension.
                        selectedActions[i] = new Random().Next(0, myEnvironments[0].actionSize[i]);
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

        #region training


        List<Episode> episodes;

        bool initialisetrainingonce = false;
        void InitialiseTraining()
        {
            if (initialisetrainingonce)
                return;

            episodes = new List<Episode>();
            foreach (var env in myEnvironments)
            {
                episodes.Add(new Episode(env, this));
            }

            initialisetrainingonce = true;

        }
        int stepHorizon = 1;
        int stepCounter = 0;
        public void Step()
        {
            if (!initialisetrainingonce)
            {
                InitialiseTraining();
            }

            foreach (var episode in episodes)
            {
                episode.Step();
                stepCounter++;
            }

            episodeCounter++;
            if(stepCounter > stepHorizon)
            {
                OptimizeModel();
                SoftUpdateTargetNetwork();
                stepCounter = 0;
            }


            //TODO: Update chart (maybe with the first agent?)
        }

        internal class Episode
        {
            T currentState;
            float cumulativeReward = 0;

            IEnvironment<T> myEnv;
            DQNAgent<T> myAgent;

            public Episode(IEnvironment<T> myEnv, DQNAgent<T> agent)
            {
                this.myEnv = myEnv;
                myAgent = agent;
                myEnv.Reset();
                currentState = myAgent.DeepCopy(myEnv.GetCurrentState());
            }


            public void Step()
            {
                if(!myEnv.isDone)
                {

                    var action = myAgent.SelectAction(currentState);
                    var reward = myEnv.Step(action);
                    var done = myEnv.isDone;

                    T nextState;
                    if (done)
                    {
                        nextState = default;
                    }
                    else
                    {
                        nextState = myAgent.DeepCopy(myEnv.GetCurrentState());
                    }
                    cumulativeReward += reward;
                    myAgent.myReplayBuffer.Push(new Transition<T>(currentState, action, null, reward, nextState));
                    currentState = nextState;
                    return;
                }
                Console.WriteLine($"Episode finished with reward {cumulativeReward}");
                cumulativeReward = 0;
                myEnv.Reset();
                currentState = myAgent.DeepCopy(myEnv.GetCurrentState());

            }

        }


        #endregion

        //TODO: Removed method for inference, need a new one

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