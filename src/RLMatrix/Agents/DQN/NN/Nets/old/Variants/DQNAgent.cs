using OneOf;
using RLMatrix.Memories;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;


namespace RLMatrix
{

    /// <summary>
    /// Represents a Deep Q-Learning agent.
    /// </summary>
    /// <typeparam name="T">The type of the state representation, either float[] or float[,].</typeparam>
    public class DQNAgent<T> : IDiscreteAgent<T>
    {
        protected torch.Device myDevice;
        protected DQNAgentOptions myOptions;
        protected List<IEnvironment<T>> myEnvironments;
        protected DQNNET myPolicyNet;
        protected DQNNET myTargetNet;
        protected OptimizerHelper myOptimizer;
        protected IMemory<T> myReplayBuffer;
        protected int episodeCounter = 0;
        protected int softUpdateCounter = 0;
        protected LRScheduler myLRScheduler;

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
            netProvider ??= new DQNNetProvider<T>(opts.Width, opts.Depth);
            myDevice = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine($"Running DQN on {myDevice.type.ToString()}");
            myOptions = opts;
            myEnvironments = envs;

            myPolicyNet = netProvider.CreateCriticNet(envs[0]).to(myDevice);
            myTargetNet = netProvider.CreateCriticNet(envs[0]).to(myDevice);
         
            myOptimizer = optim.Adam(myPolicyNet.parameters(), opts.LR);
            myLRScheduler = new optim.lr_scheduler.impl.CyclicLR(myOptimizer, myOptions.LR * 0.5f, myOptions.LR * 2f, step_size_up: 500, step_size_down: 2000, cycle_momentum: false);

            myReplayBuffer = new ReplayMemory<T>(myOptions.MemorySize, myOptions.BatchSize);
            if (myOptions.DisplayPlot != null)
            {
                myOptions.DisplayPlot.CreateOrUpdateChart(new List<double>());
            }
            torch.backends.cudnn.allow_tf32 = true;
            

        }

        public void SaveAgent(string path, bool saveExperienceBuffer = true)
        {
            System.IO.Directory.CreateDirectory(path);

            myPolicyNet.save(path + "/policy.pt");
            myTargetNet.save(path + "/target.pt");

            if (!saveExperienceBuffer)
                return;

            throw new NotImplementedException();
            //myReplayBuffer.Save(path + "/experienceBuffer.bin");
        }

        public void LoadAgent(string path)
        {
            myPolicyNet.load(path + "/policy.pt");
            myTargetNet.load(path + "/target.pt");

            myOptimizer = optim.Adam(myPolicyNet.parameters(), myOptions.LR);
            
            //check if contains experience buffer in the folder
            if (File.Exists(path + "/experienceBuffer.bin"))
            {
                throw new NotImplementedException();
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
                Tensor stateTensor = StateToTensor(state, myDevice); // Shape: [state_dim]
                Tensor qValuesAllHeads = myPolicyNet.forward(stateTensor).view(1, myEnvironments[0].actionSize.Length, myEnvironments[0].actionSize[0]); // Shape: [1, num_heads, num_actions]
                Tensor bestActions = qValuesAllHeads.argmax(dim: -1).squeeze().to(ScalarType.Int32); // Shape: [num_heads]
                return bestActions.data<int>().ToArray();
            }
        }
        private void CreateTensorsFromTransitions(ref ReadOnlySpan<TransitionInMemory<T>> transitions, out Tensor nonFinalMask, out Tensor stateBatch, out Tensor nonFinalNextStates, out Tensor actionBatch, out Tensor rewardBatch)
        {
            int length = transitions.Length;
            var fixedActionSize = myEnvironments[0].actionSize.Length; // Assuming a fixed action size for all environments

            // Pre-allocate arrays based on the known batch size
            bool[] nonFinalMaskArray = new bool[length];
            float[] batchRewards = new float[length];
            int[] flatMultiActions = new int[length * fixedActionSize];
            T[] batchStates = new T[length];
            T?[] batchNextStates = new T?[length];

            int flatMultiActionsIndex = 0;
            int nonFinalNextStatesCount = 0;

            
            for (int i = 0; i < length; i++)
            {
                var transition = transitions[i];
                nonFinalMaskArray[i] = transition.nextState != null;
                batchRewards[i] = transition.reward;
                Array.Copy(transition.discreteActions, 0, flatMultiActions, flatMultiActionsIndex, transition.discreteActions.Length);
                flatMultiActionsIndex += transition.discreteActions.Length; // Assuming a fixed length for all actions

                batchStates[i] = transition.state;
                batchNextStates[i] = transition.nextState;

                if (transition.nextState != null)
                {
                    nonFinalNextStatesCount++;
                }
            }

        
            stateBatch = StateToTensor(batchStates, myDevice);
            nonFinalMask = torch.tensor(nonFinalMaskArray, device: myDevice);
            rewardBatch = torch.tensor(batchRewards, device: myDevice);
            actionBatch = torch.tensor(flatMultiActions, new long[] { length, fixedActionSize }, torch.int64).to(myDevice); // Reshape based on actual action size


            if (nonFinalNextStatesCount > 0)
            {
                T[] nonFinalNextStatesArray = new T[nonFinalNextStatesCount];
                int index = 0;
                for (int i = 0; i < length; i++)
                {
                    if (batchNextStates[i] is not null)
                    {
                        nonFinalNextStatesArray[index++] = batchNextStates[i];
                    }
                }
                nonFinalNextStates = StateToTensor(nonFinalNextStatesArray, myDevice);
            }
            else
            {
                nonFinalNextStates = torch.zeros(new long[] { 1, stateBatch.shape[1] }, device: myDevice);
            }
        }
        /// <summary>
        /// Optimizes the model instance based on the replay buffer.
        /// </summary>
        public virtual void OptimizeModel()
        {
            int nSteps = 1;
            bool useD2QN = false;


            if (myReplayBuffer.Length < myOptions.BatchSize)
                return;

            ReadOnlySpan<TransitionInMemory<T>> transitions = myReplayBuffer.Sample();
            CreateTensorsFromTransitions(ref transitions, out Tensor nonFinalMask, out Tensor stateBatch, out Tensor nonFinalNextStates, out Tensor actionBatch, out Tensor rewardBatch);
            Tensor qValuesAllHeads = ComputeQValues(stateBatch);
            Tensor stateActionValues = ExtractStateActionValues(qValuesAllHeads, actionBatch);
            Tensor nextStateValues = ComputeNextStateValues(nonFinalNextStates, useD2QN);
            Tensor expectedStateActionValues = ComputeExpectedStateActionValues(nextStateValues, rewardBatch, nonFinalMask, nSteps, ref transitions);
            Tensor loss = ComputeLoss(stateActionValues, expectedStateActionValues);
            UpdateModel(loss);
            myLRScheduler.step();
        }

        protected virtual Tensor ComputeQValues(Tensor stateBatch)
        {
            var res = myPolicyNet.forward(stateBatch);
            return res;
        }

        protected virtual Tensor ExtractStateActionValues(Tensor qValuesAllHeads, Tensor actionBatch)
        {
            Tensor expandedActionBatch = actionBatch.unsqueeze(2);
            var res = qValuesAllHeads.gather(2, expandedActionBatch).squeeze(2).to(myDevice);
            return res;
        }

        protected virtual Tensor ComputeNextStateValues(Tensor nonFinalNextStates, bool useD2QN)
        {
            Tensor nextStateValues;
            using (no_grad())
            {
                if (nonFinalNextStates.shape[0] > 0)
                {
                    if (useD2QN)
                    {
                        // Use myPolicyNet to select the best action for each next state based on the current policy
                        Tensor nextActions = myPolicyNet.forward(nonFinalNextStates).max(2).indexes;
                        // Evaluate the selected actions' Q-values using myTargetNet
                        nextStateValues = myTargetNet.forward(nonFinalNextStates).gather(2, nextActions.unsqueeze(-1)).squeeze(-1);
                    }
                    else
                    {
                        nextStateValues = myTargetNet.forward(nonFinalNextStates).max(2).values; // [batchSize, numHeads]
                    }
                }
                else
                {
                    nextStateValues = zeros(new long[] { myOptions.BatchSize, myEnvironments[0].actionSize.Count() }).to(myDevice);
                }
            }
            return nextStateValues;
        }

        protected virtual Tensor ComputeExpectedStateActionValues(Tensor nextStateValues, Tensor rewardBatch, Tensor nonFinalMask, int nSteps, ref ReadOnlySpan<TransitionInMemory<T>> transitions)
        {
            Tensor maskedNextStateValues = zeros(new long[] { myOptions.BatchSize, myEnvironments[0].actionSize.Count() }, device: myDevice);
            maskedNextStateValues.masked_scatter_(nonFinalMask.unsqueeze(1), nextStateValues);

            if (nSteps <= 1)
            {
                var res=  (maskedNextStateValues * myOptions.GAMMA) + rewardBatch.unsqueeze(1);
                return res;
            }
            else
            {
                // Compute n-step returns
                Tensor nStepRewards = CalculateNStepReturns(ref transitions, nSteps, myOptions.GAMMA);
                return (maskedNextStateValues * Math.Pow(myOptions.GAMMA, nSteps)) + nStepRewards.unsqueeze(1);
            }
        }

        protected virtual Tensor ComputeLoss(Tensor stateActionValues, Tensor expectedStateActionValues)
        {
            SmoothL1Loss criterion = torch.nn.SmoothL1Loss();
            var res =  criterion.forward(stateActionValues, expectedStateActionValues);
            return res;
        }

        protected virtual void UpdateModel(Tensor loss)
        {
            myOptimizer.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_value_(myPolicyNet.parameters(), 100);
            myOptimizer.step();
        }

        protected Tensor CalculateNStepReturns(ref ReadOnlySpan<TransitionInMemory<T>> transitions, int nSteps, float gamma)
        {
            int batchSize = transitions.Length;
            Tensor returns = torch.zeros(batchSize, device: myDevice);

            for (int i = 0; i < batchSize; i++)
            {
                TransitionInMemory<T> currentTransition = transitions[i];
                float nStepReturn = 0;
                float discount = 1;

                for (int j = 0; j < nSteps; j++)
                {
                    nStepReturn += discount * currentTransition.reward;
                    if (currentTransition.nextTransition is null)
                    {
                        break;
                    }
                    currentTransition = currentTransition.nextTransition;

                    discount *= gamma;
                }

                returns[i] = nStepReturn;
            }
            returns.print();
            return returns;
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
        int stepCounter = 1;
        public void Step(bool isTraining = true)
        {
            if (!initialisetrainingonce)
            {
                InitialiseTraining();
            }

            foreach (var episode in episodes)
            {
                episode.Step(isTraining);
                stepCounter++;
                softUpdateCounter++;
            }

            episodeCounter++;
            
            if(stepCounter > stepHorizon)
            {
                if(isTraining)
                    OptimizeModel();
               
                if (softUpdateCounter > myOptions.SoftUpdateInterval)
                {

                    if(isTraining)
                        SoftUpdateTargetNetwork();

                    softUpdateCounter = 0;
                }
                
                stepCounter = 0;
            }


            //TODO: Update chart (maybe with the first agent?)
        }

        internal class Episode
        {
            T currentState;
            Guid currentGuid;
            float cumulativeReward = 0;

            IEnvironment<T> myEnv;
            DQNAgent<T> myAgent;

            public Episode(IEnvironment<T> myEnv, DQNAgent<T> agent)
            {
                this.myEnv = myEnv;
                myAgent = agent;
                myEnv.Reset();
                currentState = myAgent.DeepCopy(myEnv.GetCurrentState());
                currentGuid = Guid.NewGuid();
            }

            List<TransitionPortable<T>> TempBuffer = new();

            public void Step(bool isTraining)
            {
                if (!myEnv.isDone)
                {

                    var action = myAgent.SelectAction(currentState, isTraining);
                    var reward = myEnv.Step(action);
                    var done = myEnv.isDone;

                    T nextState;
                    Guid nextGuid;
                    if (done)
                    {
                        nextState = default;
                        nextGuid = default;
                    }
                    else
                    {

                        nextState = myAgent.DeepCopy(myEnv.GetCurrentState());
                        nextGuid = Guid.NewGuid();
                    }
                    cumulativeReward += reward;
                    TempBuffer.Add(new TransitionPortable<T>(currentGuid, currentState, action, null, reward: reward, NextTransitionGuid: nextGuid));
                    currentState = nextState;
                    currentGuid = nextGuid;
                    return;
                }

                var rewardCopy = cumulativeReward;
                myAgent.episodeRewards.Add(rewardCopy);
                myAgent.myReplayBuffer.Push(TempBuffer.ToTransitionInMemory<T>());
                TempBuffer.Clear();
                if (myAgent.myOptions.DisplayPlot != null)
                {
                    myAgent.myOptions.DisplayPlot.CreateOrUpdateChart(myAgent.episodeRewards);
                }

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
                    return tensor(stateArray);
                case float[,] stateMatrix:
                    return tensor(stateMatrix);
                default:
                    throw new InvalidCastException("State must be either float[] or float[,]");
            }
        }
        /// <summary>
        /// Converts the state to a tensor representation for torchsharp. Only float[] and float[,] states are supported.
        /// </summary>
        /// <param name="state">The state to convert.</param>
        /// <returns>The state as a tensor.</returns>
        protected Tensor StateToTensor(T state, Device device)
        {
            switch (state)
            {
                case float[] stateArray:
                    return tensor(stateArray, device: device);
                case float[,] stateMatrix:
                    return tensor(stateMatrix, device: device);
                default:
                    throw new InvalidCastException("State must be either float[] or float[,]");
            }
        }

        protected Tensor StateToTensor(T[] states, Device device)
        {
            // Assume the first element determines the type for all
            if (states.Length == 0)
            {
                throw new ArgumentException("States array cannot be empty.");
            }

            if (states[0] is float[])
            {
                // Handling arrays of float arrays (float[][]).
                return HandleFloatArrayStates(states as float[][], device);
            }
            else if (states[0] is float[,])
            {
                // Handling arrays of float matrices (float[][,]).
                return HandleFloatMatrixStates(states as float[][,], device);
            }
            else
            {
                throw new InvalidCastException("States must be arrays of either float[] or float[,].");
            }
        }

        Tensor HandleFloatArrayStates(float[][] states, Device device)
        {
            int totalSize = states.Length * states[0].Length;
            float[] batchData = new float[totalSize];
            int offset = 0;
            foreach (var state in states)
            {
                Buffer.BlockCopy(state, 0, batchData, offset * sizeof(float), state.Length * sizeof(float));
                offset += state.Length;
            }
            var batchShape = new long[] { states.Length, states[0].Length };
            return torch.tensor(batchData, batchShape, device: device);
        }

        Tensor HandleFloatMatrixStates(float[][,] states, Device device)
        {
            int d1 = states[0].GetLength(0);
            int d2 = states[0].GetLength(1);
            float[] batchData = new float[states.Length * d1 * d2];
            int offset = 0;

            foreach (var matrix in states)
            {
                for (int i = 0; i < d1; i++)
                {
                    Buffer.BlockCopy(matrix, i * d2 * sizeof(float), batchData, offset, d2 * sizeof(float));
                    offset += d2 * sizeof(float);
                }
            }

            var batchShape = new long[] { states.Length, d1, d2 };
            return torch.tensor(batchData, batchShape, device: device);
        }




    }
}