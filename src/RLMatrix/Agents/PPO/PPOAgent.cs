
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using static TorchSharp.torch.optim;
using RLMatrix;
using OneOf;
using RLMatrix.Memories;
using System.Security;
using System.Net.Http.Headers;
using RLMatrix.Agents.DQN.Domain;

namespace RLMatrix
{
    public class PPOAgent<T> : IDiscreteAgent<T>
    {
        protected torch.Device myDevice;
        protected PPOAgentOptions myOptions;
        protected List<IContinuousEnvironment<T>> myEnvironments;
        protected PPOActorNet myActorNet;
        protected PPOCriticNet myCriticNet;
        protected OptimizerHelper myActorOptimizer;
        protected OptimizerHelper myCriticOptimizer;
        protected EpisodicReplayMemory<T> myReplayBuffer;
        protected int episodeCounter = 0;
        protected GAIL<T> myGAIL;

        //TODO: Can this be managed? Can we have some object encapsulating all progress to peek inside current agent?
        public List<double> episodeRewards = new();


        public PPOAgent(PPOAgentOptions opts, OneOf<List<IContinuousEnvironment<T>>, List<IEnvironment<T>>> env, IPPONetProvider<T> netProvider = null, GAIL<T> GAILInstance = null)
        {
            netProvider = netProvider ?? new PPONetProviderBase<T>(opts.Width, opts.Depth, opts.UseRNN);

            //check if T is either float[] or float[,]
            if (typeof(T) != typeof(float[]) && typeof(T) != typeof(float[,]))
            {
                throw new System.ArgumentException("T must be either float[] or float[,]");
            }
            myDevice = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine($"Running PPO on {myDevice.type.ToString()}");
            myOptions = opts;




            if (env.IsT0)
            {
                myEnvironments = env.AsT0;
            }
            else
            {
                //lets convert every IEnvironment 
                myEnvironments = new List<IContinuousEnvironment<T>>();
                foreach(IEnvironment<T> dicreteEnv in env.AsT1)
                {
                    myEnvironments.Add(ContinuousEnvironmentFactory.Create(dicreteEnv));
                }

            }
            //TODO: this should be checked before assigment to global var
            if (myEnvironments == null || myEnvironments.Count == 0 || myEnvironments[0] == null)
            {
                throw new System.ArgumentException("Envs must contain at least one environment");
            }

            myGAIL = GAILInstance;
            if (myGAIL != null)
            {
                myGAIL.Initialise(myEnvironments[0].stateSize, myEnvironments[0].actionSize, myEnvironments[0].continuousActionBounds, myDevice);
            }


            myActorNet = netProvider.CreateActorNet(myEnvironments[0]).to(myDevice);
            myCriticNet = netProvider.CreateCriticNet(myEnvironments[0]).to(myDevice);

            myActorOptimizer = optim.Adam(myActorNet.parameters(), myOptions.LR, amsgrad: true);
            myCriticOptimizer = optim.Adam(myCriticNet.parameters(), myOptions.LR, amsgrad: true);

            //TODO: I think I forgot to make PPO specific memory.
            myReplayBuffer = new EpisodicReplayMemory<T>(myOptions.MemorySize);

            if (myOptions.DisplayPlot != null)
            {

                myOptions.DisplayPlot.CreateOrUpdateChart(new List<double>());
            }

        }

        //Save actor and critic networks to a folder
        public void SaveAgent(string path)
        {
            System.IO.Directory.CreateDirectory(path);
            myActorNet.save(path + "/actor.pt");
            myCriticNet.save(path + "/critic.pt");

        }
        public void LoadAgent(string path)
        {
            myActorNet.load(path + "/actor.pt");
            myCriticNet.load(path + "/critic.pt");

            myActorOptimizer = optim.Adam(myActorNet.parameters(), myOptions.LR, amsgrad: true);
            myCriticOptimizer = optim.Adam(myCriticNet.parameters(), myOptions.LR, amsgrad: true);
        }

        public (int[], float[]) SelectAction(T state, bool isTraining = true)
        {
            using (torch.no_grad())
            {
                Tensor stateTensor = StateToTensor(state);
         
                var result = myActorNet.forward(stateTensor);

                int[] discreteActions;
                float[] continuousActions;

                if (isTraining)
                {
                    // Discrete Actions
                    discreteActions = SelectDiscreteActionsFromProbs(result);

                    // Continuous Actions
                    continuousActions = SampleContinuousActions(result);
                }
                else
                {
                    // Discrete Actions
                    discreteActions = SelectGreedyDiscreteActions(result);

                    // Continuous Actions
                    continuousActions = SelectMeanContinuousActions(result);
                }

                return (discreteActions, continuousActions);
            }
        }

        public (int[], float[]) SelectAction(List<T> stateHistory, bool isTraining = true)
        {
            using (torch.no_grad())
            {
                Tensor stateBatch = stack(stateHistory.Select(t => StateToTensor(t)).ToArray()).to(myDevice);
               
                var batchResult = myActorNet.forward(stateBatch);
                //get only last action
                var result = batchResult.select(0, batchResult.size(0) - 1).unsqueeze(1);
                int[] discreteActions;
                float[] continuousActions;

                if (isTraining)
                {
                    // Discrete Actions
                    discreteActions = SelectDiscreteActionsFromProbs(result);

                    // Continuous Actions
                    continuousActions = SampleContinuousActions(result);
                }
                else
                {
                    // Discrete Actions
                    discreteActions = SelectGreedyDiscreteActions(result);

                    // Continuous Actions
                    continuousActions = SelectMeanContinuousActions(result);
                }

                return (discreteActions, continuousActions);
            }
        }

        public ((int[], float[]), Tensor) SelectAction(T state, Tensor? memoryState, bool isTraining = true)
        {
            using (torch.no_grad())
            {
                Tensor stateTensor = StateToTensor(state);
                var resultTuple = myActorNet.forward(stateTensor, memoryState);
                var result = resultTuple.Item1;

                int[] discreteActions;
                float[] continuousActions;

                if (isTraining)
                {
                    // Discrete Actions
                    discreteActions = SelectDiscreteActionsFromProbs(result);

                    // Continuous Actions
                    continuousActions = SampleContinuousActions(result);
                }
                else
                {
                    // Discrete Actions
                    discreteActions = SelectGreedyDiscreteActions(result);

                    // Continuous Actions
                    continuousActions = SelectMeanContinuousActions(result);
                }
                return ((discreteActions, continuousActions), resultTuple.Item2);
            }

        }

        #region helpers
        int[] SelectDiscreteActionsFromProbs(Tensor result)
        {
            // Assuming discrete action heads come first
            List<int> actions = new List<int>();
            for (int i = 0; i < myEnvironments[0].actionSize.Count(); i++)
            {
                var actionProbs = result.select(1, i);
                var action = torch.multinomial(actionProbs, 1);
                actions.Add((int)action.item<long>());
            }
            return actions.ToArray();
        }
        static double SampleFromStandardNormal(Random random)
        {
            double u1 = 1.0 - random.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                   Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return randStdNormal;
        }

        float[] SampleContinuousActions(Tensor result)
        {
            List<float> actions = new List<float>();
            int offset = myEnvironments[0].actionSize.Count(); // Assuming discrete action heads come first
            for (int i = 0; i < myEnvironments[0].continuousActionBounds.Count(); i++)
            {
                var mean = result.select(1, offset + i * 2).item<float>();
                var logStd = result.select(1, offset + i * 2 + 1).item<float>();
                var std = (float)Math.Exp(logStd);
                var actionValue = mean + std * (float)SampleFromStandardNormal(new Random());

                // Ensuring that action value stays within given bounds (assuming you have min and max values for each action)
                actionValue = Math.Clamp(actionValue, myEnvironments[0].continuousActionBounds[i].Item1, myEnvironments[0].continuousActionBounds[i].Item2);

                actions.Add(actionValue);
            }
            return actions.ToArray();
        }

       public int[] SelectGreedyDiscreteActions(Tensor result)
        {
            List<int> actions = new List<int>();
            for (int i = 0; i < myEnvironments[0].actionSize.Count(); i++)
            {
                var actionProbs = result.select(1, i);
                var action = actionProbs.argmax();
                actions.Add((int)action.item<long>());
            }
            return actions.ToArray();
        }

        public float[] SelectMeanContinuousActions(Tensor result)
        {
            List<float> actions = new List<float>();
            int offset = myEnvironments[0].actionSize.Count();
            for (int i = 0; i < myEnvironments[0].continuousActionBounds.Count(); i++)
            {
                var mean = result.select(1, offset + i * 2).item<float>();
                actions.Add(mean);
            }
            return actions.ToArray();
        }


        #endregion

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
        //TODO: wtf step horizon
        int stepHorizon = 400;
        int stepCounter = 0;
        public void Step(bool isTraining = true     )
        {
            if (!initialisetrainingonce)
            {
                InitialiseTraining();
            }

            foreach (var episode in episodes)
            {
                episode.Step(isTraining);
                stepCounter++;
            }

            episodeCounter++;
            if (true)
            {
                if(isTraining)
                    OptimizeModel();

                stepCounter = 0;
            }


            //TODO: Update chart (maybe with the first agent?)
        }

        internal class Episode
        {
            T currentState;
            Guid currentGuid;
            float cumulativeReward = 0;

            IContinuousEnvironment<T> myEnv;
            PPOAgent<T> myAgent;
            List<TransitionPortable<T>> episodeBuffer;
            Tensor? memoryState = null;
            List<T> statesHistory = new();

            public Episode(IContinuousEnvironment<T> myEnv, PPOAgent<T> agent)
            {
                this.myEnv = myEnv;
                myAgent = agent;
                myEnv.Reset();
                currentState = myAgent.DeepCopy(myEnv.GetCurrentState());
                episodeBuffer = new List<TransitionPortable<T>>();
                currentGuid = Guid.NewGuid();
            }


            public void Step(bool isTraining)
            {
                if (!myEnv.isDone)
                {
                    (int[], float[]) action;

                    if(!myAgent.myOptions.UseRNN)
                    {
                        action = myAgent.SelectAction(currentState, isTraining);
                    }
                    else
                    {
                        //statesHistory.Add(currentState);
                        //action = myAgent.SelectAction(statesHistory);
                        var result = myAgent.SelectAction(currentState, memoryState);
                        action = result.Item1;
                        memoryState = (result.Item2);

                    }
                    var reward = myEnv.Step(action.Item1, action.Item2);
                    var done = myEnv.isDone;

                    T nextState;
                    Guid? nextGuid;
                    if (done)
                    {
                        nextGuid = null;
                        nextState = default;
                    }
                    else
                    {
                        nextGuid = Guid.NewGuid();
                        nextState = myAgent.DeepCopy(myEnv.GetCurrentState());
                    }
                    cumulativeReward += reward;
                    episodeBuffer.Add(new TransitionPortable<T>(currentGuid, currentState, action.Item1, action.Item2, reward, nextGuid));
                    currentState = nextState;
                    currentGuid = nextGuid ?? Guid.NewGuid();
                    return;
                }
                myAgent.myReplayBuffer.Push(episodeBuffer.ToTransitionInMemory<T>());
                
                episodeBuffer = new();
                memoryState = null;
                statesHistory = new();
                var rewardCopy = cumulativeReward;
                myAgent.episodeRewards.Add(rewardCopy);
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

        public static void CreateTensorsFromTransitions(ref Device device, ref ReadOnlySpan<TransitionInMemory<T>> transitions, out Tensor stateBatch, out Tensor discreteActionBatch, out Tensor continuousActionBatch, out Tensor rewardBatch, out Tensor doneBatch)
        {
            int length = transitions.Length;
            var fixedDiscreteActionSize = transitions[0].discreteActions.Length;
            var fixedContinuousActionSize = transitions[0].continuousActions.Length;

            // Pre-allocate arrays based on the known batch size
            float[] batchRewards = new float[length];
            int[] flatDiscreteActions = new int[length * fixedDiscreteActionSize];
            float[] flatContinuousActions = new float[length * fixedContinuousActionSize];
            T[] batchStates = new T[length];
            int[] batchDone = new int[length];

            int flatDiscreteActionsIndex = 0;
            int flatContinuousActionsIndex = 0;

            for (int i = 0; i < length; i++)
            {
                var transition = transitions[i];
                batchRewards[i] = transition.reward;
                batchDone[i] = transition.nextState == null ? 1 : 0;
                Array.Copy(transition.discreteActions, 0, flatDiscreteActions, flatDiscreteActionsIndex, fixedDiscreteActionSize);
                flatDiscreteActionsIndex += fixedDiscreteActionSize;
                Array.Copy(transition.continuousActions, 0, flatContinuousActions, flatContinuousActionsIndex, fixedContinuousActionSize);
                flatContinuousActionsIndex += fixedContinuousActionSize;
                batchStates[i] = transition.state;
            }

            stateBatch = QOptimizerUtils<T>.StateBatchToTensor(batchStates, device);
            rewardBatch = torch.tensor(batchRewards, device: device);
            doneBatch = torch.tensor(batchDone, device: device);
            discreteActionBatch = torch.tensor(flatDiscreteActions, new long[] { length, fixedDiscreteActionSize }, torch.int64, device: device);
            continuousActionBatch = torch.tensor(flatContinuousActions, new long[] { length, fixedContinuousActionSize }, device: device);
        }
        public virtual void OptimizeModel()
        {
            if(myOptions.UseRNN && myOptions.BatchSize > 1)
            {
                throw new ArgumentException("Batch size larter than 1 is not yet supported with RNN");
            }

            if (myReplayBuffer.Length < myOptions.BatchSize)
            {
                return;
            }

            //TODO: implement SPAN stuff
            var transitions = myReplayBuffer.SampleEntireMemory();
            CreateTensorsFromTransitions(ref myDevice, ref transitions, out var stateBatch, out var discreteActionBatch, out var continuousActionBatch, out var rewardBatch, out var doneBatch);
            Tensor actionBatch = torch.cat(new Tensor[] { discreteActionBatch, continuousActionBatch }, dim: 1);

            Tensor policyOld = myActorNet.get_log_prob(stateBatch, actionBatch, myEnvironments[0].actionSize.Count(), myEnvironments[0].continuousActionBounds.Count()).squeeze().detach();
            Tensor valueOld = myCriticNet.forward(stateBatch).detach();


            var discountedRewards = RewardDiscount(ref transitions);
            //var advantages = AdvantageDiscountPrevious(rewardBatch, valueOld, doneBatch);
            var advantages = AdvantageDiscount(ref transitions, valueOld);

        
            if (policyOld.dim() > 1)
            {
                advantages = advantages.unsqueeze(1);
            }

            for (int i = 0; i < myOptions.PPOEpochs; i++)
            {
                // Calculate new policy and value estimates
                Tensor policy = myActorNet.get_log_prob(stateBatch, actionBatch, myEnvironments[0].actionSize.Count(), myEnvironments[0].continuousActionBounds.Count()).squeeze();
                Tensor values = myCriticNet.forward(stateBatch);;
                // Policy loss calculation
                Tensor ratios = torch.exp(policy - policyOld);



                Tensor surr1 = ratios * advantages;
                Tensor surr2 = torch.clamp(ratios, 1.0 - myOptions.ClipEpsilon, 1.0 + myOptions.ClipEpsilon) * advantages;
                Tensor entropy = myActorNet.ComputeEntropy(stateBatch, myEnvironments[0].actionSize.Count(), myEnvironments[0].continuousActionBounds.Count());
                Tensor actorLoss = -torch.min(surr1, surr2).mean() - myOptions.EntropyCoefficient * entropy.mean();

                // Optimize policy network
                myActorOptimizer.zero_grad();
                actorLoss.backward();
                torch.nn.utils.clip_grad_norm_(myActorNet.parameters(), myOptions.ClipGradNorm);
                myActorOptimizer.step();

                // Clipped value function loss 

                Tensor valueClipped = valueOld + torch.clamp(values - valueOld, -myOptions.VClipRange, myOptions.VClipRange);
                Tensor valueLoss1 = torch.pow(values - discountedRewards, 2);
                Tensor valueLoss2 = torch.pow(valueClipped - discountedRewards, 2);
                Tensor criticLoss = myOptions.CValue * torch.max(valueLoss1, valueLoss2).mean();

                // Optimize value network (updated to use the new criticLoss)
                myCriticOptimizer.zero_grad();
                criticLoss.backward();
                torch.nn.utils.clip_grad_norm_(myCriticNet.parameters(), myOptions.ClipGradNorm);
                myCriticOptimizer.step();
            }

            myReplayBuffer.ClearMemory();
        }

        Tensor RewardDiscount(ref ReadOnlySpan<TransitionInMemory<T>> transitions)
        {
            var batchSize = transitions.Length;

            if (batchSize == 1)
            {
                return torch.tensor(transitions[0].reward).to(myDevice);
            }

            // Initialize tensors
            Tensor returns = torch.zeros(batchSize).to(myDevice);

            // Find all the last transitions using LINQ
            var lastTransitions = transitions.ToArray()
                .Where(t => t.nextTransition == null)
                .Select((t, i) => (Transition: t, Index: i))
                .ToList();

            foreach (var (lastTransition, endIndex) in lastTransitions)
            {
                double discountedReward = 0;
                var currentTransition = lastTransition;
                int transitionIndex = 0;

                while (currentTransition != null)
                {
                    discountedReward = currentTransition.reward + myOptions.Gamma * discountedReward;
                    returns[endIndex - transitionIndex] = (float)discountedReward;

                    currentTransition = currentTransition.previousTransition;
                    transitionIndex++;
                }
            }

            return returns;
        }

        Tensor AdvantageDiscount(ref ReadOnlySpan<TransitionInMemory<T>> transitions, Tensor values)
        {

            var batchSize = transitions.Length;

            if (batchSize == 1)
            {
                return torch.tensor(transitions[0].reward).to(myDevice);
            }

            float[] advantages = new float[batchSize];
            float[] valueArray = QOptimizerUtils<T>.ExtractTensorData(values);  

            // Find all the last transitions using LINQ
            var lastTransitions = transitions.ToArray()
                .Where(t => t.nextTransition == null)
                .Select(t => (t))
                .ToList();

            foreach( var lastTransition in lastTransitions)
            {
                float runningAdd = 0;
                var currentTransition = lastTransition;
                int transitionIndex = transitions.IndexOf(currentTransition);

                while (currentTransition != null)
                {
                    float nextValue = currentTransition.nextTransition != null ? valueArray[transitionIndex + 1] : 0f;
                    float tdError = currentTransition.reward + myOptions.Gamma * nextValue - valueArray[transitionIndex];
                    runningAdd = tdError + myOptions.Gamma * myOptions.GaeLambda * runningAdd;

                    advantages[transitionIndex] = runningAdd;

                    currentTransition = currentTransition.previousTransition;
                    if (currentTransition != null)
                    transitionIndex = transitions.IndexOf(currentTransition);
                }
            }

            // Convert the advantages array to a tensor and normalize it
            Tensor advantagesTensor = torch.tensor(advantages).to(myDevice);
            advantagesTensor = (advantagesTensor - advantagesTensor.mean()) / (advantagesTensor.std() + 1e-10);

            return advantagesTensor;
        }

        Tensor AdvantageDiscountPrevious(Tensor rewards, Tensor values, Tensor dones)
        {
            var batchSize = rewards.size(0);

            if (batchSize == 1)
            {
                return rewards;
            }

            // Initialize tensors
            Tensor advantages = torch.zeros_like(rewards).to(myDevice);
            Tensor runningAdd = torch.tensor(0).to(myDevice);

            Tensor nextValue = torch.tensor(0).to(myDevice);  // This is V(s_{t+1}), initialized to 0 for the last timestep.

            // Iterate downwards through the batch
            for (long t = batchSize - 1; t >= 0; t--)
            {
                // Check if the current step is the end of an episode
                if (dones[t].item<int>() == 1)
                {
                    nextValue = torch.tensor(0).to(myDevice);  // Reset to 0 if the episode is done.
                    runningAdd = torch.tensor(0).to(myDevice); // Reset to 0 if the episode is done.
                }

                // Calculate the temporal difference (TD) error
                Tensor tdError = rewards[t] + myOptions.Gamma * nextValue * (1 - dones[t]) - values[t];

                // Update the running sum of discounted advantages
                runningAdd = runningAdd * myOptions.Gamma * myOptions.GaeLambda + tdError;

                // Store the advantage value for the current step
                advantages[t] = runningAdd.reshape(advantages[t].shape);


                // Update nextValue for the next iteration (going backward)
                nextValue = values[t];
            }


            // Normalize the advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10);
            return advantages;
        }

        protected T DeepCopy(T input)
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

        protected Tensor StateToTensor(T state)
        {
            switch (state)
            {
                case float[] stateArray:
                    return tensor(stateArray).to(myDevice);
                case float[,] stateMatrix:
                    return tensor(stateMatrix).to(myDevice);
                case null:
                    {
                        throw new ArgumentNullException("State cannot be null");
                    }
                default:
                    {
                        throw new InvalidCastException("State must be either float[] or float[,]");
                    }
                    
            }
        }

    }


}


