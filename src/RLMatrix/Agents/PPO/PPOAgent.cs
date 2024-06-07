
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using static TorchSharp.torch.optim;
using OneOf;
using RLMatrix.Memories;
using System.Security;
using System.Net.Http.Headers;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using RLMatrix.Agents.Common;

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
        protected ReplayMemory<T> myReplayBuffer;
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
            myReplayBuffer = new ReplayMemory<T>(myOptions.MemorySize);

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
                     //   statesHistory.Add(currentState);
                       // action = myAgent.SelectAction(statesHistory);
                        //memoryState = action.Item1;
                       
                        var result = myAgent.SelectAction(currentState, memoryState);

                        action = result.Item1;
                        memoryState = (result.Item2);
                        memoryState = memoryState?.detach();

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

        public static unsafe void CreateTensorsFromTransitions(ref Device device, IList<TransitionInMemory<T>> transitions, bool useRNN, out Tensor stateBatch, out Tensor actionBatch)
        {
            int length = transitions.Count;
            var fixedDiscreteActionSize = transitions[0].discreteActions.Length;
            var fixedContinuousActionSize = transitions[0].continuousActions.Length;

            // Pre-allocate arrays based on the known batch size
            transitions.UnpackTransitionInMemory(out T[] batchStates, out int[,] batchDiscreteActions, out float[,] batchContinuousActions);
            stateBatch = Utilities<T>.StateBatchToTensor(batchStates, device);
            using (var discreteActionBatch = torch.tensor(batchDiscreteActions, torch.int64, device: device))
            using (var continuousActionBatch = torch.tensor(batchContinuousActions, device: device))
            {
                actionBatch = torch.cat(new Tensor[] { discreteActionBatch, continuousActionBatch }, dim: 1);
            }
        }

        private static List<List<TransitionInMemory<T>>> PadTransitions(IList<TransitionInMemory<T>> transitions, Device device, out Tensor mask)
        {
            int length = transitions.Count;
            var fixedDiscreteActionSize = transitions[0].discreteActions.Length;
            var fixedContinuousActionSize = transitions[0].continuousActions.Length;
            var firstTransitions = transitions.ToArray()
                .Where(t => t.previousTransition == null)
                .Select(t => t)
                .ToList();

            Dictionary<TransitionInMemory<T>, int> sequenceLengths = new Dictionary<TransitionInMemory<T>, int>();
            foreach (var transition in firstTransitions)
            {
                int sequenceLength = CalculateSequenceLength(transition);
                sequenceLengths.Add(transition, sequenceLength);
            }

            var longestSequence = sequenceLengths.Values.Max();

            List<List<TransitionInMemory<T>>> paddedSequences = new List<List<TransitionInMemory<T>>>();
            foreach (var transition in firstTransitions)
            {
                List<TransitionInMemory<T>> paddedSequence = new List<TransitionInMemory<T>>();
                AddInitialSequence(transition, paddedSequence);
                PadSequence(paddedSequence[paddedSequence.Count - 1], longestSequence, sequenceLengths[transition], paddedSequence);
                paddedSequences.Add(paddedSequence);
            }


            #region mask
            mask = CreateMask(paddedSequences, sequenceLengths, device);
            return paddedSequences;

            Tensor CreateMask(List<List<TransitionInMemory<T>>> paddedSequences, Dictionary<TransitionInMemory<T>, int> sequenceLengths, Device device)
            {
                List<Tensor> maskList = new List<Tensor>();
                foreach (var sequence in paddedSequences)
                {
                    var originalLength = sequenceLengths[sequence[0]];
                    var paddedLength = sequence.Count;
                    bool[] maskData = Enumerable.Repeat(true, originalLength)
                                                .Concat(Enumerable.Repeat(false, paddedLength - originalLength))
                                                .ToArray();
                    maskList.Add(torch.tensor(maskData)); //.unsqueeze
                }

                var mask = torch.cat(maskList.ToArray(), dim: 0).to(device);
                return mask;

            }
            #endregion


            int CalculateSequenceLength(TransitionInMemory<T> transition)
            {
                if (transition.nextTransition == null)
                    return 1;

                return 1 + CalculateSequenceLength(transition.nextTransition);
            }

            void AddInitialSequence(TransitionInMemory<T> transition, List<TransitionInMemory<T>> paddedSequence)
            {
                while (transition != null)
                {
                    paddedSequence.Add(transition);
                    transition = transition.nextTransition;
                }
            }

            static TransitionInMemory<T> PadSequence(TransitionInMemory<T> transition, int targetLength, int currentLength, List<TransitionInMemory<T>> paddedSequence)
            {
               
                if (currentLength >= targetLength)
                    return transition;

                var paddedTransition = new TransitionInMemory<T>(
                    transition.state,
                    transition.discreteActions,
                    transition.continuousActions,
                    0f,
                    transition.state,
                    null,
                    transition
                );

                transition.nextTransition = paddedTransition;
                paddedSequence.Add(paddedTransition);

                var result = PadSequence(paddedTransition, targetLength, currentLength + 1, paddedSequence);
                return result;
            }
        }


        void OptimizeModelRNN()
        {
            using (var scope = torch.NewDisposeScope())
            {
                var paddedTransitions = PadTransitions(myReplayBuffer.SampleEntireMemory(), myDevice, out var mask);

                List<Tensor> stateBatches = new List<Tensor>();
                List<Tensor> actionBatches = new List<Tensor>();

                foreach (var sequence in paddedTransitions)
                {
                    CreateTensorsFromTransitions(ref myDevice, sequence, myOptions.UseRNN, out var stateBatchEpisode, out var actionBatchEpisode);
                    stateBatches.Add(stateBatchEpisode);
                    actionBatches.Add(actionBatchEpisode);
                }

                Tensor stateBatch = torch.stack(stateBatches.ToArray(), dim: 0);
                Tensor actionBatch = torch.cat(actionBatches.ToArray(), dim: 0);

                Tensor policyOld = myActorNet.get_log_prob(stateBatch, actionBatch, myEnvironments[0].actionSize.Count(), myEnvironments[0].continuousActionBounds.Count()).detach();
                Tensor valueOld = myCriticNet.forward(stateBatch).detach();
                Tensor maskedPolicyOld = policyOld * mask;
                Tensor maskedValueOld = valueOld * mask;

                List<Tensor> discountedRewardsList = new List<Tensor>();
                List<Tensor> advantagesList = new List<Tensor>();

                List<TransitionInMemory<T>> paddedTransitionsSummed = paddedTransitions.SelectMany(t => t).ToList();
                var (maskedDiscountedRewards, maskedAdvantages) = DiscountedRewardsAndAdvantages(paddedTransitionsSummed, maskedValueOld);
                maskedAdvantages *= mask;
                maskedDiscountedRewards *= mask;


                for (int i = 0; i < myOptions.PPOEpochs; i++)
                {
                    using (var actorScope = torch.NewDisposeScope())
                    {
                        (Tensor policy, Tensor entropy) = myActorNet.get_log_prob_entropy(stateBatch, actionBatch, myEnvironments[0].actionSize.Count(), myEnvironments[0].continuousActionBounds.Count());
                        policy *= mask;  // Apply mask to policy
                      //  entropy *= mask;  // Apply mask to entropy

                        Tensor ratios = torch.exp(policy - maskedPolicyOld);
                        Tensor surr1 = ratios * maskedAdvantages;
                        Tensor surr2 = torch.clamp(ratios, 1.0 - myOptions.ClipEpsilon, 1.0 + myOptions.ClipEpsilon) * maskedAdvantages;

                        // Select the non-masked surrogate values
                        Tensor surr = torch.masked_select(torch.min(surr1, surr2), mask.to_type(ScalarType.Bool));

                        // Select the non-masked entropy values
                        Tensor maskedEntropy = torch.masked_select(entropy, mask.to_type(ScalarType.Bool));

                        // Calculate the mean of the non-masked surrogate and entropy values
                        Tensor actorLoss = -surr.mean() - myOptions.EntropyCoefficient * maskedEntropy.mean();

                        myActorOptimizer.zero_grad();
                        actorLoss.backward();
                        torch.nn.utils.clip_grad_norm_(myActorNet.parameters(), myOptions.ClipGradNorm);
                        myActorOptimizer.step();
                    }

                    using (var criticScope = torch.NewDisposeScope())
                    {
                        Tensor values = myCriticNet.forward(stateBatch);
                        values *= mask;  // Apply mask to values
                        Tensor valueClipped = maskedValueOld + torch.clamp(values - maskedValueOld, -myOptions.VClipRange, myOptions.VClipRange);
                        Tensor valueLoss1 = torch.pow(values - maskedDiscountedRewards, 2);
                        Tensor valueLoss2 = torch.pow(valueClipped - maskedDiscountedRewards, 2);

                        // Select the non-masked loss values
                        Tensor valueLoss = torch.masked_select(torch.max(valueLoss1, valueLoss2), mask.to_type(ScalarType.Bool));

                        // Calculate the mean of the non-masked loss values
                        Tensor criticLoss = myOptions.CValue * valueLoss.mean();

                        myCriticOptimizer.zero_grad();
                        criticLoss.backward();
                        torch.nn.utils.clip_grad_norm_(myCriticNet.parameters(), myOptions.ClipGradNorm);
                        myCriticOptimizer.step();
                    }
                }
            }

            myReplayBuffer.ClearMemory();
        }


        public virtual void OptimizeModel()
        {
            if (myOptions.UseRNN && myOptions.BatchSize > 1)
            {
              //  throw new ArgumentException("Batch size larger than 1 is not yet supported with RNN");
            }

            if (myReplayBuffer.NumEpisodes < myOptions.BatchSize)
            {
                return;
            }

            if (myOptions.UseRNN)
            {
                OptimizeModelRNN();
                return;
            }

            using (var scope = torch.NewDisposeScope())
            {
                var transitions = myReplayBuffer.SampleEntireMemory();
                //ref transitions because it may be padded in case of RNN with unequal sequence lengths
                CreateTensorsFromTransitions(ref myDevice, transitions, myOptions.UseRNN, out var stateBatch, out var actionBatch);

                Tensor policyOld = myActorNet.get_log_prob(stateBatch, actionBatch, myEnvironments[0].actionSize.Count(), myEnvironments[0].continuousActionBounds.Count()).detach();
                Tensor valueOld = myCriticNet.forward(stateBatch).detach();

                (var discountedRewards, var advantages) = DiscountedRewardsAndAdvantages(transitions, valueOld);

                if (policyOld.dim() > 1)
                {
                    advantages = advantages.unsqueeze(1);
                }

                for (int i = 0; i < myOptions.PPOEpochs; i++)
                {
                    using (var actorScope = torch.NewDisposeScope())
                    {
                        //Tensor policy = myActorNet.get_log_prob(stateBatch, actionBatch, myEnvironments[0].actionSize.Count(), myEnvironments[0].continuousActionBounds.Count()).squeeze();
                        //Tensor entropy = myActorNet.ComputeEntropy(stateBatch, myEnvironments[0].actionSize.Count(), myEnvironments[0].continuousActionBounds.Count());

                        (Tensor policy, Tensor entropy) = myActorNet.get_log_prob_entropy(stateBatch, actionBatch, myEnvironments[0].actionSize.Count(), myEnvironments[0].continuousActionBounds.Count());
                        Tensor ratios = torch.exp(policy - policyOld);
                        Tensor surr1 = ratios * advantages;
                        Tensor surr2 = torch.clamp(ratios, 1.0 - myOptions.ClipEpsilon, 1.0 + myOptions.ClipEpsilon) * advantages;
                        Tensor actorLoss = -torch.min(surr1, surr2).mean() - myOptions.EntropyCoefficient * entropy.mean();

                        myActorOptimizer.zero_grad();
                        actorLoss.backward();
                        torch.nn.utils.clip_grad_norm_(myActorNet.parameters(), myOptions.ClipGradNorm);
                        myActorOptimizer.step();
                    }

                    using (var criticScope = torch.NewDisposeScope())
                    {
                        Tensor values = myCriticNet.forward(stateBatch);
                        Tensor valueClipped = valueOld + torch.clamp(values - valueOld, -myOptions.VClipRange, myOptions.VClipRange);
                        Tensor valueLoss1 = torch.pow(values - discountedRewards, 2);
                        Tensor valueLoss2 = torch.pow(valueClipped - discountedRewards, 2);
                        Tensor criticLoss = myOptions.CValue * torch.max(valueLoss1, valueLoss2).mean();

                        myCriticOptimizer.zero_grad();
                        criticLoss.backward();
                        torch.nn.utils.clip_grad_norm_(myCriticNet.parameters(), myOptions.ClipGradNorm);
                        myCriticOptimizer.step();
                    }
                }
            }

            myReplayBuffer.ClearMemory();
        }
        (Tensor, Tensor) DiscountedRewardsAndAdvantages(IList<TransitionInMemory<T>> transitions, Tensor values)
        {
            var batchSize = transitions.Count();

            if (batchSize == 1)
            {
                return (torch.tensor(transitions.First().reward).to(myDevice), torch.tensor(transitions.First().reward).to(myDevice));
            }

            float[] discountedRewards = new float[batchSize];
            float[] advantages = new float[batchSize];
            float[] valueArray = Utilities<T>.ExtractTensorData(values);

            var lastTransitions = transitions.ToArray()
                .Where(t => t.nextTransition == null)
                .Select(t => t)
                .ToList();

            foreach (var lastTransition in lastTransitions)
            {
                double discountedReward = 0;
                float runningAdd = 0;
                var currentTransition = lastTransition;
                int transitionIndex = transitions.IndexOf(currentTransition);

                while (currentTransition != null)
                {
                    discountedReward = currentTransition.reward + myOptions.Gamma * discountedReward;
                    discountedRewards[transitionIndex] = (float)discountedReward;

                    float nextValue = currentTransition.nextTransition != null ? valueArray[transitionIndex + 1] : 0f;
                    float tdError = currentTransition.reward + myOptions.Gamma * nextValue - valueArray[transitionIndex];
                    runningAdd = tdError + myOptions.Gamma * myOptions.GaeLambda * runningAdd;
                    advantages[transitionIndex] = runningAdd;

                    currentTransition = currentTransition.previousTransition;
                    if (currentTransition != null)
                        transitionIndex = transitions.IndexOf(currentTransition);
                }
            }

            Tensor discountedRewardsTensor = torch.tensor(discountedRewards).to(myDevice);
            Tensor advantagesTensor = torch.tensor(advantages).to(myDevice);
            advantagesTensor = (advantagesTensor - advantagesTensor.mean()) / (advantagesTensor.std() + 1e-10);


            return (discountedRewardsTensor, advantagesTensor);
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


