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

namespace RLMatrix
{
    public class PPOAgent<T>
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

        //TODO: Can this be managed? Can we have some object encapsulating all progress to peek inside current agent?
        public List<double> episodeRewards = new();


        public PPOAgent(PPOAgentOptions opts, OneOf<List<IContinuousEnvironment<T>>, List<IEnvironment<T>>> env, IPPONetProvider<T> netProvider = null)
        {
            netProvider = netProvider ?? new PPONetProviderBase<T>(1024);

            //check if T is either float[] or float[,]
            if (typeof(T) != typeof(float[]) && typeof(T) != typeof(float[,]))
            {
                throw new System.ArgumentException("T must be either float[] or float[,]");
            }
            myDevice = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine($"Running PPO on {myDevice.type.ToString()}");
            myOptions = opts;
            //TODO: Implement multi env
            //we need correct cast
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



            myActorNet = netProvider.CreateActorNet(myEnvironments[0]).to(myDevice);
            myCriticNet = netProvider.CreateCriticNet(myEnvironments[0]).to(myDevice);

            myActorOptimizer = optim.Adam(myActorNet.parameters(), myOptions.LR, amsgrad: true);
            myCriticOptimizer = optim.Adam(myCriticNet.parameters(), myOptions.LR * 100f, amsgrad: true);

            //TODO: I think I forgot to make PPO specific memory.
            myReplayBuffer = new ReplayMemory<T>(myOptions.MemorySize, myOptions.BatchSize);

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
            myCriticOptimizer = optim.Adam(myCriticNet.parameters(), myOptions.LR * 100f, amsgrad: true);
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
                Console.WriteLine(discreteActions[0]);
                return (discreteActions, continuousActions);
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
        int stepHorizon = 400;
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
            if (stepCounter > stepHorizon)
            {
                OptimizeModel();
                stepCounter = 0;
            }


            //TODO: Update chart (maybe with the first agent?)
        }

        internal class Episode
        {
            T currentState;
            float cumulativeReward = 0;

            IContinuousEnvironment<T> myEnv;
            PPOAgent<T> myAgent;
            List<Transition<T>> episodeBuffer;

            public Episode(IContinuousEnvironment<T> myEnv, PPOAgent<T> agent)
            {
                this.myEnv = myEnv;
                myAgent = agent;
                myEnv.Reset();
                currentState = myAgent.DeepCopy(myEnv.GetCurrentState());
                episodeBuffer = new List<Transition<T>>();
            }


            public void Step()
            {
                if (!myEnv.isDone)
                {
                    var action = myAgent.SelectAction(currentState);
                    var reward = myEnv.Step(action.Item1, action.Item2);
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
                    episodeBuffer.Add(new Transition<T>(currentState, action.Item1, action.Item2, reward, nextState));
                    currentState = nextState;
                    return;
                }
                
                foreach (var item in episodeBuffer)
                {
                    myAgent.myReplayBuffer.Push(item);
                }
                episodeBuffer = new();
                Console.WriteLine($"Episode finished with reward {cumulativeReward}");
                cumulativeReward = 0;
                myEnv.Reset();
                currentState = myAgent.DeepCopy(myEnv.GetCurrentState());

            }

        }
        #endregion


        public void TrainEpisode()
        {
            episodeCounter++;
            // Initialize the environment and get its state
            myEnvironments[0].Reset();
            T state = DeepCopy(myEnvironments[0].GetCurrentState());
            float cumulativeReward = 0;

            List<Transition<T>> transitionsInEpisode = new List<Transition<T>>();

            for (int t = 0; ; t++)
            {
                // Select an action based on the policy
                (int[], float[]) action = SelectAction(state);
                // Take a step using the selected action
                float reward = myEnvironments[0].Step(action.Item1, action.Item2);
                // Check if the episode is done
                var done = myEnvironments[0].isDone;

                T nextState;
                if (done)
                {
                    // If done, there is no next state
                    nextState = default;
                }
                else
                {
                    // If not done, get the next state
                    nextState = DeepCopy(myEnvironments[0].GetCurrentState());
                }

                if (state == null)
                    Console.WriteLine("state is null");

                // Store the transition in temporary memory
                transitionsInEpisode.Add(new Transition<T>(state, action.Item1, action.Item2, reward, nextState));

                cumulativeReward += reward;
                // If not done, move to the next state
                if (!done)
                {
                    state = nextState;
                }
                else
                {
                    //this ensures all the transitions are pushed together exactly when episode ends, so they are in order in the replay buffer
                    foreach (var item in transitionsInEpisode)
                    {
                        myReplayBuffer.Push(item);
                    }


                    OptimizeModel();


                    //TODO: hardcoded chart
                    episodeRewards.Add(cumulativeReward);

                    if(myOptions.DisplayPlot != null)
                    {
                        myOptions.DisplayPlot.CreateOrUpdateChart(episodeRewards);
                    }

                    break;
                }
            }
        }
        public virtual void OptimizeModel()
        {
            if (myReplayBuffer.Length < myOptions.BatchSize)
            {
                return;
            }

            List<Transition<T>> transitions = myReplayBuffer.SampleEntireMemory();

            // Convert to tensors
            Tensor stateBatch = stack(transitions.Select(t => StateToTensor(t.state)).ToArray()).to(myDevice);
            Tensor discreteActionBatch = stack(transitions.Select(t => tensor(t.discreteActions)).ToArray()).to(myDevice);
            Tensor continuousActionBatch = stack(transitions.Select(t => tensor(t.continuousActions)).ToArray()).to(myDevice);
            Tensor rewardBatch = stack(transitions.Select(t => tensor(t.reward)).ToArray()).to(myDevice);
            Tensor doneBatch = stack(transitions.Select(t => tensor(t.nextState == null ? 1 : 0)).ToArray()).to(myDevice);

            // Concatenate discrete and continuous action batches
            Tensor actionBatch = torch.cat(new Tensor[] { discreteActionBatch, continuousActionBatch }, dim: 1);

            Tensor policyOld = myActorNet.get_log_prob(stateBatch, actionBatch, myEnvironments[0].actionSize.Count(), myEnvironments[0].continuousActionBounds.Count()).squeeze().detach();
            Tensor valueOld = myCriticNet.forward(stateBatch).detach();

            var discountedRewards = RewardDiscount(rewardBatch, valueOld, doneBatch);
            var advantages = AdvantageDiscount(rewardBatch, valueOld, doneBatch);

            for (int i = 0; i < myOptions.PPOEpochs; i++)
            {
                Tensor policy = myActorNet.get_log_prob(stateBatch, actionBatch, myEnvironments[0].actionSize.Count(), myEnvironments[0].continuousActionBounds.Count()).squeeze();
                Tensor values = myCriticNet.forward(stateBatch).squeeze();

                Tensor ratios = torch.exp(policy - policyOld);
                Tensor surr1 = ratios * advantages;
                Tensor surr2 = torch.clamp(ratios, 1.0 - myOptions.ClipEpsilon, 1.0 + myOptions.ClipEpsilon) * advantages;

                Tensor actorLoss = -torch.min(surr1, surr2).mean();

                Tensor criticLoss = torch.pow(values - discountedRewards, 2).mean();
                // Optimize policy network
                myActorOptimizer.zero_grad();
                actorLoss.backward();
                torch.nn.utils.clip_grad_norm_(myActorNet.parameters(), myOptions.ClipGradNorm);
                myActorOptimizer.step();

                // Optimize value network
                myCriticOptimizer.zero_grad();
                criticLoss.backward();
                torch.nn.utils.clip_grad_norm_(myCriticNet.parameters(), myOptions.ClipGradNorm);
                myCriticOptimizer.step();
            }

            myReplayBuffer.ClearMemory();


            Tensor RewardDiscount(Tensor rewards, Tensor values, Tensor dones)
            {
                var batchSize = rewards.size(0);

                if (batchSize == 1)
                {
                    return rewards;
                }

                // Initialize tensors
                Tensor returns = torch.zeros_like(rewards).to(myDevice);
                double runningAdd = 0;

                // Iterate downwards through the batch
                for (long t = batchSize - 1; t >= 0; t--)
                {
                    // Check if the current step is the end of an episode
                    if (dones.cpu().ReadCpuValue<int>(t) == 1)
                    {
                        runningAdd = 0;
                    }

                    // Update the running sum of discounted rewards
                    runningAdd = runningAdd * myOptions.Gamma + rewards.cpu()[t].item<float>();

                    // Store the discounted reward for the current step
                    returns[t] = (float)runningAdd;
                }

                // Normalize the returns
                //returns = (returns - returns.mean()) / (returns.std() + 1e-10);
                return returns;
            }

            Tensor AdvantageDiscount(Tensor rewards, Tensor values, Tensor dones)
            {
                var batchSize = rewards.size(0);

                if (batchSize == 1)
                {
                    return rewards;
                }

                // Initialize tensors
                Tensor advantages = torch.zeros_like(rewards).to(myDevice);
                double runningAdd = 0;

                // Iterate downwards through the batch
                for (long t = batchSize - 1; t >= 0; t--)
                {
                    // Check if the current step is the end of an episode
                    if (dones.cpu().ReadCpuValue<int>(t) == 1)
                    {
                        runningAdd = 0;
                    }

                    // Calculate the temporal difference (TD) error
                    double tdError = rewards.cpu()[t].item<float>() + myOptions.Gamma * (t < batchSize - 1 ? values.cpu()[t + 1].item<float>() : 0) * (1 - dones.cpu().ReadCpuValue<int>(t)) - values.cpu()[t].item<float>();
                    // Update the running sum of discounted advantages
                    runningAdd = runningAdd * myOptions.Gamma * myOptions.GaeLambda + tdError;

                    // Store the advantage value for the current step
                    advantages[t] = (float)runningAdd;
                }

                // Normalize the advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10);
                return advantages;
            }

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
                default:
                    throw new InvalidCastException("State must be either float[] or float[,]");
            }
        }

    }


}
