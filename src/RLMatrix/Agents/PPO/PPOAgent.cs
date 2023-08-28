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

namespace RLMatrix
{
    public class PPOAgent<T>
    {
        protected torch.Device myDevice;
        protected PPOAgentOptions myOptions;
        protected IEnvironment<T> myEnvironment;
        protected PPOActorNet myActorNet;
        protected PPOCriticNet myCriticNet;
        protected OptimizerHelper myActorOptimizer;
        protected OptimizerHelper myCriticOptimizer;
        protected ReplayMemory<T> myReplayBuffer;
        protected int episodeCounter = 0;

        //TODO: Can this be managed? Can we have some object encapsulating all progress to peek inside current agent?
        public List<double> episodeRewards = new();


        public PPOAgent(PPOAgentOptions opts, IEnvironment<T> env, IPPONetProvider<T> netProvider = null)
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
            myEnvironment = env;

            myActorNet = netProvider.CreateActorNet(env).to(myDevice);
            myCriticNet = netProvider.CreateCriticNet(env).to(myDevice);

            myActorOptimizer = optim.Adam(myActorNet.parameters(), myOptions.LR, amsgrad: true);
            myCriticOptimizer = optim.Adam(myCriticNet.parameters(), myOptions.LR * 100f, amsgrad: true);


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

        public int SelectAction(T state, bool isTraining = true)
        {
            double sample = new Random().NextDouble();

            using (torch.no_grad())
            {
                Tensor stateTensor = StateToTensor(state);
                if (isTraining)
                {
                    var result = myActorNet.forward(stateTensor);

                    // Action is sampled from the probability distribution returned by the policy network
                    var action = SelectActionFromProbs(result);
                    return action;
                }
                else
                {
                    var result = myActorNet.forward(stateTensor);
                    // Select the action with the highest probability
                    var action = result.argmax(1);
                    return (int)action.item<long>();
                }
            }
        }

        private static int SelectActionFromProbs(Tensor probabilities)
        {
            // Sample an action from the probability distribution
            var action = torch.multinomial(probabilities, 1);
            return (int)action.item<long>();
        }

        float averageCumulativeReward = 0;
        int addedCount = 0;
        public void TrainEpisode()
        {
            episodeCounter++;
            // Initialize the environment and get its state
            myEnvironment.Reset();
            T state = DeepCopy(myEnvironment.GetCurrentState());
            float cumulativeReward = 0;

            List<Transition<T>> transitionsInEpisode = new List<Transition<T>>();

            for (int t = 0; ; t++)
            {
                // Select an action based on the policy
                var action = SelectAction(state);
                // Take a step using the selected action
                float reward = myEnvironment.Step(action);
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
                transitionsInEpisode.Add(new Transition<T>(state, action, reward, nextState));

                cumulativeReward += reward;
                // If not done, move to the next state
                if (!done)
                {
                    state = nextState;
                }
                else
                {
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
        public void OptimizeModel()
        {
            if (myReplayBuffer.Length < myOptions.BatchSize)
            {
                return;
            }


            List<Transition<T>> transitions = myReplayBuffer.SampleEntireMemory();

            // Convert to tensors
            Tensor stateBatch = stack(transitions.Select(t => StateToTensor(t.state)).ToArray()).to(myDevice);
            Tensor actionBatch = stack(transitions.Select(t => tensor(new int[] { t.action })).ToArray()).to(myDevice);
            Tensor rewardBatch = stack(transitions.Select(t => tensor(t.reward)).ToArray()).to(myDevice);
            Tensor doneBatch = stack(transitions.Select(t => tensor(t.nextState == null ? 1 : 0)).ToArray()).to(myDevice);

            Tensor policyOld = myActorNet.get_log_prob(stateBatch, actionBatch).squeeze().detach();
            Tensor valueOld = myCriticNet.forward(stateBatch).detach();


            var discountedRewards = RewardDiscount(rewardBatch, valueOld, doneBatch);
            var advantages = AdvantageDiscount(rewardBatch, valueOld, doneBatch);

            for (int i = 0; i < myOptions.PPOEpochs; i++)
            {
                Tensor policy = myActorNet.get_log_prob(stateBatch, actionBatch).squeeze();
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
