using OneOf;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;
using TorchSharp.Modules;
using RLMatrix.Memories;

namespace RLMatrix.Agents.PPO.Variants
{
    public class PPOForTesting<T> : PPOAgent<T>
    {
        public PPOForTesting(PPOAgentOptions opts, OneOf<List<IContinuousEnvironment<T>>, List<IEnvironment<T>>> env, IPPONetProvider<T> netProvider = null) : base(opts, env, netProvider)
        {

        }

        public void ImplantMemory(EpisodicReplayMemory<T> memory)
        {
            myReplayBuffer = memory;
        }

        static float[] ConvertIntArrayToFloatArray(int[] intArray)
        {
            return Array.ConvertAll(intArray, item => (float)item);
        }
        Tensor ConvertActionsToPolicyFormat(Tensor actionTakenBatch)
        {
            var discreteActionsCount = myEnvironments[0].actionSize;
            var continuousActionsCount = myEnvironments[0].continuousActionBounds.Length;

            var batchSize = actionTakenBatch.size(0);

            Tensor[] policyActions = new Tensor[batchSize];

            for (int i = 0; i < batchSize; i++)
            {
                List<Tensor> batchPolicyActions = new List<Tensor>();
                int currentActionIndex = 0;

                // Process discrete actions
                foreach (var numActions in discreteActionsCount)
                {
                    Tensor discreteAction = actionTakenBatch[i, currentActionIndex];
                    Tensor oneHotDiscreteAction = torch.nn.functional.one_hot(discreteAction.to_type(torch.int64), numActions);
                    batchPolicyActions.Add(oneHotDiscreteAction.to_type(torch.float32));
                    currentActionIndex++;
                }

                // Process continuous actions
                for (int j = 0; j < continuousActionsCount; j++)
                {
                    Tensor continuousAction = actionTakenBatch[i, currentActionIndex].unsqueeze(0);
                    batchPolicyActions.Add(continuousAction);
                    currentActionIndex++;
                }

                // Combine discrete and continuous actions for this batch
                policyActions[i] = torch.cat(batchPolicyActions, dim: 1);
            }

            // Stack all batches together
            return torch.stack(policyActions, dim: 0);
        }







        //TODO: Only tested for 1 head discrete
        private void Retrain(int initializationEpochs, double learningRate, List<Transition<T>> transitions)
        {

            // Convert to tensors
            Tensor stateBatch = stack(transitions.Select(t => StateToTensor(t.state)).ToArray()).to(myDevice);
            Tensor discreteActionBatch = stack(transitions.Select(t => tensor(t.discreteActions)).ToArray()).to(myDevice);
            Tensor continuousActionBatch = stack(transitions.Select(t => tensor(t.continuousActions)).ToArray()).to(myDevice);
            Tensor rewardBatch = stack(transitions.Select(t => tensor(t.reward)).ToArray()).to(myDevice);
            Tensor doneBatch = stack(transitions.Select(t => tensor(t.nextState == null ? 1 : 0)).ToArray()).to(myDevice);

            // Concatenate discrete and continuous action batches
            Tensor actionBatch = torch.cat(new Tensor[] { discreteActionBatch, continuousActionBatch }, dim: 1);
            Tensor actionToPolicyBatch = ConvertActionsToPolicyFormat(actionBatch);

            


            var actorOptimizer = new Adam(myActorNet.parameters(), learningRate);
            var criticOptimizer = new Adam(myCriticNet.parameters(), learningRate);

            for (int epoch = 0; epoch < initializationEpochs; epoch++)
            {
                Tensor policy = myActorNet.forward(stateBatch).squeeze(1);

                Tensor predictedRewards = myCriticNet.forward(stateBatch).squeeze(1);


                var actorCriterion = torch.nn.MSELoss();
                var criticCriterion = torch.nn.MSELoss();

                rewardBatch.print();


                Tensor actorLoss = actorCriterion.forward(policy, actionToPolicyBatch);
                Tensor criticLoss = criticCriterion.forward(predictedRewards, rewardBatch);

                actorOptimizer.zero_grad();
                actorLoss.backward();
                actorOptimizer.step();


                criticOptimizer.zero_grad();
                criticLoss.backward();
                criticOptimizer.step();
                
            }

        }

        public void RetrainFromMemoryByAge(int initializationEpochs, double learningRate)
        {
            //Sweeps from earliest to latest in 5% chunks, improving the epochs and learning rate towards final values

            for (int iteration = 0; iteration < 20; iteration++)
            {
                var transitions = myReplayBuffer.SamplePortionOfMemory((iteration) * 5, 5);
                Retrain(initializationEpochs + ((1+ iteration)/10), learningRate * ((1+ iteration)/10), transitions);
            }
        }

        public void RetrainFromMemoryByReward(int initializationEpochs, double learningRate, int topPercent = 25)
        {
               //Sweeps from lowest to highest in 5% chunks, improving the epochs and learning rate towards final values

            var transitions = myReplayBuffer.SamplePortionOfMemoryByRewards(topPercent);
            Retrain(initializationEpochs, learningRate, transitions);

            
        }




        public override void OptimizeModel()
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
                //actorLoss.print();

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


    }
}
