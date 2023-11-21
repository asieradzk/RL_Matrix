using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLMatrix
{
    /// <summary>
    /// Defines options for a PPO (Proximal Policy Optimization) Agent, a type of reinforcement learning agent.
    /// </summary>
    public class GAILOptions
    {
        public int BatchSize;
        public float LR;
        public int discriminatorEpochs;
        public float discriminatorTrainRate;
        public int NNWidth;
        public int NNDepth;
        public float rewardFactor;


        public GAILOptions(
            int batchSize = 16,
            float lr = 1e-5f,
            int discriminatorEpochs = 1,
            float discriminatorTrainRate = 1,
            int nNWidth = 1024,
            int nNDepth = 3,
            float rewardFactor = 5)
        {
            BatchSize = batchSize;
            LR = lr;
            this.discriminatorEpochs = discriminatorEpochs;
            this.discriminatorTrainRate = discriminatorTrainRate;
            this.discriminatorTrainRate = discriminatorTrainRate;
            NNWidth = nNWidth;
            NNDepth = nNDepth;
            this.rewardFactor = rewardFactor;
        }
    }
}

