using RLMatrix.Agents.Common;

namespace RLMatrix
{


    /// <summary>
    /// Defines options for a PPO (Proximal Policy Optimization) Agent, a type of reinforcement learning agent.
    /// </summary>
    public class PPOAgentOptions : IAgentOptions    
    {
        /// <summary>
        /// The number of episodes that will be in each training batch. When number of episodes is smaller than batch size no optimization will occur and agent accumulates expierneces until this number is greater or equal. The replay buffer is deleted after each optimization step.
        /// </summary>
        public int BatchSize;

        /// <summary>
        /// The maximum number of previous experiences the agent can keep for future learning. As new experiences are added, the oldest ones are discarded. Defaults to 10000.
        /// </summary>
        public int MemorySize;

        /// <summary>
        /// Determines the importance of future rewards when the agent calculates the discounted cumulative return. A higher value gives greater importance to long-term rewards. Defaults to 0.99.
        /// </summary>
        public float Gamma;

        /// <summary>
        /// Lambda factor for Generalized Advantage Estimation. It controls the bias-variance trade-off in the estimation of advantage. Defaults to 0.95.
        /// </summary>
        public float GaeLambda;

        /// <summary>
        /// The learning rate for the agent's neural network. Higher values can make learning faster, but too high can make the learning process unstable and the agent may fail to learn. Defaults to 1e-5f.
        /// </summary>
        public float LR;


        /// <summary>
        /// Number of hidden layers in the neural network. Defaults to 2.
        /// </summary>
        public int Depth;

        /// <summary>
        /// Number of neurons in each hidden layer. Defaults to 1024.
        /// </summary>
        public int Width;

        /// <summary>
        /// Clipping factor for PPO's objective function. It prevents excessively large policy updates. Defaults to 0.2.
        /// </summary>
        public float ClipEpsilon;

        /// <summary>
        /// Clipping range for value loss. It limits the changes in the value function. Defaults to 0.2.
        /// </summary>
        public float VClipRange;

        /// <summary>
        /// Coefficient for value loss. It determines the contribution of value loss to the total loss. Defaults to 0.5.
        /// </summary>
        public float CValue;

        /// <summary>
        /// Number of PPO epochs. The number of times the algorithm will iterate through the whole batch during each update step. Defaults to 2.
        /// </summary>
        public int PPOEpochs;

        /// <summary>
        /// Maximum allowed gradient norm. It limits the magnitude of the gradients to prevent excessively large updates. Defaults to 0.5.
        /// </summary>
        public float ClipGradNorm;


        /// <summary>
        /// Entropy coefficient. It determines the contribution of entropy to the total loss. Defaults to 0.1.
        /// </summary>
        public float EntropyCoefficient;

        /// <summary>
        /// Determines whether the agent should use a recurrent neural network. Defaults to false.
        /// </summary>
        public bool UseRNN;

        public PPOAgentOptions(
            int batchSize = 16,
            int memorySize = 10000,
            float gamma = 0.99f,
            float gaeLambda = 0.95f,
            float lr = 1e-5f,
            int depth = 2,
            int width = 1024,
            float clipEpsilon = 0.2f,
            float vClipRange = 0.2f,
            float cValue = 0.5f,
            int ppoEpochs = 2,
            float clipGradNorm = 0.5f,
            float entropyCoefficient = 0.1f,
            bool useRNN = false)
        {
            BatchSize = batchSize;
            MemorySize = memorySize;
            Gamma = gamma;
            GaeLambda = gaeLambda;
            LR = lr;
            Depth = depth;
            Width = width;
            ClipEpsilon = clipEpsilon;
            VClipRange = vClipRange;
            CValue = cValue;
            PPOEpochs = ppoEpochs;
            ClipGradNorm = clipGradNorm;
            EntropyCoefficient = entropyCoefficient;
            UseRNN = useRNN;
        }
    }
}
