namespace RLMatrix
{
    /// <summary>
    /// Defines options for a PPO (Proximal Policy Optimization) Agent, a type of reinforcement learning agent.
    /// </summary>
    public class PPOAgentOptions
    {
        /// <summary>
        /// The number of experiences sampled from the memory buffer during each update step. Larger values may lead to more stable learning, but with increased computational cost and memory use. Defaults to 64.
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
        /// The learning rate for the agent's neural network. Higher values can make learning faster, but too high can make the learning process unstable and the agent may fail to learn. Defaults to 0.001.
        /// </summary>
        public float LR;

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
        /// Number of PPO epochs. The number of times the algorithm will iterate through the whole batch during each update step. Defaults to 4.
        /// </summary>
        public int PPOEpochs;

        /// <summary>
        /// Maximum allowed gradient norm. It limits the magnitude of the gradients to prevent excessively large updates. Defaults to 0.5.
        /// </summary>
        public float ClipGradNorm;

        /// <summary>
        /// Determines whether the agent's progress (i.e., reward over time) should be plotted. Defaults to null.
        /// </summary>
        public IRLChartService? DisplayPlot;

        public PPOAgentOptions(
            int batchSize = 64,
            int memorySize = 10000,
            float gamma = 0.99f,
            float gaeLambda = 0.95f,
            float lr = 0.001f,
            float clipEpsilon = 0.2f,
            float vClipRange = 0.2f,
            float cValue = 0.5f,
            int ppoEpochs = 4,
            float clipGradNorm = 0.5f,
            IRLChartService? displayPlot = null)
        {
            BatchSize = batchSize;
            MemorySize = memorySize;
            Gamma = gamma;
            GaeLambda = gaeLambda;
            LR = lr;
            ClipEpsilon = clipEpsilon;
            VClipRange = vClipRange;
            CValue = cValue;
            PPOEpochs = ppoEpochs;
            ClipGradNorm = clipGradNorm;
            DisplayPlot = displayPlot;
        }
    }
}
