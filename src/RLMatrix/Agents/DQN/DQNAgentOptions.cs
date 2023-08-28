namespace RLMatrix
{
    /// <summary>
    /// Defines options for a DQN (Deep Q-Network) Agent, a type of reinforcement learning agent.
    /// </summary>
    public class DQNAgentOptions
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
        /// Determines the importance of future rewards when the agent calculates the Q-value (quality of action). A higher value gives greater importance to long-term rewards. Defaults to 0.99.
        /// </summary>
        public float GAMMA;

        /// <summary>
        /// The starting value for epsilon in the epsilon-greedy action selection strategy. A higher epsilon encourages the agent to explore the environment by choosing random actions. Initially, the agent needs to explore a lot, hence the default is 1.0.
        /// </summary>
        public float EPS_START;

        /// <summary>
        /// The lowest possible value for epsilon in the epsilon-greedy action selection strategy. Even when the agent has learned a lot about the environment, it should still explore occasionally, hence epsilon never goes below this value. Defaults to 0.01.
        /// </summary>
        public float EPS_END;

        /// <summary>
        /// The rate at which epsilon decreases. As the agent learns about the environment, it needs to rely less on random actions and more on its learned policy. This value controls how fast that transition happens. Defaults to 80f.
        /// </summary>
        public float EPS_DECAY;

        /// <summary>
        /// The rate at which the agent's policy network is updated to the target network. Lower values mean the updates are more gradual, making the learning process more stable. Defaults to 0.01.
        /// </summary>
        public float TAU;

        /// <summary>
        /// The learning rate for the agent's neural network. Higher values can make learning faster, but too high can make the learning process unstable and the agent may fail to learn. Defaults to 0.001.
        /// </summary>
        public float LR;

        /// <summary>
        /// Determines whether the agent's progress (i.e., reward over time) should be plotted. Defaults to false.
        /// </summary>
        public IRLChartService? DisplayPlot;

        public DQNAgentOptions(
            int batchSize = 64,
            int memorySize = 10000,
            float gamma = 0.99f,
            float epsStart = 1.0f,
            float epsEnd = 0.01f,
            float epsDecay = 80f,
            float tau = 0.01f,
            float lr = 0.001f,
            IRLChartService? displayPlot = null)
        {
            BatchSize = batchSize;
            MemorySize = memorySize;
            GAMMA = gamma;
            EPS_START = epsStart;
            EPS_END = epsEnd;
            EPS_DECAY = epsDecay;
            TAU = tau;
            LR = lr;
            DisplayPlot = displayPlot;
        }
    }
}
