using RLMatrix.Agents.Common;

namespace RLMatrix
{
    /// <summary>
    /// Defines options for a DQN (Deep Q-Network) Agent, a type of reinforcement learning agent.
    /// </summary>
    public class DQNAgentOptions : RLMatrix.Agents.Common.IAgentOptions
    {
        /// <summary>
        /// The number of experiences sampled from the memory buffer during each update step.
        /// Larger values may lead to more stable learning, but with increased computational cost and memory use.
        /// Defaults to 64.
        /// </summary>
        public int BatchSize { get; set; } = 64;

        /// <summary>
        /// The maximum number of previous experiences the agent can keep for future learning.
        /// As new experiences are added, the oldest ones are discarded.
        /// Defaults to 10000.
        /// </summary>
        public int MemorySize { get; set; } = 10000;

        /// <summary>
        /// Determines the importance of future rewards when the agent calculates the Q-value (quality of action).
        /// A higher value gives greater importance to long-term rewards.
        /// Defaults to 0.99.
        /// </summary>
        public float GAMMA { get; set; } = 0.99f;

        /// <summary>
        /// The starting value for epsilon in the epsilon-greedy action selection strategy.
        /// A higher epsilon encourages the agent to explore the environment by choosing random actions.
        /// Initially, the agent needs to explore a lot, hence the default is 1.0.
        /// </summary>
        public float EPS_START { get; set; } = 1.0f;

        /// <summary>
        /// The lowest possible value for epsilon in the epsilon-greedy action selection strategy.
        /// Even when the agent has learned a lot about the environment, it should still explore occasionally,
        /// hence epsilon never goes below this value.
        /// Defaults to 0.005.
        /// </summary>
        public float EPS_END { get; set; } = 0.005f;

        /// <summary>
        /// The rate at which epsilon decreases.
        /// As the agent learns about the environment, it needs to rely less on random actions and more on its learned policy.
        /// This value controls how fast that transition happens.
        /// Defaults to 80f.
        /// </summary>
        public float EPS_DECAY { get; set; } = 80f;

        /// <summary>
        /// The rate at which the agent's policy network is updated to the target network.
        /// Lower values mean the updates are more gradual, making the learning process more stable.
        /// Defaults to 0.5.
        /// </summary>
        public float TAU { get; set; } = 0.5f;

        /// <summary>
        /// The learning rate for the agent's neural network.
        /// Higher values can make learning faster, but too high can make the learning process unstable and the agent may fail to learn.
        /// Defaults to 0.001.
        /// </summary>
        public float LR { get; set; } = 0.001f;

        /// <summary>
        /// Number of hidden layers in the neural network.
        /// Defaults to 2.
        /// </summary>
        public int Depth { get; set; } = 2;

        /// <summary>
        /// Number of neurons in each hidden layer.
        /// Defaults to 1024.
        /// </summary>
        public int Width { get; set; } = 1024;

        /// <summary>
        /// The number of atoms used in the Categorical DQN algorithm.
        /// Defaults to 51.
        /// </summary>
        public int NumAtoms { get; set; } = 51;

        /// <summary>
        /// The standard deviation of the Gaussian noise used for exploration in the Noisy DQN algorithm.
        /// Defaults to 0.2.
        /// </summary>
        public float GaussianNoiseStd { get; set; } = 0.2f;

        /// <summary>
        /// The minimum value of the support for the value distribution in the Categorical DQN algorithm.
        /// Defaults to 1.
        /// </summary>
        public float VMin { get; set; } = 1f;

        /// <summary>
        /// The maximum value of the support for the value distribution in the Categorical DQN algorithm.
        /// Defaults to 400.
        /// </summary>
        public float VMax { get; set; } = 400f;

        /// <summary>
        /// A small positive constant that is added to the priorities of the experiences in the Prioritized Experience Replay algorithm
        /// to ensure that all experiences have a non-zero probability of being sampled.
        /// Defaults to 0.01.
        /// </summary>
        public float PriorityEpsilon { get; set; } = 0.01f;

        /// <summary>
        /// The number of steps between each soft update of the target network in the Double DQN algorithm.
        /// Defaults to 1.
        /// </summary>
        public int SoftUpdateInterval { get; set; } = 1;

        /// <summary>
        /// The number of steps to look ahead when calculating the target Q-value in the n-step DQN algorithm.
        /// A value of 0 means that the agent uses the regular one-step TD target.
        /// Defaults to 0.
        /// </summary>
        public int NStepReturn { get; set; } = 0;

        /// <summary>
        /// Determines whether to use Double DQN algorithm.
        /// Defaults to false.
        /// </summary>
        public bool DoubleDQN { get; set; } = false;

        /// <summary>
        /// Determines whether to use Dueling DQN architecture.
        /// Defaults to false.
        /// </summary>
        public bool DuelingDQN { get; set; } = false;

        /// <summary>
        /// Determines whether to use Noisy DQN algorithm for exploration.
        /// Defaults to false.
        /// </summary>
        public bool NoisyLayers { get; set; } = false;

        /// <summary>
        /// Determines whether to use Categorical DQN algorithm.
        /// Defaults to false.
        /// </summary>
        public bool CategoricalDQN { get; set; } = false;

        /// <summary>
        /// Determines whether to use Prioritized Experience Replay algorithm.
        /// Defaults to false.
        /// </summary>
        public bool PrioritizedExperienceReplay { get; set; } = false;

        /// <summary>
        /// Determines whether to use batched input processing for action selection during training.
        /// To use this, you should guarantee some degree of stochasticity otherwise all agents in a batch will take the same action.
        /// Defaults to false.
        /// </summary>
        public bool BatchedInputProcessing { get; set; } = false;

        /// <summary>
        /// Determines whether to use Boltzmann exploration instead of epsilon-greedy exploration.
        /// Defaults to false.
        /// </summary>
        public bool BoltzmannExploration { get; set; } = false;

        /// <summary>
        /// The scale of the noise used in the Noisy DQN algorithm.
        /// Defaults to 0.00015.
        /// </summary>
        public float NoisyLayersScale { get; set; } = 0.00015f;


        public DQNAgentOptions(
            int batchSize = 64,
            int memorySize = 10000,
            float gamma = 0.99f,
            float epsStart = 1.0f,
            float epsEnd = 0.005f,
            float epsDecay = 80f,
            float tau = 0.5f,
            float lr = 0.001f,
            int depth = 2,
            int width = 1024,
            int numAtoms = 51,
            float gaussianNoiseStd = 0.2f,
            float vMin = 1f,
            float vMax = 400f,
            float priorityEpsilon = 0.01f,
            int softUpdateInterval = 1,
            int nStepReturn = 0,
            bool doubleDQN = false,
            bool duelingDQN = false,
            bool noisyLayers = false,
            bool categoricalDQN = false,
            bool prioritizedExperienceReplay = false,
            bool batchedActionProcessing = false,
            bool boltzmannExploration = false,
            float noisyLayersScale = 0.00015f)
        {
            BatchSize = batchSize;
            MemorySize = memorySize;
            GAMMA = gamma;
            EPS_START = epsStart;
            EPS_END = epsEnd;
            EPS_DECAY = epsDecay;
            TAU = tau;
            LR = lr;
            Depth = depth;
            Width = width;
            NumAtoms = numAtoms;
            GaussianNoiseStd = gaussianNoiseStd;
            VMin = vMin;
            VMax = vMax;
            PriorityEpsilon = priorityEpsilon;
            SoftUpdateInterval = softUpdateInterval;
            NStepReturn = nStepReturn;
            DoubleDQN = doubleDQN;
            DuelingDQN = duelingDQN;
            NoisyLayers = noisyLayers;
            CategoricalDQN = categoricalDQN;
            PrioritizedExperienceReplay = prioritizedExperienceReplay;
            NoisyLayersScale = noisyLayersScale;
            BatchedInputProcessing = batchedActionProcessing;
            BoltzmannExploration = boltzmannExploration;
        }
    }
}
