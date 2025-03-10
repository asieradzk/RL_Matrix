namespace RLMatrix.Common;

/// <summary>
///     Defines options for a DQN (Deep Q-Network) Agent, a type of reinforcement learning agent.
/// </summary>
/// <param name="BatchSize">
///     The number of experiences sampled from the memory buffer during each update step.
///     Larger values may lead to more stable learning, but with increased computational cost and memory use.
/// </param>
/// <param name="MemorySize">
///     The maximum number of previous experiences the agent can keep for future learning.
///     As new experiences are added, the oldest ones are discarded.
/// </param>
/// <param name="Gamma">
///     Determines the importance of future rewards when the agent calculates the Q-value (quality of action).
///     A higher value gives greater importance to long-term rewards.
/// </param>
/// <param name="EpsilonStart">
///     The starting value for epsilon in the epsilon-greedy action selection strategy.
///     A higher epsilon encourages the agent to explore the environment by choosing random actions.
///     <remarks>Initially, the agent needs to explore a lot, hence the default is <c>1.0</c>.</remarks>
/// </param>
/// <param name="EpsilonMin">
///     The lowest possible value for epsilon in the epsilon-greedy action selection strategy.
///     Even when the agent has learned a lot about the environment, it should still explore occasionally,
///     hence epsilon never goes below this value.
/// </param>
/// <param name="EpsilonDecay">
///     The rate at which epsilon decreases.
///     As the agent learns about the environment, it needs to rely less on random actions and more on its learned policy.
///     This value controls how fast that transition happens.
/// </param>
/// <param name="Tau">
///     The rate at which the agent's policy network is updated to the target network.
///     Lower values mean the updates are more gradual, making the learning process more stable.
/// </param>
/// <param name="LearningRate">
///     The learning rate for the agent's neural network.
///     Higher values can make learning faster, but too high can make the learning process unstable and the agent may fail to learn.
/// </param>
/// <param name="Depth">
///     Number of hidden layers in the neural network.
/// </param>
/// <param name="Width">
///     Number of neurons in each hidden layer.
/// </param>
/// <param name="NumberOfAtoms">
///     The number of atoms used in the Categorical DQN algorithm.
/// </param>
/// <param name="GaussianNoiseStandardDeviation">
///     The standard deviation of the Gaussian noise used for exploration in the Noisy DQN algorithm.
/// </param>
/// <param name="ValueDistributionMin">
///     The minimum value of the support for the value distribution in the Categorical DQN algorithm.
/// </param>
/// <param name="ValueDistributionMax">
///     The maximum value of the support for the value distribution in the Categorical DQN algorithm.
/// </param>
/// <param name="PriorityEpsilon">
///     A small positive constant that is added to the priorities of the experiences in the Prioritized Experience Replay algorithm
///     to ensure that all experiences have a non-zero probability of being sampled.
/// </param>
/// <param name="SoftUpdateInterval">
///     The number of steps between each soft update of the target network in the Double DQN algorithm.
/// </param>
/// <param name="LookAheadSteps">
///     The number of steps to look ahead when calculating the target Q-value in the n-step DQN algorithm.
///     A value of 0 means that the agent uses the regular one-step TD target.
/// </param>
/// <param name="UseDoubleDQN">
///     Determines whether to use Double DQN algorithm.
/// </param>
/// <param name="UseDuelingDQN">
///     Determines whether to use Dueling DQN architecture.
/// </param>
/// <param name="UseNoisyLayers">
///     Determines whether to use Noisy DQN algorithm for exploration.
/// </param>
/// <param name="UseCategoricalDQN">
///     Determines whether to use Categorical DQN algorithm.
/// </param>
/// <param name="UsePrioritizedExperienceReplay">
///     Determines whether to use Prioritized Experience Replay algorithm.
/// </param>
/// <param name="UseBatchedInputProcessing">
///     Determines whether to use batched input processing for action selection during training.
///     To use this, you should guarantee some degree of stochasticity otherwise all agents in a batch will take the same action.
/// </param>
/// <param name="UseBoltzmannExploration">
///     Determines whether to use Boltzmann exploration instead of epsilon-greedy exploration.
/// </param>
/// <param name="NoisyLayersScale">
///     The scale of the noise used in the Noisy DQN algorithm.
/// </param>
public sealed record DQNAgentOptions(
    int BatchSize = 64,
    int MemorySize = 10_000,
    float Gamma = 0.99f,
    float EpsilonStart = 1.0f,
    float EpsilonMin = 0.005f,
    float EpsilonDecay = 80f,
    float Tau = 0.5f,
    float LearningRate = 0.001f,
    int Depth = 2,
    int Width = 1024,
    int NumberOfAtoms = 51,
    float GaussianNoiseStandardDeviation = 0.2f,
    float ValueDistributionMin = 1.0f,
    float ValueDistributionMax = 400f,
    float PriorityEpsilon = 0.01f,
    int SoftUpdateInterval = 1,
    int LookAheadSteps = 0,
    bool UseDoubleDQN = false,
    bool UseDuelingDQN = false,
    bool UseNoisyLayers = false,
    bool UseCategoricalDQN = false,
    bool UsePrioritizedExperienceReplay = false,
    bool UseBatchedInputProcessing = false,
    bool UseBoltzmannExploration = false,
    float NoisyLayersScale = 0.00015f) : IAgentOptions;