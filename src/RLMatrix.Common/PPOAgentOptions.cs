namespace RLMatrix.Common;

/// <summary>
///     Defines options for a PPO (Proximal Policy Optimization) Agent, a type of reinforcement learning agent.
/// </summary>
/// <param name="BatchSize">
///     The number of episodes that will be in each training batch. When number of episodes is smaller than batch size no optimization will occur
///     and agent accumulates experiences until this number is greater or equal. The replay buffer is deleted after each optimization step.
/// </param>
/// <param name="MemorySize">
///     The maximum number of previous experiences the agent can keep for future learning. As new experiences are added, the oldest ones are discarded.
/// </param>
/// <param name="Gamma">
///     Determines the importance of future rewards when the agent calculates the discounted cumulative return. A higher value gives greater importance to long-term rewards.
/// </param>
/// <param name="GAELambda">
///     Lambda factor for Generalized Advantage Estimation. It controls the bias-variance trade-off in the estimation of advantage.
/// </param>
/// <param name="LearningRate">
///     The learning rate for the agent's neural network. Higher values can make learning faster, but too high can make the learning process unstable and the agent may fail to learn.
/// </param>
/// <param name="Depth">
///     Number of hidden layers in the neural network.
/// </param>
/// <param name="Width">
///     Number of neurons in each hidden layer.
/// </param>
/// <param name="EpsilonClippingFactor">
///     Clipping factor for PPO's objective function. It prevents excessively large policy updates.
/// </param>
/// <param name="ValueLossClipRange">
///     Clipping range for value loss. It limits the changes in the value function. 
/// </param>
/// <param name="ValueLossCoefficient">
///     Coefficient for value loss. It determines the contribution of value loss to the total loss.
/// </param>
/// <param name="PPOEpochCount">
///     Number of PPO epochs. The number of times the algorithm will iterate through the whole batch during each update step.
/// </param>
/// <param name="ClipGradientNorm">
///     Maximum allowed gradient norm. It limits the magnitude of the gradients to prevent excessively large updates.
/// </param>
/// <param name="EntropyCoefficient">
///     Entropy coefficient. It determines the contribution of entropy to the total loss.
/// </param>
/// <param name="UseRNN">
///     Determines whether the agent should use a recurrent neural network.
/// </param>
public sealed record PPOAgentOptions(
    int BatchSize = 16,
    int MemorySize = 10_000,
    float Gamma = 0.99f,
    float GAELambda = 0.95f,
    float LearningRate = 1e-5f,
    int Depth = 2,
    int Width = 1024,
    float EpsilonClippingFactor = 0.2f,
    float ValueLossClipRange = 0.2f,
    float ValueLossCoefficient = 0.5f,
    int PPOEpochCount = 2,
    float ClipGradientNorm = 0.5f,
    float EntropyCoefficient = 0.1f,
    bool UseRNN = false) : IAgentOptions;