namespace RLMatrix;

// TODO: XML docs - These defaults were taken from PPOAgentOptions, some defaults may be very bad or wrong.
public record GAILOptions(
    int BatchSize = 16,
    float LearningRate = 1e-5f,
    int DiscriminatorEpochCount = 2,
    float DiscriminatorTrainRate = 1e-5f, // pulled from the aether
    int NeuralNetworkWidth = 1024,
    int NeuralNetworkDepth = 2,
    float RewardFactor = 1f); // pulled from the aether