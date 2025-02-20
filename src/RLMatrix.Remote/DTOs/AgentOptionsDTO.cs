using MessagePack;
using RLMatrix.Common;

namespace RLMatrix.Remote;

[MessagePackObject]
public class AgentOptionsDTO
{
    [Key(0)] 
    public string AgentType { get; set; } = null!;

    [Key(1)]
    public int BatchSize { get; set; }

    [Key(2)]
    public int MemorySize { get; set; }

    [Key(3)]
    public float Gamma { get; set; }

    [Key(4)]
    public float LearningRate { get; set; }

    [Key(5)]
    public int Depth { get; set; }

    [Key(6)]
    public int Width { get; set; }

    // DQN-specific options
    [Key(7)]
    public float? EpsilonStart { get; set; }

    [Key(8)]
    public float? EpsilonMin { get; set; }

    [Key(9)]
    public float? EpsilonDecay { get; set; }

    [Key(10)]
    public float? Tau { get; set; }

    [Key(11)]
    public int? NumberOfAtoms { get; set; }

    [Key(12)]
    public float? GaussianNoiseStandardDeviation { get; set; }

    [Key(13)]
    public float? ValueDistributionMin { get; set; }

    [Key(14)]
    public float? ValueDistributionMax { get; set; }

    [Key(15)]
    public float? PriorityEpsilon { get; set; }

    [Key(16)]
    public int? SoftUpdateInterval { get; set; }

    [Key(17)]
    public int? LookAheadSteps { get; set; }

    [Key(18)]
    public bool? UseDoubleDQN { get; set; }

    [Key(19)]
    public bool? UseDuelingDQN { get; set; }

    [Key(20)]
    public bool? UseNoisyLayers { get; set; }

    [Key(21)]
    public bool? UseCategoricalDQN { get; set; }

    [Key(22)]
    public bool? UsePrioritizedExperienceReplay { get; set; }

    [Key(23)]
    public bool? UseBatchedInputProcessing { get; set; }

    [Key(24)]
    public bool? UseBoltzmannExploration { get; set; }

    [Key(25)]
    public float? NoisyLayersScale { get; set; }

    // PPO-specific options
    [Key(26)]
    public float? GAELambda { get; set; }

    [Key(27)]
    public float? EpsilonClippingFactor { get; set; }

    [Key(28)]
    public float? ValueLossClipRange { get; set; }

    [Key(29)]
    public float? ValueLossCoefficient { get; set; }

    [Key(30)]
    public int? PPOEpochCount { get; set; }

    [Key(31)]
    public float? ClipGradientNorm { get; set; }

    [Key(32)]
    public float? EntropyCoefficient { get; set; }

    [Key(33)]
    public bool? UseRNN { get; set; }
}

public static class AgentOptionsDTOExtensions
{
    public static IAgentOptions ToAgentOptions(this AgentOptionsDTO dto)
    {
        if (dto.AgentType == "DQN")
        {
            return new DQNAgentOptions
            {
                BatchSize = dto.BatchSize,
                MemorySize = dto.MemorySize,
                Gamma = dto.Gamma,
                LearningRate = dto.LearningRate,
                Depth = dto.Depth,
                Width = dto.Width,
                EpsilonStart = dto.EpsilonStart ?? 1.0f,
                EpsilonMin = dto.EpsilonMin ?? 0.005f,
                EpsilonDecay = dto.EpsilonDecay ?? 80f,
                Tau = dto.Tau ?? 0.5f,
                NumberOfAtoms = dto.NumberOfAtoms ?? 51,
                GaussianNoiseStandardDeviation = dto.GaussianNoiseStandardDeviation ?? 0.2f,
                ValueDistributionMin = dto.ValueDistributionMin ?? 1f,
                ValueDistributionMax = dto.ValueDistributionMax ?? 400f,
                PriorityEpsilon = dto.PriorityEpsilon ?? 0.01f,
                SoftUpdateInterval = dto.SoftUpdateInterval ?? 1,
                LookAheadSteps = dto.LookAheadSteps ?? 0,
                UseDoubleDQN = dto.UseDoubleDQN ?? false,
                UseDuelingDQN = dto.UseDuelingDQN ?? false,
                UseNoisyLayers = dto.UseNoisyLayers ?? false,
                UseCategoricalDQN = dto.UseCategoricalDQN ?? false,
                UsePrioritizedExperienceReplay = dto.UsePrioritizedExperienceReplay ?? false,
                UseBatchedInputProcessing = dto.UseBatchedInputProcessing ?? false,
                UseBoltzmannExploration = dto.UseBoltzmannExploration ?? false,
                NoisyLayersScale = dto.NoisyLayersScale ?? 0.00015f
            };
        }
        
        if (dto.AgentType == "PPO")
        {
            return new PPOAgentOptions
            {
                BatchSize = dto.BatchSize,
                MemorySize = dto.MemorySize,
                Gamma = dto.Gamma,
                LearningRate = dto.LearningRate,
                Depth = dto.Depth,
                Width = dto.Width,
                GAELambda = dto.GAELambda ?? 0.95f,
                EpsilonClippingFactor = dto.EpsilonClippingFactor ?? 0.2f,
                ValueLossClipRange = dto.ValueLossClipRange ?? 0.2f,
                ValueLossCoefficient = dto.ValueLossCoefficient ?? 0.5f,
                PPOEpochCount = dto.PPOEpochCount ?? 2,
                ClipGradientNorm = dto.ClipGradientNorm ?? 0.5f,
                EntropyCoefficient = dto.EntropyCoefficient ?? 0.1f,
                UseRNN = dto.UseRNN ?? false
            };
        }
        
        throw new ArgumentException("Invalid agent type");
    }
    
    public static AgentOptionsDTO ToAgentOptionsDTO(this IAgentOptions options)
    {
        return options switch
        {
            DQNAgentOptions dqn => new AgentOptionsDTO
            {
                AgentType = "DQN",
                BatchSize = dqn.BatchSize,
                MemorySize = dqn.MemorySize,
                Gamma = dqn.Gamma,
                LearningRate = dqn.LearningRate,
                Depth = dqn.Depth,
                Width = dqn.Width,
                EpsilonStart = dqn.EpsilonStart,
                EpsilonMin = dqn.EpsilonMin,
                EpsilonDecay = dqn.EpsilonDecay,
                Tau = dqn.Tau,
                NumberOfAtoms = dqn.NumberOfAtoms,
                GaussianNoiseStandardDeviation = dqn.GaussianNoiseStandardDeviation,
                ValueDistributionMin = dqn.ValueDistributionMin,
                ValueDistributionMax = dqn.ValueDistributionMax,
                PriorityEpsilon = dqn.PriorityEpsilon,
                SoftUpdateInterval = dqn.SoftUpdateInterval,
                LookAheadSteps = dqn.LookAheadSteps,
                UseDoubleDQN = dqn.UseDoubleDQN,
                UseDuelingDQN = dqn.UseDuelingDQN,
                UseNoisyLayers = dqn.UseNoisyLayers,
                UseCategoricalDQN = dqn.UseCategoricalDQN,
                UsePrioritizedExperienceReplay = dqn.UsePrioritizedExperienceReplay,
                UseBatchedInputProcessing = dqn.UseBatchedInputProcessing,
                UseBoltzmannExploration = dqn.UseBoltzmannExploration,
                NoisyLayersScale = dqn.NoisyLayersScale
            },
            PPOAgentOptions ppo => new AgentOptionsDTO
            {
                AgentType = "PPO",
                BatchSize = ppo.BatchSize,
                MemorySize = ppo.MemorySize,
                Gamma = ppo.Gamma,
                LearningRate = ppo.LearningRate,
                Depth = ppo.Depth,
                Width = ppo.Width,
                GAELambda = ppo.GAELambda,
                EpsilonClippingFactor = ppo.EpsilonClippingFactor,
                ValueLossClipRange = ppo.ValueLossClipRange,
                ValueLossCoefficient = ppo.ValueLossCoefficient,
                PPOEpochCount = ppo.PPOEpochCount,
                ClipGradientNorm = ppo.ClipGradientNorm,
                EntropyCoefficient = ppo.EntropyCoefficient,
                UseRNN = ppo.UseRNN
            },
            _ => throw new ArgumentOutOfRangeException(nameof(options))
        };
    }
}



