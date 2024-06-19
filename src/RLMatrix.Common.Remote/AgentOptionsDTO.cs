using MessagePack;
using OneOf;
namespace RLMatrix.Common.Remote;

[MessagePackObject]
public class AgentOptionsDTO
{
    [Key(0)]
    public string AgentType { get; set; }

    [Key(1)]
    public int BatchSize { get; set; }

    [Key(2)]
    public int MemorySize { get; set; }

    [Key(3)]
    public float Gamma { get; set; }

    [Key(4)]
    public float LR { get; set; }

    [Key(5)]
    public int Depth { get; set; }

    [Key(6)]
    public int Width { get; set; }

    // DQN-specific options
    [Key(7)]
    public float? EPS_START { get; set; }

    [Key(8)]
    public float? EPS_END { get; set; }

    [Key(9)]
    public float? EPS_DECAY { get; set; }

    [Key(10)]
    public float? TAU { get; set; }

    [Key(11)]
    public int? NumAtoms { get; set; }

    [Key(12)]
    public float? GaussianNoiseStd { get; set; }

    [Key(13)]
    public float? VMin { get; set; }

    [Key(14)]
    public float? VMax { get; set; }

    [Key(15)]
    public float? PriorityEpsilon { get; set; }

    [Key(16)]
    public int? SoftUpdateInterval { get; set; }

    [Key(17)]
    public int? NStepReturn { get; set; }

    [Key(18)]
    public bool? DoubleDQN { get; set; }

    [Key(19)]
    public bool? DuelingDQN { get; set; }

    [Key(20)]
    public bool? NoisyLayers { get; set; }

    [Key(21)]
    public bool? CategoricalDQN { get; set; }

    [Key(22)]
    public bool? PrioritizedExperienceReplay { get; set; }

    [Key(23)]
    public bool? BatchedInputProcessing { get; set; }

    [Key(24)]
    public bool? BoltzmannExploration { get; set; }

    [Key(25)]
    public float? NoisyLayersScale { get; set; }

    // PPO-specific options
    [Key(26)]
    public float? GaeLambda { get; set; }

    [Key(27)]
    public float? ClipEpsilon { get; set; }

    [Key(28)]
    public float? VClipRange { get; set; }

    [Key(29)]
    public float? CValue { get; set; }

    [Key(30)]
    public int? PPOEpochs { get; set; }

    [Key(31)]
    public float? ClipGradNorm { get; set; }

    [Key(32)]
    public float? EntropyCoefficient { get; set; }

    [Key(33)]
    public bool? UseRNN { get; set; }
}

public static class AgentOptionsDTOExtensions
{
    public static OneOf<DQNAgentOptions, PPOAgentOptions> ToAgentOptions(this AgentOptionsDTO dto)
    {
        if (dto.AgentType == "DQN")
        {
            return new DQNAgentOptions
            {
                BatchSize = dto.BatchSize,
                MemorySize = dto.MemorySize,
                GAMMA = dto.Gamma,
                LR = dto.LR,
                Depth = dto.Depth,
                Width = dto.Width,
                EPS_START = dto.EPS_START ?? 1.0f,
                EPS_END = dto.EPS_END ?? 0.005f,
                EPS_DECAY = dto.EPS_DECAY ?? 80f,
                TAU = dto.TAU ?? 0.5f,
                NumAtoms = dto.NumAtoms ?? 51,
                GaussianNoiseStd = dto.GaussianNoiseStd ?? 0.2f,
                VMin = dto.VMin ?? 1f,
                VMax = dto.VMax ?? 400f,
                PriorityEpsilon = dto.PriorityEpsilon ?? 0.01f,
                SoftUpdateInterval = dto.SoftUpdateInterval ?? 1,
                NStepReturn = dto.NStepReturn ?? 0,
                DoubleDQN = dto.DoubleDQN ?? false,
                DuelingDQN = dto.DuelingDQN ?? false,
                NoisyLayers = dto.NoisyLayers ?? false,
                CategoricalDQN = dto.CategoricalDQN ?? false,
                PrioritizedExperienceReplay = dto.PrioritizedExperienceReplay ?? false,
                BatchedInputProcessing = dto.BatchedInputProcessing ?? false,
                BoltzmannExploration = dto.BoltzmannExploration ?? false,
                NoisyLayersScale = dto.NoisyLayersScale ?? 0.00015f
            };
        }
        else if (dto.AgentType == "PPO")
        {
            return new PPOAgentOptions
            {
                BatchSize = dto.BatchSize,
                MemorySize = dto.MemorySize,
                Gamma = dto.Gamma,
                LR = dto.LR,
                Depth = dto.Depth,
                Width = dto.Width,
                GaeLambda = dto.GaeLambda ?? 0.95f,
                ClipEpsilon = dto.ClipEpsilon ?? 0.2f,
                VClipRange = dto.VClipRange ?? 0.2f,
                CValue = dto.CValue ?? 0.5f,
                PPOEpochs = dto.PPOEpochs ?? 2,
                ClipGradNorm = dto.ClipGradNorm ?? 0.5f,
                EntropyCoefficient = dto.EntropyCoefficient ?? 0.1f,
                UseRNN = dto.UseRNN ?? false
            };
        }
        else
        {
            throw new ArgumentException("Invalid agent type");
        }
    }
    public static AgentOptionsDTO ToAgentOptionsDTO(this OneOf<DQNAgentOptions, PPOAgentOptions> options)
    {
        return options.Match(
            dqn => new AgentOptionsDTO
            {
                AgentType = "DQN",
                BatchSize = dqn.BatchSize,
                MemorySize = dqn.MemorySize,
                Gamma = dqn.GAMMA,
                LR = dqn.LR,
                Depth = dqn.Depth,
                Width = dqn.Width,
                EPS_START = dqn.EPS_START,
                EPS_END = dqn.EPS_END,
                EPS_DECAY = dqn.EPS_DECAY,
                TAU = dqn.TAU,
                NumAtoms = dqn.NumAtoms,
                GaussianNoiseStd = dqn.GaussianNoiseStd,
                VMin = dqn.VMin,
                VMax = dqn.VMax,
                PriorityEpsilon = dqn.PriorityEpsilon,
                SoftUpdateInterval = dqn.SoftUpdateInterval,
                NStepReturn = dqn.NStepReturn,
                DoubleDQN = dqn.DoubleDQN,
                DuelingDQN = dqn.DuelingDQN,
                NoisyLayers = dqn.NoisyLayers,
                CategoricalDQN = dqn.CategoricalDQN,
                PrioritizedExperienceReplay = dqn.PrioritizedExperienceReplay,
                BatchedInputProcessing = dqn.BatchedInputProcessing,
                BoltzmannExploration = dqn.BoltzmannExploration,
                NoisyLayersScale = dqn.NoisyLayersScale
            },
            ppo => new AgentOptionsDTO
            {
                AgentType = "PPO",
                BatchSize = ppo.BatchSize,
                MemorySize = ppo.MemorySize,
                Gamma = ppo.Gamma,
                LR = ppo.LR,
                Depth = ppo.Depth,
                Width = ppo.Width,
                GaeLambda = ppo.GaeLambda,
                ClipEpsilon = ppo.ClipEpsilon,
                VClipRange = ppo.VClipRange,
                CValue = ppo.CValue,
                PPOEpochs = ppo.PPOEpochs,
                ClipGradNorm = ppo.ClipGradNorm,
                EntropyCoefficient = ppo.EntropyCoefficient,
                UseRNN = ppo.UseRNN
            }
        );
    }
}



