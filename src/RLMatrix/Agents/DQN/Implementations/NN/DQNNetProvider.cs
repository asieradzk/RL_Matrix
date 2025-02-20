namespace RLMatrix;

/// <summary>
///     Default implementation of IDQNNetProvider, makes network with 3 hidden layers
/// </summary>
public sealed class DQNNetProvider : IDQNNetProvider
{
    private enum NetworkType
    {
        Vanilla,
        Dueling,
        Categorical,
        CategoricalDueling
    }

    private readonly int _neuronsPerLayer;
    private readonly int _depth;
    private readonly NetworkType _networkType;
    
    /// <summary>
    ///     Constructs a simple DQN NN provider.
    /// </summary>
    /// <param name="neuronsPerLayer">Number of neurons in hidden layers.</param>
    /// <param name="depth">Depth of hidden layers.</param>
    /// <param name="useDueling">Whether to use dueling DQN.</param>
    /// <param name="categorical">Whether to use Categorical DQN.</param>
    public DQNNetProvider(int neuronsPerLayer = 256, int depth = 2, bool useDueling = false, bool categorical = false)
    {
        _neuronsPerLayer = neuronsPerLayer;
        _depth = depth;

        _networkType = (categorical, useDueling) switch
        {
            (true, true) => NetworkType.CategoricalDueling,
            (true, false) => NetworkType.Categorical,
            (false, true) => NetworkType.Dueling,
            (false, false) => NetworkType.Vanilla
        };
    }

    public DQNNET CreateCriticNet<TState>(EnvironmentSizeDTO env, bool noisyLayers = false, float noiseScale = 0f, int numAtoms = 51)
        where TState : notnull
    {
        if (typeof(TState) == typeof(float[]))
        {
            if (env.StateDimensions is not { Dimensions: var dimensions, Dimensions.Length: 1 })
                throw new Exception("Unexpected observation dimension for 1D state.");
            
            var actionSize = env.DiscreteActionDimensions;
            return _networkType switch
            {
                NetworkType.Vanilla => new DQN1D("1DDQN", dimensions[0], _neuronsPerLayer, actionSize, _depth, noisyLayers, noiseScale),
                NetworkType.Dueling => new DuelingDQN("1DDuelingDQN", dimensions[0], _neuronsPerLayer, actionSize, _depth, noisyLayers, noiseScale),
                NetworkType.Categorical => new CategoricalDQN1D("1DCategoricalDQN", dimensions[0], _neuronsPerLayer, actionSize, _depth, numAtoms, noisyLayers, noiseScale),
                NetworkType.CategoricalDueling => new CategoricalDuelingDQN1D("1DCategoricalDuelingDQN", dimensions[0], _neuronsPerLayer, actionSize, _depth, numAtoms, noisyLayers, noiseScale),
                _ => throw new ArgumentOutOfRangeException(nameof(_networkType), _networkType, null)
            };
        }

        if (typeof(TState) == typeof(float[,]))
        {
            if (env.StateDimensions is not { Dimensions: var dimensions, Dimensions.Length: 2 })
                throw new Exception("Unexpected observation dimension for 2D state.");

            var (h, w) = (dimensions[0], dimensions[1]);
            var actionSize2 = env.DiscreteActionDimensions;
            return _networkType switch
            {
                NetworkType.Vanilla => new DQN2D("2DDQN", h, w, actionSize2, _neuronsPerLayer, depth: _depth, noisyLayers, noiseScale),
                NetworkType.Dueling => new DuelingDQN2D("2DDuelingDQN", h, w, actionSize2, _neuronsPerLayer, depth: _depth, noisyLayers, noiseScale),
                NetworkType.Categorical => new CategoricalDQN2D("2DCategoricalDQN", h, w, actionSize2, _neuronsPerLayer, depth: _depth, numAtoms, noisyLayers, noiseScale),
                NetworkType.CategoricalDueling => new CategoricalDuelingDQN2D("2DCategoricalDuelingDQN", h, w, actionSize2, _neuronsPerLayer, depth: _depth, numAtoms, noisyLayers, noiseScale),
                _ => throw new ArgumentOutOfRangeException(nameof(_networkType), _networkType, null)
            };
        }
        
        throw new Exception("Unexpected type");
    }
}

public interface IDQNNetProvider
{
    DQNNET CreateCriticNet<TState>(EnvironmentSizeDTO env, bool noisyLayers = false, float noiseScale = 0f, int numAtoms = 51) where TState : notnull;
}