namespace RLMatrix;

/// <summary>
///     Default implementation of PPONetProviderBase, makes network with 3 hidden layers
/// </summary>
/// <typeparam name="TState">Type of observation space (1D or 2D). For 2D a simple convoluted network is created.</typeparam>
public class PPONetProviderBase<TState> : IPPONetProvider
    where TState : notnull
{
    private readonly int _neuronsPerLayer;
    private readonly int _depth;
    private readonly bool _useRNN;

    /// <summary>
    ///     Default constructor for simple PPO NNs - critic and actor.
    /// </summary>
    /// <param name="neuronsPerLayer">Number of neurons in hidden layers.</param>
    /// <param name="depth">Depth of hidden layers.</param>
    /// <param name="useRNN">Whether to use a Recurrent Neural Network.</param>
    public PPONetProviderBase(int neuronsPerLayer = 256, int depth = 2, bool useRNN = false)
    {
        _neuronsPerLayer = neuronsPerLayer;
        _depth = depth;
        _useRNN = useRNN;
    }

    public PPOActorNet CreateActorNet(EnvironmentSizeDTO env)
    {
        if (typeof(TState) == typeof(float[]))
        {
            if (env.StateDimensions is not { Dimensions: var dimensions, Dimensions.Length: 1 })
                throw new Exception("Unexpected observation dimension for 1D state.");
            
            return new PPOActorNet1D("1DDQN", dimensions[0], _neuronsPerLayer, env.DiscreteActionDimensions, 
                (env as ContinuousEnvironmentSizeDTO)?.ContinuousActionDimensions ?? [],  _depth, _useRNN);
        }

        if (typeof(TState) == typeof(float[,]))
        {
            if (env.StateDimensions is not { Dimensions: var dimensions, Dimensions.Length: 2 })
                throw new Exception("Unexpected observation dimension for 2D state.");
            
            return new PPOActorNet2D("2DDQN", dimensions[0], dimensions[1], env.DiscreteActionDimensions, 
                (env as ContinuousEnvironmentSizeDTO)?.ContinuousActionDimensions ?? [], _neuronsPerLayer, _depth, _useRNN);
        }
        
        throw new Exception("Unexpected type");
    }

    public PPOCriticNet CreateCriticNet(EnvironmentSizeDTO env)
    {
        if (typeof(TState) == typeof(float[]))
        {
            if (env.StateDimensions is not { Dimensions: var dimensions, Dimensions.Length: 1 })
                throw new Exception("Unexpected observation dimension for 1D state.");
            
            return new PPOCriticNet1D("1DDQN", dimensions[0], _neuronsPerLayer, _depth, _useRNN);
        }

        if (typeof(TState) == typeof(float[,]))
        {
            if (env.StateDimensions is not { Dimensions: var dimensions, Dimensions.Length: 2 })
                throw new Exception("Unexpected observation dimension for 2D state.");
            
            // TODO: this used to be "1DDQN", I think this was a copy-paste bug in the original impl.
            return new PPOCriticNet2D("2DDQN", dimensions[0], dimensions[1], _neuronsPerLayer, _depth, _useRNN);
        }
        
        throw new Exception("Unexpected type");
    }
}