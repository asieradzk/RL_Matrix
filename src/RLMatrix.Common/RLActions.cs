namespace RLMatrix.Common;

/// <summary>
///     Represents the decided actions taken by a reinforcement learning algorithm.
/// </summary>
public sealed record RLActions
{
    private readonly float[]? _continuousActions;
    
    private RLActions(int[] discreteActions, float[]? continuousActions = null)
    {
        if (discreteActions.Length == 0)
            throw new ArgumentException("Discrete actions must have at least one element.");
        
        DiscreteActions = discreteActions;

        if (_continuousActions is { Length: 0 })
            throw new ArgumentException("Continuous actions must have at least one element if an array is provided.");
        
        _continuousActions = continuousActions;
    }

    /// <summary>
    ///     The discrete actions taken.
    /// </summary>
    public int[] DiscreteActions { get; }
    
    /// <summary>
    ///     The continuous actions taken.
    /// </summary>
    /// <exception cref="InvalidOperationException">Attempting to access continuous actions in a discrete context.</exception>
    public float[] ContinuousActions => _continuousActions ?? throw new InvalidOperationException("Continuous actions access in discrete context.");
    
    /// <summary>
    ///     Whether these actions originated from a continuous context (i.e., contain both discrete and continuous actions).
    /// </summary>
    public bool IsContinuous => _continuousActions is not null;
    
    public static RLActions Discrete(int[] discreteActions) => new(discreteActions);
    public static RLActions Continuous(int[] discreteActions, float[] continuousActions) => new(discreteActions, continuousActions);
    public static RLActions Interpret(int[] discreteActions, float[]? continuousActions)
    {
        // We assume that an empty or null continuous action array means it's discrete.
        return continuousActions is { Length: > 0 }
            ? Continuous(discreteActions, continuousActions)
            : Discrete(discreteActions);
    }

    // TODO: re-enable these after
    //public static implicit operator RLActions(int[] discreteActions) => Discrete(discreteActions);
    //public static implicit operator RLActions((int[] DiscreteActions, float[] ContinuousActions) tuple) => Continuous(tuple.DiscreteActions, tuple.ContinuousActions);
}