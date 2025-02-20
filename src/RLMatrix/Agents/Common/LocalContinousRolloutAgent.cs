using RLMatrix.Common;

namespace RLMatrix;

/// <summary>
///     Represents a local continuous rollout agent for reinforcement learning.
/// </summary>
/// <typeparam name="TState">The type of the state.</typeparam>
public class LocalContinuousRolloutAgent<TState> : LocalRolloutAgent<TState, IContinuousProxy<TState>, IContinuousEnvironment<TState>>
    where TState : notnull
{
    protected LocalContinuousRolloutAgent(ICollection<IContinuousEnvironment<TState>> environments, PPOAgentOptions options) 
        : base(environments, CreateProxy(environments.First(), options))
    { }
    
    private static IContinuousProxy<TState> CreateProxy(IContinuousEnvironment<TState> environment, PPOAgentOptions options)
    {
        return new LocalContinuousPPOAgent<TState>(options, environment.DiscreteActionDimensions, environment.ContinuousActionDimensions, environment.StateDimensions);
    }
}