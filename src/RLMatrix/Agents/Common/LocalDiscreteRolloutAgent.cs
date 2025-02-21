using RLMatrix.Common;

namespace RLMatrix;

public class LocalDiscreteRolloutAgent<TState> : LocalRolloutAgent<TState, IDiscreteProxy<TState>, IDiscreteEnvironment<TState>>
    where TState : notnull
{
    public LocalDiscreteRolloutAgent(ICollection<IDiscreteEnvironment<TState>> environments, IAgentOptions options) 
        : base(environments, CreateProxy(environments.First(), options))
    { }

    private static IDiscreteProxy<TState> CreateProxy(IDiscreteEnvironment<TState> environment, IAgentOptions options)
    {
        return options switch
        {
            DQNAgentOptions dqn => new LocalDiscreteQAgent<TState>(dqn, environment.ActionDimensions, environment.StateDimensions),
            PPOAgentOptions ppo => new LocalDiscretePPOAgent<TState>(ppo, environment.ActionDimensions, environment.StateDimensions),
            _ => throw new NotImplementedException($"Only {nameof(DQNAgentOptions)} and {nameof(PPOAgentOptions)} are supported.")
        };
    }
}