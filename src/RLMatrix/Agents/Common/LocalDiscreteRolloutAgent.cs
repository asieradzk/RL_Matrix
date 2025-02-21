using RLMatrix.Common;

namespace RLMatrix;

public class LocalDiscreteRolloutAgent<TState>(ICollection<IDiscreteEnvironment<TState>> environments, IAgentOptions options)
    : LocalRolloutAgent<TState, IDiscreteProxy<TState>, IDiscreteEnvironment<TState>>(environments, CreateProxy(environments.First(), options))
    where TState : notnull
{
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