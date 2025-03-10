namespace RLMatrix;

/// <summary>
///     Defines a discrete agent.
/// </summary>
/// <typeparam name="TState">The type of state the agent operates on.</typeparam>
public interface IDiscreteAgent<TState> : IAgent<TState> where TState : notnull;