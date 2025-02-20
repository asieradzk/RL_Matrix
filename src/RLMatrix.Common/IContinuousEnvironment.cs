namespace RLMatrix.Common;

/// <summary>
///     Defines the structure of a reinforcement learning environment that uses both discrete and continuous actions.
/// </summary>
/// <typeparam name="TState">The type that describes the state of the environment.</typeparam>
public interface IContinuousEnvironment<TState> : IEnvironment<TState>
    where TState : notnull
{
    /// <summary>
    ///     The dimensions of the discrete actions.
    ///     <para>For example, a two-element array <c>{2, 5}</c> would result in two actions in the integer ranges <c>[0,1]</c> and <c>[0,4]</c>.</para>
    /// </summary>
    int[] DiscreteActionDimensions { get; }
    
    /// <summary>
    ///     The dimensions of the continuous actions.
    ///     <para>For example, a two-element array <c>{(1,5),(2,3)}</c> would result in two actions in the float ranges <c>[1,5]</c> and <c>[2,3]</c>.</para>
    /// </summary>
    ContinuousActionDimensions[] ContinuousActionDimensions { get; }
}