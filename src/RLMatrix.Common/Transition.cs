namespace RLMatrix.Common;

/// <summary>
/// Represents a transition between two steps in an <see cref="Episode{TState}"/>.
/// </summary>
/// <param name="Id">The ID of this transition.</param>
/// <param name="State">The current state.</param>
/// <param name="Actions">The actions that were taken.</param>
/// <param name="Reward">The reward from one step to another.</param>
/// <param name="NextTransitionId">The ID of the next <see cref="Transition{TState}"/>, if it exists.</param>
/// <typeparam name="TState">The type that describes the state of the environment for this transition.</typeparam>
public sealed record Transition<TState>(Guid Id, TState State, RLActions Actions, float Reward, Guid? NextTransitionId) where TState : notnull;