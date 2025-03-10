using RLMatrix.Common;

namespace RLMatrix;

/// <summary>
///     Defines an interface for objects that have a memory.
/// </summary>
/// <typeparam name="TState">The type of state stored in the memory.</typeparam>
public interface IHasMemory<TState>
    where TState : notnull
{
    /// <summary>
    ///     Gets or sets the memory of the object.
    /// </summary>
    IMemory<TState> Memory { get; }

    /// <summary>
    ///     Adds transitions to the memory.
    /// </summary>
    /// <param name="transitions">The transitions to add.</param>
    ValueTask AddTransitionsAsync(IEnumerable<Transition<TState>> transitions);
}