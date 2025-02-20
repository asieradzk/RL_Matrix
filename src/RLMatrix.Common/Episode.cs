namespace RLMatrix.Common;

/// <summary>
///     Represents a single episode (or sequence of actions and transitions) between the reinforcement learning algorithm and its environment.
/// </summary>
/// <typeparam name="TState">The type that describes the state of the environment.</typeparam>
public sealed class Episode<TState>
    where TState : notnull
{
    private Guid? _currentGuid;

    /// <summary>
    ///     The cumulative (total) reward of this episode.
    /// </summary>
    public float CumulativeReward { get; private set; }

    /// <summary>
    ///     All previously completed <see cref="Transition{TState}"/>s of this episode.
    /// </summary>
    public List<Transition<TState>> CompletedTransitions { get; } = [];

    public void AddTransition(TState state, bool isDone, RLActions actions, float reward = 1f)
    {
        _currentGuid ??= Guid.NewGuid();

        Guid? nextGuid = null;
        if (!isDone)
        {
            nextGuid = Guid.NewGuid();
        }

        var transition = new Transition<TState>(_currentGuid.Value, state, actions, reward, nextGuid);

        CumulativeReward += reward;
        _currentGuid = nextGuid;

        if (isDone)
        {
            CompletedTransitions.Add(transition);
        }
    }
}