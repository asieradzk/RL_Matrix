namespace RLMatrix;

/// <summary>
///     The ReplayMemory class represents the memory of the agent in reinforcement learning.
///     It is used to store and retrieve past experiences (MemoryTransitions).
/// </summary>
public class ReplayMemory<TState> : IMemory<TState>, IStorableMemory
    where TState : notnull
{
    private int _capacity;
    private readonly List<MemoryTransition<TState>> _memory;
    private readonly Random _random = new();
    
    /// <summary>
    ///     Initializes a new instance of the ReplayMemory class.
    /// </summary>
    /// <param name="capacity">The maximum number of TransitionInMemorys the memory can hold.</param>
    public ReplayMemory(int capacity)
    {
        _capacity = capacity;
        _memory = new List<MemoryTransition<TState>>(capacity);
    }

    /// <summary>
    ///     Gets the number of TransitionInMemorys currently stored in the memory.
    /// </summary>
    public int Length => _memory.Count;
    
    public int EpisodeCount => _memory.Count(x => x.NextState == null); // return num of transitions with no next state

    /// <summary>
    ///     Adds a new TransitionInMemory to the memory. 
    ///     If the memory capacity is reached, the oldest TransitionInMemory is removed.
    /// </summary>
    /// <param name="transition">The TransitionInMemory to be added.</param>
    public ValueTask PushAsync(MemoryTransition<TState> transition)
    {
        if (_memory.Count >= _capacity)
        {
            _memory.RemoveAt(0);
        }
        
        _memory.Add(transition);
        return new();
    }

    /// <summary>
    ///     Adds multiple <see cref="MemoryTransition{TState}"/>s to the memory.
    ///     If the memory capacity is exceeded, the capacity is automatically increased.
    /// </summary>
    /// <param name="transitions">The <see cref="MemoryTransition{TState}"/>s to be added.</param>
    public async ValueTask PushAsync(IList<MemoryTransition<TState>> transitions)
    {
        var count = transitions.Count;
        if (_memory.Count + count > _capacity)
        {
            // Increase the capacity to accommodate the new transitions
            _capacity = _memory.Count + count;
        }
        
        await ProcessAndUploadEpisodesAsync(transitions);
        _memory.AddRange(transitions);
    }
    
    // SOLID violation your mother tried to warn you about.
    public async ValueTask ProcessAndUploadEpisodesAsync(IList<MemoryTransition<TState>> transitions)
    {
        var firstTransitions = transitions.Where(t => t.PreviousTransition == null).ToList();

        foreach (var firstTransition in firstTransitions)
        {
            var episodeReward = 0f;
            var episodeLength = 0;
            var currentTransition = firstTransition;

            while (currentTransition != null)
            {
                episodeReward += currentTransition.Reward;
                episodeLength++;
                currentTransition = currentTransition.NextTransition;
            }

            var dashboard = await DashboardProvider.Instance.GetDashboardAsync();
            dashboard.UpdateEpisodeData(episodeReward, episodeReward, episodeLength);
        }
    }

    /// <summary>
    ///     Samples the entire memory.
    /// </summary>
    /// <returns>An IList of all <see cref="MemoryTransition{TState}"/>s in the memory.</returns>
    public IList<MemoryTransition<TState>> SampleEntireMemory()
    {
        return _memory;
    }

    /// <summary>
    ///     Samples a specified number of <see cref="MemoryTransition{TState}"/>s randomly from the memory.
    /// </summary>
    /// <param name="batchSize">The number of <see cref="MemoryTransition{TState}"/>s to sample.</param>
    /// <returns>An IList of randomly sampled <see cref="MemoryTransition{TState}"/>s.</returns>
    public IList<MemoryTransition<TState>> Sample(int batchSize)
    {
        if (batchSize > _memory.Count)
        {
            throw new InvalidOperationException("Batch size cannot be greater than current memory size.");
        }

        return _memory.OrderBy(_ => _random.Next()).Take(batchSize).ToList();
    }

    /// <summary>
    ///     Clears all TransitionInMemorys from the memory.
    /// </summary>
    public void ClearMemory()
    {
        _memory.Clear();
    }

    /// <summary>
    ///     Serializes the ReplayMemory and saves it to a file.
    /// </summary>
    /// <param name="path">The path where to save the serialized memory.</param>
    public void Save(string path)
    {
        using var fs = new FileStream(path, FileMode.Create);
        throw new NotImplementedException();
    }

    /// <summary>
    ///     Loads ReplayMemory from a file and deserializes it.
    /// </summary>
    /// <param name="path">The path from where to load the serialized memory.</param>
    public void Load(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"File {path} does not exist.");

        using var fs = new FileStream(path, FileMode.Open);
        throw new NotImplementedException();
    }
}