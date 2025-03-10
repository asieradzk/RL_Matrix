namespace RLMatrix;

/// <summary>
///     The PrioritizedReplayMemory class represents a prioritized memory of the agent in reinforcement learning.
///     It is used to store and retrieve past experiences (transitions) based on their priority.
/// </summary>
public class PrioritizedReplayMemory<TState> : IMemory<TState>, IStorableMemory
    where TState : notnull
{
    private readonly Random _random = new();
    private readonly int _capacity;
    private readonly SumTree _sumTree;
    private readonly MemoryTransition<TState>[] _memory;
    private int _currentIndex;
    private int[] _lastSampledIndices = [];
    
    /// <summary>
    ///     Initializes a new instance of the PrioritizedReplayMemory class.
    /// </summary>
    /// <param name="capacity">The maximum number of transitions the memory can hold.</param>
    public PrioritizedReplayMemory(int capacity)
    {
        _capacity = capacity;
        _sumTree = new SumTree(capacity);
        _memory = new MemoryTransition<TState>[capacity];
    }

    /// <summary>
    ///     Gets the number of transitions currently stored in the memory.
    /// </summary>
    public int Length { get; private set; }

    public int EpisodeCount { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

    /// <summary>
    ///     Adds a new transition to the memory with a priority.
    ///     If the memory capacity is reached, the oldest transition is removed.
    /// </summary>
    /// <param name="transition">The transition to be added.</param>
    public ValueTask PushAsync(MemoryTransition<TState> transition)
    {
        var priority = _sumTree.MaxPriority;
        _memory[_currentIndex] = transition;
        _sumTree.Add(priority);

        _currentIndex = (_currentIndex + 1) % _capacity;

        if (Length < _capacity)
        {
            Length++;
        }

        return new();
    }

    /// <summary>
    ///     Adds multiple transitions to the memory with a priority.
    /// </summary>
    /// <param name="transitions">The transitions to be added.</param>
    public async ValueTask PushAsync(IList<MemoryTransition<TState>> transitions)
    {
        var transitionsCount = transitions.Count;
        if (transitionsCount > _capacity)
        {
            throw new ArgumentException("Number of transitions exceeds the memory capacity.");
        }

        var priority = _sumTree.MaxPriority;
        if (Length + transitionsCount <= _capacity)
        {
            for (var i = 0; i < transitionsCount; i++)
            {
                _memory[_currentIndex] = transitions[i];
                _sumTree.Add(priority);
                _currentIndex = (_currentIndex + 1) % _capacity;
            }
            Length += transitionsCount;
        }
        else
        {
            var remainingCapacity = _capacity - Length;
            for (var i = 0; i < remainingCapacity; i++)
            {
                _memory[_currentIndex] = transitions[i];
                _sumTree.Add(priority);
                _currentIndex = (_currentIndex + 1) % _capacity;
            }
            
            for (var i = remainingCapacity; i < transitionsCount; i++)
            {
                _memory[i - remainingCapacity] = transitions[i];
                _sumTree.Add(priority);
            }
            
            _currentIndex = transitionsCount - remainingCapacity;
            Length = _capacity;
        }

        await ProcessAndUploadEpisodesAsync(transitions);
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
    /// <returns>A list containing all transitions in the memory.</returns>
    public IList<MemoryTransition<TState>> SampleEntireMemory()
    {
        return _memory.Take(Length).ToList();
    }

    /// <summary>
    ///     Samples a specified number of transitions based on their priority.
    /// </summary>
    /// <param name="batchSize">The number of transitions to sample.</param>
    /// <returns>An IList of sampled transitions.</returns>
    public IList<MemoryTransition<TState>> Sample(int batchSize)
    {
        if (batchSize > Length)
        {
            throw new InvalidOperationException("Batch size cannot be greater than current memory size.");
        }

        _lastSampledIndices = new int[batchSize];

        var segment = _sumTree.TotalSum / batchSize;
        for (var i = 0; i < batchSize; i++)
        {
            var a = segment * i;
            var b = segment * (i + 1);
            var value = (float)(_random.NextDouble() * (b - a) + a);
            _lastSampledIndices[i] = _sumTree.Retrieve(value);
        }

        return _lastSampledIndices.Select(i => _memory[i]).ToList();
    }

    /// <summary>
    ///     Gets the indices of the last sampled transitions.
    /// </summary>
    /// <returns>An array of the last sampled indices.</returns>
    public int[] GetSampledIndices()
    {
        return _lastSampledIndices;
    }

    /// <summary>
    ///     Updates the priority of a transition at the specified index.
    /// </summary>
    /// <param name="index">The index of the transition.</param>
    /// <param name="priority">The new priority value.</param>
    public void UpdatePriority(int index, float priority)
    {
        if (index >= 0 && index < Length)
        {
            _sumTree.Update(index, priority);
        }
        else
        {
            throw new ArgumentOutOfRangeException(nameof(index), "Index is out of range for priority update.");
        }
    }

    /// <summary>
    ///     Clears all transitions from the memory.
    /// </summary>
    public void ClearMemory()
    {
        Length = 0;
        _currentIndex = 0;
        _sumTree.Clear();
    }

    /// <summary>
    ///     Serializes the PrioritizedReplayMemory and saves it to a file.
    /// </summary>
    /// <param name="path">The path where to save the serialized memory.</param>
    public void Save(string path)
    {
        throw new NotImplementedException();
    }

    /// <summary>
    ///     Loads PrioritizedReplayMemory from a file and deserializes it.
    /// </summary>
    /// <param name="path">The path from where to load the serialized memory.</param>
    public void Load(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"File {path} does not exist.");

        throw new NotImplementedException();
    }

    // TODO: what's the point of `unsafe` here? preventing from something else modifying _memory while we find the index?
    public unsafe int FindTransitionIndex(TState state)
    {
        fixed (MemoryTransition<TState>* memoryPtr = _memory)
        {
            for (var i = 0; i < Length; i++)
            {
                if (Equals(memoryPtr[i].NextState, state))
                {
                    return i;
                }
            }
        }
        
        Console.WriteLine("Transition not found.");
        return -1;
    }

    public ref MemoryTransition<TState> GetTransition(int index)
    {
        if (index < 0 || index >= Length)
        {
            throw new IndexOutOfRangeException("Index is out of range.");
        }
        
        return ref _memory[index];
    }
}