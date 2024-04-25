using RLMatrix;
using RLMatrix.Memories;
using System;
using System.Runtime.Serialization.Formatters.Binary;

/// <summary>
/// The PrioritizedReplayMemory class represents a prioritized memory of the agent in reinforcement learning.
/// It is used to store and retrieve past experiences (transitions) based on their priority.
/// </summary>
public class PrioritizedReplayMemory<TState> : IMemory<TState>, IStorableMemory
{
    private readonly int capacity;
    private readonly int batchSize;
    private readonly Random random = new Random();
    private readonly SumTree sumTree;
    private TransitionInMemory<TState>[] memory;
    private int count;
    private int currentIndex;
    private int[] lastSampledIndices;

    /// <summary>
    /// Gets the number of transitions currently stored in the memory.
    /// </summary>
    public int Length => count;

    /// <summary>
    /// Initializes a new instance of the PrioritizedReplayMemory class.
    /// </summary>
    /// <param name="capacity">The maximum number of transitions the memory can hold.</param>
    /// <param name="batchSize">The number of transitions to be returned when sampling.</param>
    public PrioritizedReplayMemory(int capacity, int batchSize)
    {
        this.capacity = capacity;
        this.batchSize = batchSize;
        this.sumTree = new SumTree(capacity);
        this.memory = new TransitionInMemory<TState>[capacity];
        this.count = 0;
        this.currentIndex = 0;
        this.lastSampledIndices = new int[batchSize];
    }

    /// <summary>
    /// Adds a new transition to the memory with a priority.
    /// If the memory capacity is reached, the oldest transition is removed.
    /// </summary>
    /// <param name="transition">The transition to be added.</param>
    public void Push(TransitionInMemory<TState> transition)
    {
        float priority = sumTree.MaxPriority;
        memory[currentIndex] = transition;
        sumTree.Add(priority);

        currentIndex = (currentIndex + 1) % capacity;

        if (count < capacity)
        {
            count++;
        }
    }

    /// <summary>
    /// Samples a batch of transitions based on their priority.
    /// </summary>
    /// <returns>A span of sampled transitions.</returns>
    public ReadOnlySpan<TransitionInMemory<TState>> Sample()
    {
        if (batchSize > count)
        {
            throw new InvalidOperationException("Batch size cannot be greater than current memory size.");
        }

        float segment = sumTree.TotalSum / batchSize;
        for (int i = 0; i < batchSize; i++)
        {
            float a = segment * i;
            float b = segment * (i + 1);
            float value = (float)(random.NextDouble() * (b - a) + a);
            lastSampledIndices[i] = sumTree.Retrieve(value);
        }

        return new ReadOnlySpan<TransitionInMemory<TState>>(lastSampledIndices.Select(i => memory[i]).ToArray());
    }

    /// <summary>
    /// Gets the indices of the last sampled transitions.
    /// </summary>
    /// <returns>A span of the last sampled indices.</returns>
    public Span<int> GetSampledIndices()
    {
        return new Span<int>(lastSampledIndices);
    }

    /// <summary>
    /// Updates the priority of a transition at the specified index.
    /// </summary>
    /// <param name="index">The index of the transition.</param>
    /// <param name="priority">The new priority value.</param>
    public void UpdatePriority(int index, float priority)
    {
        if (index >= 0 && index < count)
        {
            sumTree.Update(index, priority);
        }
        else
        {
            throw new ArgumentOutOfRangeException(nameof(index), "Index is out of range for priority update.");
        }
    }

    /// <summary>
    /// Clears all transitions from the memory.
    /// </summary>
    public void ClearMemory()
    {
        count = 0;
        currentIndex = 0;
        sumTree.Clear();
    }

    /// <summary>
    /// Serializes the PrioritizedReplayMemory and saves it to a file.
    /// </summary>
    /// <param name="pathToFile">The path where to save the serialized memory.</param>
    public void Save(string pathToFile)
    {
        throw new NotImplementedException();
    }

    /// <summary>
    /// Loads PrioritizedReplayMemory from a file and deserializes it.
    /// </summary>
    /// <param name="pathToFile">The path from where to load the serialized memory.</param>
    public void Load(string pathToFile)
    {
        if (!File.Exists(pathToFile))
            throw new FileNotFoundException($"File {pathToFile} does not exist.");

        throw new NotImplementedException();
    }

    /// <summary>
    /// Samples a specified number of transitions based on their priority.
    /// </summary>
    /// <param name="sampleSize">The number of transitions to sample.</param>
    /// <returns>A span of sampled transitions.</returns>
    public ReadOnlySpan<TransitionInMemory<TState>> Sample(int sampleSize)
    {
        if (sampleSize > count)
        {
            throw new InvalidOperationException("Sample size cannot be greater than current memory size.");
        }

        var indices = new int[sampleSize];
        float segment = sumTree.TotalSum / sampleSize;
        for (int i = 0; i < sampleSize; i++)
        {
            float a = segment * i;
            float b = segment * (i + 1);
            float value = (float)(random.NextDouble() * (b - a) + a);
            indices[i] = sumTree.Retrieve(value);
        }

        return new ReadOnlySpan<TransitionInMemory<TState>>(indices.Select(i => memory[i]).ToArray());
    }

    public unsafe int FindTransitionIndex(TState state)
    {
        fixed (TransitionInMemory<TState>* memoryPtr = memory)
        {
            for (int i = 0; i < count; i++)
            {
                if (Equals(memoryPtr[i].nextState, state))
                {
                    return i;
                }
            }
        }
        Console.WriteLine("Transition not found.");
        return -1;
    }

    public ref TransitionInMemory<TState> GetTransition(int index)
    {
        if (index < 0 || index >= count)
        {
            throw new IndexOutOfRangeException("Index is out of range.");
        }
        return ref memory[index];
    }

    public void Push(IEnumerable<TransitionInMemory<TState>> transitions)
    {
        foreach (var transition in transitions)
        {
            Push(transition);
        }
    }

}