using RLMatrix;
using RLMatrix.Agents.Common;
using RLMatrix.Dashboard;
using RLMatrix.Memories;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;

/// <summary>
/// The PrioritizedReplayMemory class represents a prioritized memory of the agent in reinforcement learning.
/// It is used to store and retrieve past experiences (transitions) based on their priority.
/// </summary>
public class PrioritizedReplayMemory<TState> : IMemory<TState>, IStorableMemory
{
    private readonly int capacity;
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

    public int NumEpisodes { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

    /// <summary>
    /// Initializes a new instance of the PrioritizedReplayMemory class.
    /// </summary>
    /// <param name="capacity">The maximum number of transitions the memory can hold.</param>
    public PrioritizedReplayMemory(int capacity)
    {
        this.capacity = capacity;
        this.sumTree = new SumTree(capacity);
        this.memory = new TransitionInMemory<TState>[capacity];
        this.count = 0;
        this.currentIndex = 0;
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
    /// Adds multiple transitions to the memory with a priority.
    /// </summary>
    /// <param name="transitions">The transitions to be added.</param>
    public void Push(IList<TransitionInMemory<TState>> transitions)
    {
        int transitionsCount = transitions.Count;
        if (transitionsCount > capacity)
        {
            throw new ArgumentException("Number of transitions exceeds the memory capacity.");
        }

        float priority = sumTree.MaxPriority;
        if (count + transitionsCount <= capacity)
        {
            for (int i = 0; i < transitionsCount; i++)
            {
                memory[currentIndex] = transitions[i];
                sumTree.Add(priority);
                currentIndex = (currentIndex + 1) % capacity;
            }
            count += transitionsCount;
        }
        else
        {
            int remainingCapacity = capacity - count;
            for (int i = 0; i < remainingCapacity; i++)
            {
                memory[currentIndex] = transitions[i];
                sumTree.Add(priority);
                currentIndex = (currentIndex + 1) % capacity;
            }
            for (int i = remainingCapacity; i < transitionsCount; i++)
            {
                memory[i - remainingCapacity] = transitions[i];
                sumTree.Add(priority);
            }
            currentIndex = transitionsCount - remainingCapacity;
            count = capacity;
        }

        ProcessAndUploadEpisodes(transitions);
    }

    //SOLID violation your mother tried to warn you about.
    public void ProcessAndUploadEpisodes(IList<TransitionInMemory<TState>> transitions)
    {
        var firstTransitions = transitions.Where(t => t.previousTransition == null).ToList();

        foreach (var firstTransition in firstTransitions)
        {
            double episodeReward = 0;
            int episodeLength = 0;
            var currentTransition = firstTransition;

            while (currentTransition != null)
            {
                episodeReward += currentTransition.reward;
                episodeLength++;
                currentTransition = currentTransition.nextTransition;
            }
            DashboardProvider.Instance.UpdateEpisodeData(episodeReward, episodeReward, episodeLength);

        }
    }

    /// <summary>
    /// Samples the entire memory.
    /// </summary>
    /// <returns>An IList of all transitions in the memory.</returns>
    public IList<TransitionInMemory<TState>> SampleEntireMemory()
    {
        return memory.Take(count).ToList();
    }

    /// <summary>
    /// Samples a specified number of transitions based on their priority.
    /// </summary>
    /// <param name="batchSize">The number of transitions to sample.</param>
    /// <returns>An IList of sampled transitions.</returns>
    public IList<TransitionInMemory<TState>> Sample(int batchSize)
    {
        if (batchSize > count)
        {
            throw new InvalidOperationException("Batch size cannot be greater than current memory size.");
        }

        lastSampledIndices = new int[batchSize];

        float segment = sumTree.TotalSum / batchSize;
        for (int i = 0; i < batchSize; i++)
        {
            float a = segment * i;
            float b = segment * (i + 1);
            float value = (float)(random.NextDouble() * (b - a) + a);
            lastSampledIndices[i] = sumTree.Retrieve(value);
        }

        return lastSampledIndices.Select(i => memory[i]).ToList();
    }

    /// <summary>
    /// Gets the indices of the last sampled transitions.
    /// </summary>
    /// <returns>An array of the last sampled indices.</returns>
    public int[] GetSampledIndices()
    {
        return lastSampledIndices;
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
}