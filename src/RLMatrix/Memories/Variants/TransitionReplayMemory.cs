using System;
using System.Collections.Generic;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using RLMatrix.Memories;

namespace RLMatrix
{
    /// <summary>
    /// The TransitionInMemoryReplayMemory class represents the memory of the agent in reinforcement learning.
    /// It is used to store and retrieve past experiences (TransitionInMemorys) for algorithms like PPO that require complete episodes.
    /// </summary>
    public class TransitionReplayMemory<TState> : IMemory<TState>, IStorableMemory
    {
        private readonly int capacity;
        private readonly int batchSize;
        private TransitionInMemory<TState>[] memory;
        private int count;
        private int currentIndex;
        private readonly Random random = new Random();

        /// <summary>
        /// Gets the number of TransitionInMemorys currently stored in the memory.
        /// </summary>
        public int Length => count;

        /// <summary>
        /// Initializes a new instance of the TransitionInMemoryReplayMemory class.
        /// </summary>
        /// <param name="capacity">The maximum number of TransitionInMemorys the memory can hold.</param>
        /// <param name="batchSize">The number of TransitionInMemorys to be returned when sampling.</param>
        public TransitionReplayMemory(int capacity, int batchSize)
        {
            if (typeof(TState) != typeof(float[]) && typeof(TState) != typeof(float[,]))
            {
                throw new ArgumentException("TState must be either float[] or float[,]");
            }

            this.capacity = capacity;
            this.batchSize = batchSize;
            memory = new TransitionInMemory<TState>[capacity];
            count = 0;
            currentIndex = 0;
        }

        /// <summary>
        /// Adds a new TransitionInMemory to the memory.
        /// If the memory capacity is reached, the oldest TransitionInMemory is removed.
        /// </summary>
        /// <param name="TransitionInMemory">The TransitionInMemory to be added.</param>
        public void Push(TransitionInMemory<TState> TransitionInMemory)
        {
            memory[currentIndex] = TransitionInMemory;
            currentIndex = (currentIndex + 1) % capacity;

            if (count < capacity)
            {
                count++;
            }
        }

        /// <summary>
        /// Samples a batch of TransitionInMemorys randomly from the memory.
        /// </summary>
        /// <returns>A span of randomly sampled TransitionInMemorys.</returns>
        public ReadOnlySpan<TransitionInMemory<TState>> Sample()
        {
            if (batchSize > count)
            {
                throw new InvalidOperationException("Batch size cannot be greater than current memory size.");
            }

            var indices = Enumerable.Range(0, count)
                                    .OrderBy(x => random.Next())
                                    .Take(batchSize)
                                    .ToArray();

            return new ReadOnlySpan<TransitionInMemory<TState>>(indices.Select(i => memory[i]).ToArray());
        }

        /// <summary>
        /// Samples a specified number of TransitionInMemorys randomly from the memory.
        /// </summary>
        /// <param name="sampleSize">The number of TransitionInMemorys to sample.</param>
        /// <returns>A span of randomly sampled TransitionInMemorys.</returns>
        public ReadOnlySpan<TransitionInMemory<TState>> Sample(int sampleSize)
        {
            if (sampleSize > count)
            {
                throw new InvalidOperationException("Sample size cannot be greater than current memory size.");
            }

            var indices = Enumerable.Range(0, count)
                                    .OrderBy(x => random.Next())
                                    .Take(sampleSize)
                                    .ToArray();

            return new ReadOnlySpan<TransitionInMemory<TState>>(indices.Select(i => memory[i]).ToArray());
        }

        /// <summary>
        /// Serializes the TransitionInMemoryReplayMemory and saves it to a file.
        /// </summary>
        /// <param name="pathToFile">The path where to save the serialized memory.</param>
        public void Save(string pathToFile)
        {
            using var fs = new FileStream(pathToFile, FileMode.Create);
            throw new NotImplementedException();
        }

        /// <summary>
        /// Loads TransitionInMemoryReplayMemory from a file and deserializes it.
        /// </summary>
        /// <param name="pathToFile">The path from where to load the serialized memory.</param>
        public void Load(string pathToFile)
        {
            if (!File.Exists(pathToFile))
                throw new FileNotFoundException($"File {pathToFile} does not exist.");

            using var fs = new FileStream(pathToFile, FileMode.Open);
            throw new NotImplementedException();
        }

        public int FindTransitionInMemoryIndex(TState state)
        {
            throw new NotImplementedException();
        }

        public ref TransitionInMemory<TState> GetTransitionInMemory(int index)
        {
            throw new NotImplementedException();
        }

        public void Push(IEnumerable<TransitionInMemory<TState>> TransitionInMemorys)
        {
            throw new NotImplementedException();
        }
    }
}