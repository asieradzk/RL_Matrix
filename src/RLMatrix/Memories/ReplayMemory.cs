using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using RLMatrix.Memories;

namespace RLMatrix
{
    /// <summary>
    /// The ReplayMemory class represents the memory of the agent in reinforcement learning.
    /// It is used to store and retrieve past experiences (TransitionInMemorys).
    /// </summary>
    public class ReplayMemory<TState> : IMemory<TState>, IStorableMemory
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
        /// Initializes a new instance of the ReplayMemory class.
        /// </summary>
        /// <param name="capacity">The maximum number of TransitionInMemorys the memory can hold.</param>
        /// <param name="batchSize">The number of TransitionInMemorys to be returned when sampling.</param>
        public ReplayMemory(int capacity, int batchSize)
        {
            if (typeof(TState) != typeof(float[]) && typeof(TState) != typeof(float[,]))
            {
                throw new ArgumentException("TState must be either float[] or float[,]");
            }

            this.batchSize = batchSize;
            this.capacity = capacity;
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

            int[] indices = GetShuffledIndices(count, batchSize);
            TransitionInMemory<TState>[] sampledTransitionInMemorys = new TransitionInMemory<TState>[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                sampledTransitionInMemorys[i] = memory[indices[i]];
            }

            return new ReadOnlySpan<TransitionInMemory<TState>>(sampledTransitionInMemorys);
        }


        /// <summary>
        /// Clears all TransitionInMemorys from the memory.
        /// </summary>
        public void ClearMemory()
        {
            count = 0;
            currentIndex = 0;
        }

        /// <summary>
        /// Serializes the ReplayMemory and saves it to a file.
        /// </summary>
        /// <param name="pathToFile">The path where to save the serialized memory.</param>
        public void Save(string pathToFile)
        {
            using var fs = new FileStream(pathToFile, FileMode.Create);
            throw new NotImplementedException();
        }

        /// <summary>
        /// Loads ReplayMemory from a file and deserializes it.
        /// </summary>
        /// <param name="pathToFile">The path from where to load the serialized memory.</param>
        public void Load(string pathToFile)
        {
            if (!File.Exists(pathToFile))
                throw new FileNotFoundException($"File {pathToFile} does not exist.");

            using var fs = new FileStream(pathToFile, FileMode.Open);
           throw new NotImplementedException();
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

            int[] indices = GetShuffledIndices(count, sampleSize);
            return new ReadOnlySpan<TransitionInMemory<TState>>(memory, indices[0], sampleSize);
        }

        private int[] GetShuffledIndices(int count, int size)
        {
            int[] indices = new int[size];
            for (int i = 0; i < size; i++)
            {
                indices[i] = random.Next(count);
            }
            return indices;
        }


        private int searchDepth;
        public unsafe int FindTransitionInMemoryIndex(TState state)
        {
            fixed (TransitionInMemory<TState>* memoryPtr = memory)
            {
                for (int i = 0; i < count; i++)
                {
                    if (Equals(memoryPtr[i].state, state))
                    {
                        return i;
                    }
                }
            }
            return -1;
        }
        public ref TransitionInMemory<TState> GetTransitionInMemory(int index)
        {
            if (index < 0 || index >= count)
            {
                throw new IndexOutOfRangeException("Index is out of range.");
            }
            return ref memory[index];
        }

        public void Push(IEnumerable<TransitionInMemory<TState>> TransitionInMemorys)
        {
            foreach (var TransitionInMemory in TransitionInMemorys)
            {
                Push(TransitionInMemory);
            }
        }
    }
}
