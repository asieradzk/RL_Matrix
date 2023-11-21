using System;
using System.Collections.Generic;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using RLMatrix.Memories;

namespace RLMatrix
{
    public class TransitionReplayMemory<TState> : IMemory<TState>, IStorableMemory
    {
        private readonly int capacity;
        private readonly int batchSize;
        public List<Transition<TState>> memory;
        private readonly Random random = new Random();
        public int myCount => memory.Count;

        /// <summary>
        /// Initializes a new instance of the DQNReplayMemory class.
        /// </summary>
        /// <param name="capacity">The maximum number of transitions the memory can hold.</param>
        /// <param name="batchSize">The number of transitions to be returned when sampling.</param>
        public TransitionReplayMemory(int capacity, int batchSize)
        {
            if (typeof(TState) != typeof(float[]) && typeof(TState) != typeof(float[,]))
            {
                throw new ArgumentException("TState must be either float[] or float[,]");
            }

            this.capacity = capacity;
            this.batchSize = batchSize;
            memory = new List<Transition<TState>>(capacity);
        }

        /// <summary>
        /// Adds a new transition to the memory. 
        /// If the memory capacity is reached, the oldest transition is removed.
        /// </summary>
        /// <param name="transition">The transition to be added.</param>
        public void Push(Transition<TState> transition)
        {
            if (memory.Count >= capacity)
            {
                memory.RemoveAt(0); // Remove oldest if capacity is reached
            }

            memory.Add(transition);
        }

        /// <summary>
        /// Samples a batch of transitions randomly from the memory.
        /// </summary>
        /// <returns>A list of randomly sampled transitions.</returns>
        public List<Transition<TState>> Sample()
        {
            if (batchSize > memory.Count)
            {
                throw new InvalidOperationException("Batch size cannot be greater than current memory size.");
            }

            return Enumerable.Range(0, batchSize)
                             .Select(_ => memory[random.Next(memory.Count)])
                             .ToList();
        }
        public List<Transition<TState>> Sample(int sampleSize)
        {
            if (sampleSize > memory.Count)
            {
                throw new InvalidOperationException("Batch size cannot be greater than current memory size.");
            }

            return Enumerable.Range(0, sampleSize)
                             .Select(_ => memory[random.Next(memory.Count)])
                             .ToList();
        }

        /// <summary>
        /// Serializes the DQNReplayMemory and saves it to a file.
        /// </summary>
        /// <param name="pathToFile">The path where to save the serialized memory.</param>
        public void Save(string pathToFile)
        {
            using var fs = new FileStream(pathToFile, FileMode.Create);
            var bf = new BinaryFormatter();
            bf.Serialize(fs, memory);
        }

        /// <summary>
        /// Loads DQNReplayMemory from a file and deserializes it.
        /// </summary>
        /// <param name="pathToFile">The path from where to load the serialized memory.</param>
        public void Load(string pathToFile)
        {
            if (!File.Exists(pathToFile))
                throw new FileNotFoundException($"File {pathToFile} does not exist.");

            using var fs = new FileStream(pathToFile, FileMode.Open);
            var bf = new BinaryFormatter();
            memory = (List<Transition<TState>>)bf.Deserialize(fs);
        }
    }
}
