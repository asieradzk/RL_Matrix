using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;


namespace RLMatrix
{
    /// <summary>
    /// The ReplayMemory class represents the memory of the agent in reinforcement learning.
    /// It is used to store and retrieve past experiences (transitions).
    /// </summary>
    public class ReplayMemory<TState>
    {
        private readonly int capacity;
        private readonly int batchSize;
        private List<Transition<TState>> memory;
        private readonly Random random = new Random();

        /// <summary>
        /// Gets the number of transitions currently stored in the memory.
        /// </summary>
        public int Length => memory.Count;

        /// <summary>
        /// Initializes a new instance of the ReplayMemory class.
        /// </summary>
        /// <param name="capacity">The maximum number of transitions the memory can hold.</param>
        /// <param name="batchSize">The number of transitions to be returned when sampling.</param>
        public ReplayMemory(int capacity, int batchSize)
        {
            if (typeof(TState) != typeof(float[]) && typeof(TState) != typeof(float[,]))
            {
                throw new ArgumentException("TState must be either float[] or float[,]");
            }

            this.batchSize = batchSize;
            this.capacity = capacity;
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

        /// <summary>
        /// Samples episodes randomly from the memory.
        /// An episode is a sequence of transitions ending with a terminal state.
        /// </summary>
        /// <returns>A list of transitions representing one or more episodes.</returns>
        public List<Transition<TState>> SampleEpisodes()
        {
            if (batchSize > memory.Count)
            {
                throw new InvalidOperationException("Batch size cannot be greater than current memory size.");
            }

            var episodes = SplitIntoEpisodes();
            var sampledItems = new List<Transition<TState>>(batchSize);

            while (sampledItems.Count < batchSize && episodes.Count > 0)
            {
                // Pick a random episode
                int index = random.Next(episodes.Count);
                var selectedEpisode = episodes[index];
                episodes.RemoveAt(index);

                // Add transitions from the selected episode to the sampled items
                sampledItems.AddRange(selectedEpisode);
            }

            return sampledItems;
        }

        /// <summary>
        /// Splits the memory into separate episodes.
        /// </summary>
        /// <returns>A list of episodes, each represented as a list of transitions.</returns>
        private List<List<Transition<TState>>> SplitIntoEpisodes()
        {
            var episodes = new List<List<Transition<TState>>>();
            var currentEpisode = new List<Transition<TState>>();

            foreach (var transition in memory)
            {
                if (transition.nextState != null)
                {
                    currentEpisode.Add(transition);
                }
                else
                {
                    if (currentEpisode.Any())
                    {
                        episodes.Add(currentEpisode);
                        currentEpisode = new List<Transition<TState>>();
                    }
                }
            }

            // Add the last episode if it wasn't added inside the loop
            if (currentEpisode.Any())
            {
                episodes.Add(currentEpisode);
            }

            return episodes;
        }

        /// <summary>
        /// Returns all transitions currently stored in the memory.
        /// </summary>
        /// <returns>A list of all transitions in the memory.</returns>
        public List<Transition<TState>> SampleEntireMemory()
        {
            return new List<Transition<TState>>(memory);
        }

        /// <summary>
        /// Clears all transitions from the memory.
        /// </summary>
        public void ClearMemory()
        {
            memory.Clear();
        }

        /// <summary>
        /// Serializes the ReplayMemory and saves it to a file.
        /// </summary>
        /// <param name="pathToFile">The path where to save the serialized memory.</param>
        public void Save(string pathToFile)
        {
            using var fs = new FileStream(pathToFile, FileMode.Create);
            var bf = new BinaryFormatter();
            bf.Serialize(fs, memory);
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
            var bf = new BinaryFormatter();
            memory = (List<Transition<TState>>)bf.Deserialize(fs);
        }

    }
}
