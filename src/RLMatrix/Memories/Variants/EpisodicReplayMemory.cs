using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;

namespace RLMatrix.Memories
{
    /// <summary>
    /// The EpisodicReplayMemory class represents the memory of the agent in reinforcement learning,
    /// focusing on storing entire episodes of transitions.
    /// </summary>
    public class EpisodicReplayMemory<TState> : IEpisodicMemory<TState>, IStorableMemory
    {
        private readonly int capacity; // Maximum number of episodes
        private List<List<Transition<TState>>> episodes;
        private readonly Random random = new Random();

        /// <summary>
        /// Gets the number of episodes currently stored in the memory.
        /// </summary>
        public int Length => episodes.Count;

        /// <summary>
        /// Initializes a new instance of the EpisodicReplayMemory class.
        /// </summary>
        /// <param name="capacity">The maximum number of episodes the memory can hold.</param>
        public EpisodicReplayMemory(int capacity)
        {
            this.capacity = capacity;
            episodes = new List<List<Transition<TState>>>(capacity);
        }

        /// <summary>
        /// Adds a new episode to the memory. 
        /// If the memory capacity is reached, the oldest episode is removed.
        /// </summary>
        /// <param name="episode">The episode to be added.</param>
        public void Push(List<Transition<TState>> episode)
        {
            if (episodes.Count >= capacity)
            {
                episodes.RemoveAt(0); // Remove oldest episode if capacity is reached
            }

            episodes.Add(episode);
        }

        public void Push(Transition<TState> transition)
        {
            // This might be an uncommon use case for episodic memory since episodes are preferred
            // but it could be implemented if needed. For now, we'll throw an exception to signal that it's not supported.
            throw new NotSupportedException("Pushing single transitions is not supported for EpisodicReplayMemory. Use Push(episode) instead.");
        }

        /// <summary>
        /// Samples a single episode randomly from the memory.
        /// </summary>
        /// <returns>A list of transitions representing a single episode.</returns>
        public List<Transition<TState>> Sample()
        {
            int index = random.Next(episodes.Count);
            return episodes[index];
        }

        /// <summary>
        /// Serializes the EpisodicReplayMemory and saves it to a file.
        /// </summary>
        /// <param name="pathToFile">The path where to save the serialized memory.</param>
        public void Save(string pathToFile)
        {
            using var fs = new FileStream(pathToFile, FileMode.Create);
            var bf = new BinaryFormatter();
            bf.Serialize(fs, episodes);
        }

        /// <summary>
        /// Loads EpisodicReplayMemory from a file and deserializes it.
        /// </summary>
        /// <param name="pathToFile">The path from where to load the serialized memory.</param>
        public void Load(string pathToFile)
        {
            if (!File.Exists(pathToFile))
                throw new FileNotFoundException($"File {pathToFile} does not exist.");

            using var fs = new FileStream(pathToFile, FileMode.Open);
            var bf = new BinaryFormatter();
            episodes = (List<List<Transition<TState>>>)bf.Deserialize(fs);
        }
    }
}
