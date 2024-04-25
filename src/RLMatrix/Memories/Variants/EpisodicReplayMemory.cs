using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using Tensorboard;

namespace RLMatrix.Memories
{
    /// <summary>
    /// The EpisodicReplayMemory class represents the memory of the agent in reinforcement learning,
    /// focusing on storing entire episodes of TransitionInMemorys.
    /// </summary>
    public class EpisodicReplayMemory<TState> : IEpisodicMemory<TState>, IStorableMemory
    {
        private readonly int capacity; // Maximum number of episodes
        private List<TransitionInMemory<TState>[]> episodes;
        private readonly Random random = new Random();
        public int myCount => episodes.Sum(episode => episode.Length);

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
            episodes = new List<TransitionInMemory<TState>[]>(capacity);
        }

        /// <summary>
        /// Adds a new episode to the memory. 
        /// If the memory capacity is reached, the oldest episode is removed.
        /// </summary>
        /// <param name="episode">The episode to be added.</param>
        public void Push(List<TransitionInMemory<TState>> episode)
        {
            if (episodes.Count >= capacity)
            {
                episodes.RemoveAt(0); // Remove oldest episode if capacity is reached
            }

            episodes.Add(episode.ToArray());
        }

        public void Push(TransitionInMemory<TState> TransitionInMemory)
        {
            // This might be an uncommon use case for episodic memory since episodes are preferred
            // but it could be implemented if needed. For now, we'll throw an exception to signal that it's not supported.
            throw new NotSupportedException("Pushing single TransitionInMemorys is not supported for EpisodicReplayMemory. Use Push(episode) instead.");
        }

        /// <summary>
        /// Clears all TransitionInMemorys from the memory.
        /// </summary>
        public void ClearMemory()
        {
            episodes.Clear();
        }

        /// <summary>
        /// Samples a single episode randomly from the memory.
        /// </summary>
        /// <returns>A span of TransitionInMemorys representing a single episode.</returns>
        public ReadOnlySpan<TransitionInMemory<TState>> Sample()
        {
            int index = random.Next(episodes.Count);
            return episodes[index];
        }

        public ReadOnlySpan<TransitionInMemory<TState>> Sample(int sampleSize)
        {
            // Flatten the list of episodes into a single array of TransitionInMemorys
            TransitionInMemory<TState>[] allTransitionInMemorys = episodes.SelectMany(e => e).ToArray();

            // Check if the sample size is larger than the number of available TransitionInMemorys
            if (sampleSize > allTransitionInMemorys.Length)
            {
                throw new InvalidOperationException("Sample size cannot be greater than the total number of TransitionInMemorys in memory.");
            }

            // Randomly sample TransitionInMemorys
            int[] indices = Enumerable.Range(0, allTransitionInMemorys.Length)
                                      .OrderBy(_ => random.Next())
                                      .Take(sampleSize)
                                      .ToArray();

            TransitionInMemory<TState>[] sampledTransitionInMemorys = new TransitionInMemory<TState>[sampleSize];
            for (int i = 0; i < sampleSize; i++)
            {
                sampledTransitionInMemorys[i] = allTransitionInMemorys[indices[i]];
            }

            return sampledTransitionInMemorys;
        }

        /// <summary>
        /// Samples all episodes from the memory
        /// </summary>
        /// <returns>A span of TransitionInMemorys (in order) of TState type where TState is the observation shape 1d/2d</returns>
        public Span<TransitionInMemory<TState>> SampleEntireMemory()
        {
            return episodes.SelectMany(episode => episode).ToArray();
        }

        /// <summary>
        /// Samples a portion of memory based on the specified start percentage and percentage range.
        /// For instance, arguments of 50, 9 will return 9% of total experiences between 50% and 59% (oldest to most recent).
        /// </summary>
        /// <param name="startPercentage">The starting percentage of experiences to sample</param>
        /// <param name="howManyPercent">The percentage of experiences to sample</param>
        /// <returns>A span of TransitionInMemorys (in order) of TState type where TState is the observation shape 1d/2d</returns>
        public Span<TransitionInMemory<TState>> SamplePortionOfMemory(int startPercentage, int howManyPercent)
        {
            if (startPercentage < 0 || howManyPercent <= 0 || startPercentage + howManyPercent > 100)
                throw new ArgumentException("Invalid argument");

            int totalTransitionInMemorys = myCount;
            int startIndex = (int)(totalTransitionInMemorys * (startPercentage / 100.0));
            int endIndex = (int)(totalTransitionInMemorys * ((startPercentage + howManyPercent) / 100.0));

            TransitionInMemory<TState>[] result = new TransitionInMemory<TState>[endIndex - startIndex];
            int currentIndex = 0;

            foreach (var episode in episodes)
            {
                foreach (var TransitionInMemory in episode)
                {
                    if (currentIndex >= startIndex && currentIndex < endIndex)
                    {
                        result[currentIndex - startIndex] = TransitionInMemory;
                    }
                    if (currentIndex >= endIndex)
                    {
                        break;
                    }
                    currentIndex++;
                }
                if (currentIndex >= endIndex)
                {
                    break;
                }
            }

            return result;
        }

        /// <summary>
        /// Samples a portion of memory based on the cumulative rewards of episodes.
        /// The method ranks episodes by their total reward and selects the top-performing ones.
        /// </summary>
        /// <param name="topPercentage">The top percentage of episodes to sample based on cumulative rewards.</param>
        /// <returns>A span of TransitionInMemorys from the top-performing episodes.</returns>
        public Span<TransitionInMemory<TState>> SamplePortionOfMemoryByRewards(int topPercentage)
        {
            // Dictionary to hold cumulative rewards for each episode
            var episodeRewards = new Dictionary<TransitionInMemory<TState>[], float>();

            // Calculate cumulative rewards for each episode
            foreach (var episode in episodes)
            {
                float cumulativeReward = episode.Sum(TransitionInMemory => TransitionInMemory.reward);
                episodeRewards.Add(episode, cumulativeReward);
            }

            // Sort episodes by their cumulative rewards in descending order
            var sortedEpisodes = episodeRewards.OrderByDescending(pair => pair.Value)
                                                .Select(pair => pair.Key)
                                                .ToArray();

            // Calculate the number of episodes to sample based on the top percentage
            // Ceiling is used to always round up to ensure at least one episode is selected when topPercentage is > 0
            int episodesToSample = (int)Math.Ceiling(sortedEpisodes.Length * (topPercentage / 100.0));

            // Select TransitionInMemorys from the top-performing episodes based on cumulative rewards
            TransitionInMemory<TState>[] result = sortedEpisodes.Take(episodesToSample)
                                                        .SelectMany(e => e)
                                                        .ToArray();

            return result;
        }

        /// <summary>
        /// Serializes the EpisodicReplayMemory and saves it to a file.
        /// </summary>
        /// <param name="pathToFile">The path where to save the serialized memory.</param>
        public void Save(string pathToFile)
        {
            using var fs = new FileStream(pathToFile, FileMode.Create);
            throw new NotImplementedException();
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
/*
namespace RLMatrix.Memories
{
    /// <summary>
    /// The EpisodicReplayMemory class represents the memory of the agent in reinforcement learning,
    /// focusing on storing entire episodes of TransitionInMemorys.
    /// </summary>
    public class EpisodicReplayMemory<TState> : IEpisodicMemory<TState>, IStorableMemory
    {
        private readonly int capacity; // Maximum number of episodes
        public List<List<TransitionInMemory<TState>>> episodes;
        private readonly Random random = new Random();
        public int myCount => episodes.Aggregate(0, (acc, episode) => acc + episode.Count);

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
            episodes = new List<List<TransitionInMemory<TState>>>(capacity);
        }

        /// <summary>
        /// Adds a new episode to the memory. 
        /// If the memory capacity is reached, the oldest episode is removed.
        /// </summary>
        /// <param name="episode">The episode to be added.</param>
        public void Push(List<TransitionInMemory<TState>> episode)
        {
            if (episodes.Count >= capacity)
            {
                episodes.RemoveAt(0); // Remove oldest episode if capacity is reached
            }

            episodes.Add(episode);
        }

        public void Push(TransitionInMemory<TState> TransitionInMemory)
        {
            // This might be an uncommon use case for episodic memory since episodes are preferred
            // but it could be implemented if needed. For now, we'll throw an exception to signal that it's not supported.
            throw new NotSupportedException("Pushing single TransitionInMemorys is not supported for EpisodicReplayMemory. Use Push(episode) instead.");
        }

        /// <summary>
        /// Clears all TransitionInMemorys from the memory.
        /// </summary>
        public void ClearMemory()
        {
            episodes.Clear();
        }

        /// <summary>
        /// Samples a single episode randomly from the memory.
        /// </summary>
        /// <returns>A list of TransitionInMemorys representing a single episode.</returns>
        public List<TransitionInMemory<TState>> Sample()
        {
            int index = random.Next(episodes.Count);
            return episodes[index];
        }
        public List<TransitionInMemory<TState>> Sample(int sampleSize)
        {
            // Flatten the list of episodes into a single list of TransitionInMemorys
            var allTransitionInMemorys = episodes.SelectMany(e => e).ToList();

            // Check if the sample size is larger than the number of available TransitionInMemorys
            if (sampleSize > allTransitionInMemorys.Count)
            {
                throw new InvalidOperationException("Batch size cannot be greater than the total number of TransitionInMemorys in memory.");
            }

            // Randomly sample TransitionInMemorys
            return Enumerable.Range(0, sampleSize)
                             .Select(_ => allTransitionInMemorys[random.Next(allTransitionInMemorys.Count)])
                             .ToList();
        }


        /// <summary>
        /// Samples all episodes from the memory
        /// </summary>
        /// <returns> A List of TransitionInMemorys (in order) of TState type where TState is the observation shape 1d/2d</returns>
        public List<TransitionInMemory<TState>> SampleEntireMemory()
        {
            var result = new List<TransitionInMemory<TState>>();
            //This is done this way because policy optimization works on Episode (TransitionInMemory) batches not batches of episodes
            foreach(var episode in episodes)
            {
                result.AddRange(episode);
            }
            return result;
        }
        /// <summary>
        /// Samples % of experiences upwards starting at startPercentage
        /// For instance, arguments of 50, 9 will return 9% of total experiences between 50% and 59% (oldest to most recent)
        /// </summary>
        /// <returns> A List of TransitionInMemorys (in order) of TState type where TState is the observation shape 1d/2d</returns>
        /// <param name="startPercentage">The starting percentage of experiences to sample</param>
        /// <param name="howManyPercent">The percentage of experiences to sample</param>
        public List<TransitionInMemory<TState>> SamplePortionOfMemory(int startPercentage, int howManyPercent)
        {
            if (startPercentage < 0 || howManyPercent <= 0 || startPercentage + howManyPercent > 100)
                throw new Exception("invalid argument");


            var result = new List<TransitionInMemory<TState>>();
            int totalTransitionInMemorys = episodes.Sum(episode => episode.Count);
            int startIndex = (int)(totalTransitionInMemorys * (startPercentage / 100.0));
            int endIndex = (int)(totalTransitionInMemorys * ((startPercentage + howManyPercent) / 100.0));

            int currentCount = 0;
            foreach (var episode in episodes)
            {
                foreach (var TransitionInMemory in episode)
                {
                    if (currentCount >= startIndex && currentCount < endIndex)
                    {
                        result.Add(TransitionInMemory);
                    }
                    if (currentCount >= endIndex)
                    {
                        break;
                    }
                    currentCount++;
                }
                if (currentCount >= endIndex)
                {
                    break;
                }
            }

            return result;
        }

        /// <summary>
        /// Samples a portion of memory based on the cumulative rewards of episodes.
        /// The method ranks episodes by their total reward and selects the top-performing ones.
        /// </summary>
        /// <param name="topPercentage">The top percentage of episodes to sample based on cumulative rewards.</param>
        /// <returns>A List of TransitionInMemorys from the top-performing episodes.</returns>
        public List<TransitionInMemory<TState>> SamplePortionOfMemoryByRewards(int topPercentage)
        {
            var result = new List<TransitionInMemory<TState>>();

            // Dictionary to hold cumulative rewards for each episode
            var episodeRewards = new Dictionary<List<TransitionInMemory<TState>>, float>();

            // Calculate cumulative rewards for each episode
            foreach (var episode in episodes)
            {
                // Sum up the rewards for each TransitionInMemory in the episode
                float cumulativeReward = episode.Sum(TransitionInMemory => TransitionInMemory.reward);
                episodeRewards.Add(episode, cumulativeReward);
            }

            // Sort episodes by their cumulative rewards in descending order
            var sortedEpisodes = episodeRewards.OrderByDescending(pair => pair.Value)
                                                .Select(pair => pair.Key)
                                                .ToList();


            // Calculate the number of episodes to sample based on the top percentage
            // Ceiling is used to always round up to ensure at least one episode is selected when topPercentage is > 0
            int episodesToSample = (int)Math.Ceiling(sortedEpisodes.Count * (topPercentage / 100.0));

            // Select TransitionInMemorys from the top-performing episodes based on cumulative rewards
            for (int i = 0; i < episodesToSample; i++)
            {
                result.AddRange(sortedEpisodes[i]);
        
            }
            Console.WriteLine(result.Count);

            return result;
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
            episodes = (List<List<TransitionInMemory<TState>>>)bf.Deserialize(fs);
        }
    }
}
*/
