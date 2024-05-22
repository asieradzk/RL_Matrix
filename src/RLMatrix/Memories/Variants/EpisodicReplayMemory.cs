/*using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using Tensorboard;

namespace RLMatrix.Memories
{
    /// <summary>
    /// The EpisodicReplayMemory class represents the memory of the agent in reinforcement learning,
    /// focusing on storing entire episodes of transitions.
    /// </summary>
    public class EpisodicReplayMemory<TState> : IStorableMemory
    {
        private readonly int capacity; // Maximum number of episodes
        public List<List<Transition<TState>>> episodes;
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
        /// Clears all transitions from the memory.
        /// </summary>
        public void ClearMemory()
        {
            episodes.Clear();
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
        public List<Transition<TState>> Sample(int sampleSize)
        {
            // Flatten the list of episodes into a single list of transitions
            var allTransitions = episodes.SelectMany(e => e).ToList();

            // Check if the sample size is larger than the number of available transitions
            if (sampleSize > allTransitions.Count)
            {
                throw new InvalidOperationException("Batch size cannot be greater than the total number of transitions in memory.");
            }

            // Randomly sample transitions
            return Enumerable.Range(0, sampleSize)
                             .Select(_ => allTransitions[random.Next(allTransitions.Count)])
                             .ToList();
        }


        /// <summary>
        /// Samples all episodes from the memory
        /// </summary>
        /// <returns> A List of transitions (in order) of TState type where TState is the observation shape 1d/2d</returns>
        public List<Transition<TState>> SampleEntireMemory()
        {
            var result = new List<Transition<TState>>();
            //This is done this way because policy optimization works on Episode (transition) batches not batches of episodes
            foreach (var episode in episodes)
            {
                result.AddRange(episode);
            }
            return result;
        }
        /// <summary>
        /// Samples % of experiences upwards starting at startPercentage
        /// For instance, arguments of 50, 9 will return 9% of total experiences between 50% and 59% (oldest to most recent)
        /// </summary>
        /// <returns> A List of transitions (in order) of TState type where TState is the observation shape 1d/2d</returns>
        /// <param name="startPercentage">The starting percentage of experiences to sample</param>
        /// <param name="howManyPercent">The percentage of experiences to sample</param>
        public List<Transition<TState>> SamplePortionOfMemory(int startPercentage, int howManyPercent)
        {
            if (startPercentage < 0 || howManyPercent <= 0 || startPercentage + howManyPercent > 100)
                throw new Exception("invalid argument");


            var result = new List<Transition<TState>>();
            int totalTransitions = episodes.Sum(episode => episode.Count);
            int startIndex = (int)(totalTransitions * (startPercentage / 100.0));
            int endIndex = (int)(totalTransitions * ((startPercentage + howManyPercent) / 100.0));

            int currentCount = 0;
            foreach (var episode in episodes)
            {
                foreach (var transition in episode)
                {
                    if (currentCount >= startIndex && currentCount < endIndex)
                    {
                        result.Add(transition);
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
        /// <returns>A List of transitions from the top-performing episodes.</returns>
        public List<Transition<TState>> SamplePortionOfMemoryByRewards(int topPercentage)
        {
            var result = new List<Transition<TState>>();

            // Dictionary to hold cumulative rewards for each episode
            var episodeRewards = new Dictionary<List<Transition<TState>>, float>();

            // Calculate cumulative rewards for each episode
            foreach (var episode in episodes)
            {
                // Sum up the rewards for each transition in the episode
                float cumulativeReward = episode.Sum(transition => transition.reward);
                episodeRewards.Add(episode, cumulativeReward);
            }

            // Sort episodes by their cumulative rewards in descending order
            var sortedEpisodes = episodeRewards.OrderByDescending(pair => pair.Value)
                                                .Select(pair => pair.Key)
                                                .ToList();


            // Calculate the number of episodes to sample based on the top percentage
            // Ceiling is used to always round up to ensure at least one episode is selected when topPercentage is > 0
            int episodesToSample = (int)Math.Ceiling(sortedEpisodes.Count * (topPercentage / 100.0));

            // Select transitions from the top-performing episodes based on cumulative rewards
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
    }
}

*/





using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;

namespace RLMatrix.Memories
{
    public class EpisodicReplayMemory<TState> : IEpisodicMemory<TState>, IStorableMemory
    {
        private readonly int capacity;
        private List<List<TransitionInMemory<TState>>> episodes;
        private readonly Random random = new Random();

        public int Length => episodes.Count;

        public EpisodicReplayMemory(int capacity)
        {
            this.capacity = capacity;
            episodes = new List<List<TransitionInMemory<TState>>>(capacity);
        }

        public void Push(List<TransitionInMemory<TState>> episode)
        {
            if (episodes.Count >= capacity)
            {
                episodes.RemoveAt(0);
            }
            episodes.Add(episode);
        }

        public void Push(TransitionInMemory<TState> transition)
        {
            throw new NotSupportedException("Pushing single transitions is not supported for EpisodicReplayMemory. Use Push(episode) instead.");
        }

        public void Push(IEnumerable<TransitionInMemory<TState>> transitions)
        {
            Push(transitions.ToList());
        }

        public void ClearMemory()
        {
            episodes.Clear();
        }

        public ReadOnlySpan<TransitionInMemory<TState>> Sample()
        {
            int index = random.Next(episodes.Count);
            return episodes[index].ToArray();
        }

        public ReadOnlySpan<TransitionInMemory<TState>> Sample(int batchSize)
        {
            var allTransitions = episodes.SelectMany(e => e).ToArray();

            if (batchSize > allTransitions.Length)
            {
                throw new InvalidOperationException("Batch size cannot be greater than the total number of transitions in memory.");
            }

            var indices = Enumerable.Range(0, allTransitions.Length)
                                    .OrderBy(_ => random.Next())
                                    .Take(batchSize)
                                    .ToArray();

            var sampledTransitions = new TransitionInMemory<TState>[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                sampledTransitions[i] = allTransitions[indices[i]];
            }

            return sampledTransitions;
        }

        public ReadOnlySpan<TransitionInMemory<TState>> SampleEntireMemory()
        {
            return episodes.SelectMany(episode => episode).ToArray();
        }

        public Span<TransitionInMemory<TState>> SamplePortionOfMemory(int startPercentage, int howManyPercent)
        {
            if (startPercentage < 0 || howManyPercent <= 0 || startPercentage + howManyPercent > 100)
                throw new ArgumentException("Invalid argument");

            int totalTransitions = Length;
            int startIndex = (int)(totalTransitions * (startPercentage / 100.0));
            int endIndex = (int)(totalTransitions * ((startPercentage + howManyPercent) / 100.0));

            var result = new TransitionInMemory<TState>[endIndex - startIndex];
            int currentIndex = 0;

            foreach (var episode in episodes)
            {
                foreach (var transition in episode)
                {
                    if (currentIndex >= startIndex && currentIndex < endIndex)
                    {
                        result[currentIndex - startIndex] = transition;
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

        public Span<TransitionInMemory<TState>> SamplePortionOfMemoryByRewards(int topPercentage)
        {
            var episodeRewards = new Dictionary<List<TransitionInMemory<TState>>, float>();

            foreach (var episode in episodes)
            {
                float cumulativeReward = episode.Sum(transition => transition.reward);
                episodeRewards.Add(episode, cumulativeReward);
            }

            var sortedEpisodes = episodeRewards.OrderByDescending(pair => pair.Value)
                                                .Select(pair => pair.Key)
                                                .ToArray();

            int episodesToSample = (int)Math.Ceiling(sortedEpisodes.Length * (topPercentage / 100.0));

            var result = sortedEpisodes.Take(episodesToSample)
                                       .SelectMany(e => e)
                                       .ToArray();

            return result;
        }

        public void Save(string pathToFile)
        {
            using var fs = new FileStream(pathToFile, FileMode.Create);
            throw new NotImplementedException();
        }

        public void Load(string pathToFile)
        {
            if (!File.Exists(pathToFile))
                throw new FileNotFoundException($"File {pathToFile} does not exist.");

            using var fs = new FileStream(pathToFile, FileMode.Open);
            throw new NotImplementedException();
        }
    }
}

