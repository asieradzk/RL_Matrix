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

        public int Length => episodes.Sum(episode => episode.Count);

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