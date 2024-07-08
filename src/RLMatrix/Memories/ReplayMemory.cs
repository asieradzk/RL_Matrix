using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using RLMatrix.Memories;
using RLMatrix.Agents.Common;
using RLMatrix.Dashboard;

namespace RLMatrix
{
    /// <summary>
    /// The ReplayMemory class represents the memory of the agent in reinforcement learning.
    /// It is used to store and retrieve past experiences (TransitionInMemorys).
    /// </summary>
    public class ReplayMemory<TState> : IMemory<TState>, IStorableMemory
    {
        private int capacity;
        private List<TransitionInMemory<TState>> memory;
        private readonly Random random = new Random();

        /// <summary>
        /// Gets the number of TransitionInMemorys currently stored in the memory.
        /// </summary>
        public int Length => memory.Count;
        public int NumEpisodes { get
            {
                //return num of transitions with no next state
                return memory.Count(x => x.nextState == null);
            }
        }


        /// <summary>
        /// Initializes a new instance of the ReplayMemory class.
        /// </summary>
        /// <param name="capacity">The maximum number of TransitionInMemorys the memory can hold.</param>
        public ReplayMemory(int capacity)
        {
            this.capacity = capacity;
            memory = new List<TransitionInMemory<TState>>(capacity);
        }

        /// <summary>
        /// Adds a new TransitionInMemory to the memory. 
        /// If the memory capacity is reached, the oldest TransitionInMemory is removed.
        /// </summary>
        /// <param name="transition">The TransitionInMemory to be added.</param>
        public void Push(TransitionInMemory<TState> transition)
        {
            if (memory.Count >= capacity)
            {
                memory.RemoveAt(0);
            }
            memory.Add(transition);
        }

        /// <summary>
        /// Adds multiple TransitionInMemorys to the memory.
        /// If the memory capacity is exceeded, the capacity is automatically increased.
        /// </summary>
        /// <param name="transitions">The TransitionInMemorys to be added.</param>
        public void Push(IList<TransitionInMemory<TState>> transitions)
        {
            int count = transitions.Count;
            if (memory.Count + count > capacity)
            {
                // Increase the capacity to accommodate the new transitions
                this.capacity = memory.Count + count;
            }
            ProcessAndUploadEpisodes(transitions);
            memory.AddRange(transitions);
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
        /// <returns>An IList of all TransitionInMemorys in the memory.</returns>
        public IList<TransitionInMemory<TState>> SampleEntireMemory()
        {
            return memory;
        }

        /// <summary>
        /// Samples a specified number of TransitionInMemorys randomly from the memory.
        /// </summary>
        /// <param name="batchSize">The number of TransitionInMemorys to sample.</param>
        /// <returns>An IList of randomly sampled TransitionInMemorys.</returns>
        public IList<TransitionInMemory<TState>> Sample(int batchSize)
        {
            if (batchSize > memory.Count)
            {
                throw new InvalidOperationException("Batch size cannot be greater than current memory size.");
            }

            return memory.OrderBy(x => random.Next()).Take(batchSize).ToList();
        }

        /// <summary>
        /// Clears all TransitionInMemorys from the memory.
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
    }
}