using RLMatrix.Common.Dashboard;
using System;
using System.Threading.Tasks;

namespace RLMatrix.Common
{
    /// <summary>
    /// Defines the contract for dashboard clients.
    /// </summary>
    public interface IDashboardClient
    {
        /// <summary>
        /// Adds a new data point to the experiment data collection.
        /// </summary>
        /// <param name="experimentId">The unique identifier for the experiment.</param>
        /// <param name="loss">The loss value.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="reward">The reward value.</param>
        /// <param name="cumulativeReward">The cumulative reward.</param>
        /// <param name="episodeLength">The episode length.</param>
        Task AddDataPoint(ExperimentData data);

        /// <summary>
        /// Delegate for saving the model.
        /// </summary>
        /// <param name="path">The path where the model should be saved.</param>
        Func<string, Task> SaveModel { get; set; }

        /// <summary>
        /// Delegate for loading the model.
        /// </summary>
        /// <param name="path">The path from where the model should be loaded.</param>
        Func<string, Task> LoadModel { get; set; }

        /// <summary>
        /// Delegate for saving the buffer.
        /// </summary>
        /// <param name="path">The path where the buffer should be saved.</param>
        Func<string, Task> SaveBuffer { get; set; }
    }
}