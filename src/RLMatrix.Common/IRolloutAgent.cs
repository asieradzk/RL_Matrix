namespace RLMatrix.Common;

/// <summary>
///     Represents a discrete rollout agent for reinforcement learning.
/// </summary>
public interface IRolloutAgent : ISavable
{
    /// <summary>
    ///     Performs a step in the reinforcement learning process.
    /// </summary>
    /// <param name="isTraining">Whether the agent is in training mode.</param>
    ValueTask StepAsync(bool isTraining = true);
}