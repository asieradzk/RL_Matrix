namespace RLMatrix.Common;

/// <summary>
///     Represents a result from an <see cref="IEnvironment{TState}"/> step execution.
/// </summary>
/// <param name="Reward">The reward from the step.</param>
/// <param name="IsDone">Whether this step should be the last.</param>
public record StepResult(float Reward, bool IsDone);