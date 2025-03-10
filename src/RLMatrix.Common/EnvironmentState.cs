namespace RLMatrix.Common;

public sealed record EnvironmentState<TState>(Guid EnvironmentId, TState State) where TState : notnull;