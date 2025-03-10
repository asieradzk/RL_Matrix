namespace RLMatrix;

public record RLMemoryState<TState>(TState State, Tensor? MemoryState, Tensor? MemoryState2) where TState : notnull;