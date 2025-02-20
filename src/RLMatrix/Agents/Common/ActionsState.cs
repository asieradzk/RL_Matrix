using RLMatrix.Common;

namespace RLMatrix;

public sealed record ActionsState(RLActions Actions, Tensor? MemoryState, Tensor? MemoryState2);