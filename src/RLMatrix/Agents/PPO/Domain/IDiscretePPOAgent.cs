namespace RLMatrix;

public interface IDiscretePPOAgent<TState> : IDiscreteAgent<TState>, IRecurrentActionSelectable<TState> where TState : notnull;