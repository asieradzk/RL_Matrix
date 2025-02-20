namespace RLMatrix;

//TODO: ISP violation?
public interface IContinuousPPOAgent<TState> : IContinuousAgent<TState>, IRecurrentActionSelectable<TState> where TState : notnull;