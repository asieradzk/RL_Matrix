using RLMatrix.Agents.Common;

namespace RLMatrix.Agents.PPO.Implementations
{
    public interface IDiscretePPOAgent<T> : IDiscreteAgentCore<T>, IHasMemory<T>, IHasOptimizer<T>, ISelectActionsRecurrent<T> //TODO: ISP violation
    {

    }
}


    

