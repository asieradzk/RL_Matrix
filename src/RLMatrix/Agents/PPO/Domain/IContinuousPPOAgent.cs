using RLMatrix.Agents.Common;

namespace RLMatrix.Agents.PPO.Implementations
{
    public interface IContinuousPPOAgent<T> : IContinuousAgent<T>, IHasMemory<T>, IHasOptimizer<T>, ISavable, ISelectContinuousAndDiscreteActions<T>, ISelectContinuousAndDiscreteActionsRecurrent<T> //TODO: ISP violation
    {

    }
}


    

