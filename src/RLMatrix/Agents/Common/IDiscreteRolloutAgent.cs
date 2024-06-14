namespace RLMatrix.Agents.Common
{
    public interface IDiscreteRolloutAgent<TState>
    {
        Task Step(bool isTraining = true);
    }
}
