namespace RLMatrix.Agents.Common
{
#if NET8_0
    public interface IDiscreteRolloutAgent<TState>
    {
        Task Step(bool isTraining = true);

        ValueTask Save(string path);
        public ValueTask Load(string path);
    }
#elif NETSTANDARD2_0
 public interface IDiscreteRolloutAgent<TState>
    {
        Task Step(bool isTraining = true);

        Task Save(string path);
        public Task Load(string path);
    }
#endif



}
