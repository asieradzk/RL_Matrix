using RLMatrix.Agents.DQN.Domain;
using RLMatrix.Memories;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace RLMatrix.Agents.Common
{
    public interface IHasMemory<T>
    {
        public IMemory<T> Memory { get; set; }
        public void AddTransition(IEnumerable<TransitionPortable<T>> transitions);
    }
    public interface ISelectActions<T>
    {
        public int[][] SelectActions(T[] states, bool isTraining);
    }

    public interface ISelectActionsRecurrent<T>
    {
        public (int[] actions, Tensor? memoryState)[] SelectActionsRecurrent((T state, Tensor? memoryState)[] states, bool isTraining);
    }

    public interface IHasOptimizer<T>
    {
        public IOptimize<T> Optimizer { get; init; }
    }
    public interface IOptimize<T>
    {
        void Optimize(IMemory<T> replayBuffer);
    }
    public interface IDiscreteAgentCore<T>
    {
        public int[] ActionSizes { get; init; }
        public int[][] SelectActions(T[] states, bool isTraining);
        public void OptimizeModel();
    }
}
