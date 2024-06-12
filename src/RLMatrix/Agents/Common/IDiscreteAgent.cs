using RLMatrix.Agents.DQN.Domain;
using RLMatrix.Memories;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;

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
        public (int[] actions, Tensor? memoryState, Tensor? memoryState2)[] SelectActionsRecurrent((T state, Tensor? memoryState, Tensor? memoryState2)[] states, bool isTraining);
    }

    public interface IHasOptimizer<T>
    {
        public IOptimize<T> Optimizer { get; init; }
    }
    public interface ISavable
    {
        void Save(string path);
        void Load(string path, LRScheduler scheduler = null);
    }
    public interface IOptimize<T>
    {
        void Optimize(IMemory<T> replayBuffer);
        void UpdateOptimizers(LRScheduler scheduler);
    }
    public interface IDiscreteAgentCore<T>
    {
        public int[] ActionSizes { get; init; }
        public int[][] SelectActions(T[] states, bool isTraining);
        public void OptimizeModel();
    }
}
