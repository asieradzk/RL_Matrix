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
    public interface IContinuousAgent<T> : IHasMemory<T>, ISelectContinuousAndDiscreteActions<T>, IHasOptimizer<T>, ISavable
    {
        public int[] DiscreteDimensions { get; init; }
        public (float min, float max)[] ContinuousActionBounds { get; init; }
        public void OptimizeModel();
    }

    public interface ISelectContinuousAndDiscreteActions<T>
    {
        public (int[] discreteActions, float[][] continuousActions) SelectActions(T[] states, bool isTraining);
    }

    public interface ISelectContinuousAndDiscreteActionsRecurrent<T>
    {
        public ((int[] discreteActions, float[] continuousActions) actions, Tensor? memoryState, Tensor? memoryState2)[] SelectActionsRecurrent((T state, Tensor? memoryState, Tensor? memoryState2)[] states, bool isTraining);
    }
}