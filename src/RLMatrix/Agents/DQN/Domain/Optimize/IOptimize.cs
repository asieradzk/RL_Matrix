using RLMatrix.Memories;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static TorchSharp.torchvision;

namespace RLMatrix.Agents.DQN.Domain
{
    public interface IComputeQValues
    {
        Tensor ComputeQValues(Tensor states, Module<Tensor, Tensor> policyNet, int[] ActionSizes, int numAtoms);
    }

    public interface IExtractStateActionValues
    {
        Tensor ExtractStateActionValues(Tensor qValues, Tensor actions);
    }

    public interface IComputeNextStateValues
    {
        Tensor ComputeNextStateValues(Tensor nonFinalNextStates, Module<Tensor, Tensor> targetNet, Module<Tensor, Tensor> policyNet, DQNAgentOptions opts, int[] ActionSize, Device device);
    }

    public interface IComputeNStepReturns<T>
    {
        Tensor ComputeNStepReturns(ref ReadOnlySpan<TransitionInMemory<T>> transitions, DQNAgentOptions opts, Device device);
    }

    public interface IComputeExpectedStateActionValues<T>
    {
        Tensor ComputeExpectedStateActionValues(Tensor nextStateValues, Tensor rewardBatch, Tensor nonFinalMask, DQNAgentOptions opts, ref ReadOnlySpan<TransitionInMemory<T>> transitions, int[] ActionCount, Device device);
    }

    public interface IComputeLoss
    {
        Tensor ComputeLoss(Tensor expectedStateActionValues, Tensor stateActionValues);
    }

    public interface IOptimize<T>
    {
        void Optimize(IMemory<T> replayBuffer);
    }

    public interface IGAIL<T>
    {
        void OptimiseDiscriminator(IMemory<T> replayBuffer);
        Tensor AugmentRewardBatch(Tensor stateBatch, Tensor actionBatch, Tensor rewardBatch);
    }
}