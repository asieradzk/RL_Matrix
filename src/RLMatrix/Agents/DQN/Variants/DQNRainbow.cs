using System;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace RLMatrix.Agents.DQN.Variants
{
    public class RainbowAgent<T> : DQNAgentPER<T>
    {
        private readonly float _vMin;
        private readonly float _vMax;
        private readonly int _numAtoms;
        private readonly float _deltaZ;
        private Tensor _support;

        public RainbowAgent(DQNAgentOptions opts, List<IEnvironment<T>> envs, IDQNNetProvider<T> netProvider = null)
            : base(opts, envs, netProvider ?? new RainbowNetworkProvider<T>(opts.Width, opts.Depth, opts.NumAtoms))
        {
            _vMin = opts.VMin;
            _vMax = opts.VMax;
            _numAtoms = opts.NumAtoms;
            _deltaZ = (_vMax - _vMin) / (_numAtoms - 1);
            _support = torch.linspace(_vMin, _vMax, steps: _numAtoms).to(myDevice); // Shape: [num_atoms]

            if (noisyLayers == null)
            {
                noisyLayers = new();
                noisyLayers.AddRange(from module in myPolicyNet.modules()
                                     where module is NoisyLinear
                                     select (NoisyLinear)module);
            }
        }

        private List<NoisyLinear> noisyLayers;

        public override int[] SelectAction(T state, bool isTraining = true)
        {
            if (isTraining)
            {
                myPolicyNet.train();
                ResetNoise();
            }
            else
            {
                myPolicyNet.eval();
            }

            return ActionsFromState(state);
        }

        public override int[] ActionsFromState(T state)
        {
            using (torch.no_grad())
            {
               
                Tensor stateTensor = StateToTensor(state, myDevice); // Shape: [state_dim]
                Tensor qValuesAllHeads = myPolicyNet.forward(stateTensor).view(1, myEnvironments[0].actionSize.Length, myEnvironments[0].actionSize[0], _numAtoms); // Shape: [1, num_heads, num_actions, num_atoms]
                Tensor expectedQValues = (qValuesAllHeads * _support).sum(dim: -1); // Shape: [1, num_heads, num_actions]
                Tensor bestActions = expectedQValues.argmax(dim: -1).squeeze().to(ScalarType.Int32); // Shape: [num_heads]
                return bestActions.data<int>().ToArray();
            }
        }

        protected override Tensor ComputeQValues(Tensor stateBatch)
        {
            return myPolicyNet.forward(stateBatch).view(stateBatch.shape[0], myEnvironments[0].actionSize.Length, myEnvironments[0].actionSize[0], _numAtoms); // Shape: [batch_size, num_heads, num_actions, num_atoms]
        }

        protected override Tensor ExtractStateActionValues(Tensor qValuesAllHeads, Tensor actionBatch)
        {
            Tensor expandedActionBatch = actionBatch.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, _numAtoms);
            Tensor selectedActionDistributions = qValuesAllHeads.gather(2, expandedActionBatch).squeeze(2);
            return selectedActionDistributions; // Shape: [batch_size, num_heads, num_atoms]
        }

        public override unsafe void OptimizeModel()
        {
            if (myReplayBuffer.Length < myOptions.BatchSize)
                return;

            ReadOnlySpan<TransitionInMemory<T>> transitions = myReplayBuffer.Sample();
            ReadOnlySpan<int> sampledIndices = null;

            if (myReplayBuffer is PrioritizedReplayMemory<T> prioritizedReplayBuffer)
            {
                sampledIndices = prioritizedReplayBuffer.GetSampledIndices();
            }

            CreateTensorsFromTransitions(ref transitions, out Tensor nonFinalMask, out Tensor stateBatch, out Tensor nonFinalNextStates, out Tensor actionBatch, out Tensor rewardBatch);

            Tensor qValuesAllHeads = ComputeQValues(stateBatch);
            Tensor stateActionValues = ExtractStateActionValues(qValuesAllHeads, actionBatch);

            Tensor nextStateValues;
            using (no_grad())
            {
                if (nonFinalNextStates.shape[0] > 0)
                {
                    nextStateValues = myTargetNet.forward(nonFinalNextStates).max(2).values;
                }
                else
                {
                    nextStateValues = zeros(new long[] { myOptions.BatchSize, myEnvironments[0].actionSize.Count() }).to(myDevice);
                }
            }

            Tensor expectedStateActionValues = ComputeExpectedStateActionValues(nextStateValues, rewardBatch, nonFinalMask, 1, ref transitions);
            Tensor loss = ComputeLoss(stateActionValues, expectedStateActionValues);
            UpdateModel(loss);

            if (sampledIndices != null)
            {
                UpdatePrioritizedReplayMemory(stateActionValues, expectedStateActionValues.detach(), sampledIndices.ToArray().ToList());
            }
        }

        protected override Tensor ComputeExpectedStateActionValues(Tensor nextStateValues, Tensor rewardBatch, Tensor nonFinalMask, int nSteps, ref ReadOnlySpan<TransitionInMemory<T>> transitions)
        {
            Tensor maskedDist = zeros(new long[] { myOptions.BatchSize, myEnvironments[0].actionSize.Count(), _numAtoms }).to(myDevice); // [batch_size, num_heads, num_atoms]

            if (nonFinalMask.sum().item<long>() > 0)
            {
                Tensor nStepRewardBatch;
                if (nSteps > 1)
                {

                    nStepRewardBatch = CalculateNStepReturns(ref transitions, nSteps, myOptions.GAMMA)[nonFinalMask];
                }
                else
                {
                    nStepRewardBatch = rewardBatch[nonFinalMask];
                }
                Tensor projectedDist = ProjectDistribution(nextStateValues, nStepRewardBatch); // [batch_size_non_final, num_heads, num_atoms]
                maskedDist.index_copy_(0, nonFinalMask.nonzero().squeeze(), projectedDist); // [batch_size, num_heads, num_atoms]
            }

            Tensor terminalMask = nonFinalMask.logical_not();
            Tensor terminalRewards = rewardBatch[terminalMask].unsqueeze(-1).unsqueeze(-1);
            Tensor terminalDist = zeros(new long[] { terminalMask.sum().item<long>(), myEnvironments[0].actionSize.Count(), _numAtoms }).to(myDevice);

            Tensor atomIndices = ((terminalRewards - _vMin) / _deltaZ).round().to(torch.int64).clamp(0, _numAtoms - 1).to(myDevice);
            Tensor scatterSource = ones_like(terminalDist).to(terminalDist.dtype);

            terminalDist.scatter_(2, atomIndices.expand(-1, myEnvironments[0].actionSize.Count(), -1).to(myDevice), scatterSource);

            if (terminalMask.sum().item<long>() > 0)
            {
                maskedDist.index_copy_(0, terminalMask.nonzero().squeeze(), terminalDist);
            }

            return maskedDist;
        }
        private Tensor ProjectDistribution(Tensor distribution, Tensor rewards)
        {
            Tensor projected = zeros_like(distribution); // [batch_size_non_final, num_heads, num_atoms]
            Tensor zValues = arange(_vMin, _vMax + _deltaZ, _deltaZ).to(myDevice); // [num_atoms]

            Tensor zTilde = (rewards.view(-1, 1, 1) + myOptions.GAMMA * zValues.view(1, 1, -1)).clamp(_vMin, _vMax); // [batch_size_non_final, 1, num_atoms]
            Tensor bj = (zTilde - _vMin) / _deltaZ; // [batch_size_non_final, 1, num_atoms]
            Tensor lj = bj.floor(); // [batch_size_non_final, 1, num_atoms]
            Tensor uj = bj.ceil(); // [batch_size_non_final, 1, num_atoms]

            Tensor ljMask = (lj >= 0) & (lj < _numAtoms); // [batch_size_non_final, 1, num_atoms]
            Tensor ujMask = (uj >= 0) & (uj < _numAtoms); // [batch_size_non_final, 1, num_atoms]

            Tensor lj_masked = lj.to(ScalarType.Int64) * ljMask.to(ScalarType.Int64); // [batch_size_non_final, 1, num_atoms]
            Tensor uj_masked = uj.to(ScalarType.Int64) * ujMask.to(ScalarType.Int64); // [batch_size_non_final, 1, num_atoms]

            Tensor lower_part = distribution.gather(2, lj_masked.expand(-1, distribution.size(1), -1)) * (uj - bj) * ljMask; // [batch_size_non_final, num_heads, num_atoms]
            Tensor upper_part = distribution.gather(2, uj_masked.expand(-1, distribution.size(1), -1)) * (bj - lj) * ujMask; // [batch_size_non_final, num_heads, num_atoms]

            projected.scatter_add_(2, lj_masked.expand(-1, distribution.size(1), -1), lower_part); // [batch_size_non_final, num_heads, num_atoms]
            projected.scatter_add_(2, uj_masked.expand(-1, distribution.size(1), -1), upper_part); // [batch_size_non_final, num_heads, num_atoms]

            return projected;
        }

        protected override Tensor ComputeLoss(Tensor stateActionDistributions, Tensor targetDistributions)
        {
            var criterion = torch.nn.KLDivLoss(false, reduction: nn.Reduction.None);
            var loss = criterion.forward(stateActionDistributions.log(), targetDistributions).mean(new long[] { 0, -1 }).sum();
            return loss;
        }

        public virtual void ResetNoise()
        {
            foreach (var module in noisyLayers)
            {
                module.ResetNoise();
            }
        }
    }
}