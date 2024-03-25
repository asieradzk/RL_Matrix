using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace RLMatrix.Agents.DQN.Variants
{
    public class DQNAgentRainbow<T> : DQNPerNoisy<T>
    {
        private readonly float _vMin;
        private readonly float _vMax;
        private readonly int _numAtoms;
        private readonly float _deltaZ;

        public DQNAgentRainbow(DQNAgentOptions opts, List<IEnvironment<T>> envs, IDQNNetProvider<T> netProvider = null)
            : base(opts, envs, netProvider ?? new RainbowNetworkProvider<T>(opts.Width, opts.Depth, opts.NumAtoms))
        {
            _vMin = opts.VMin;
            _vMax = opts.VMax;
            _numAtoms = opts.NumAtoms;
            _deltaZ = (_vMax - _vMin) / (_numAtoms - 1);
            support = torch.linspace(_vMin, _vMax, steps: _numAtoms).to(myDevice); // Shape: [num_atoms]

        }
        Tensor support;

        public override int[] SelectAction(T state, bool isTraining = true)
        {
            if (isTraining)
            {
                ResetNoise();
            }

            //return random actions for test
            return this.ActionsFromState(state);
        }

        public override int[] ActionsFromState(T state)
        {
            using (torch.no_grad())
            {
                Tensor stateTensor = StateToTensor(state); // Shape: [state_dim]
                Tensor qValuesAllHeads = myPolicyNet.forward(stateTensor).view(1, myEnvironments[0].actionSize.Length, myEnvironments[0].actionSize[0], _numAtoms); // Shape: [1, num_heads, num_actions, num_atoms]
                Tensor expectedQValues = (qValuesAllHeads * support).sum(dim: -1); // Shape: [1, num_heads, num_actions]
                Tensor bestActions = expectedQValues.argmax(dim: -1).squeeze().to(ScalarType.Int32); // Shape: [num_heads]
                return bestActions.data<int>().ToArray();
            
            }
        }

        protected override Tensor ExtractStateActionValues(Tensor qValuesAllHeads, Tensor actionBatch)
        {
            // Ensure actionBatch is expanded to index properly into qValuesAllHeads
            Tensor expandedActionBatch = actionBatch.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, _numAtoms);
            // Gather the Q-value distributions for the selected actions across all heads
            Tensor selectedActionDistributions = qValuesAllHeads.gather(2, expandedActionBatch).squeeze(2);
            return selectedActionDistributions; // Shape: [batch_size, num_heads, num_atoms]
        }

        protected override Tensor ComputeNextStateValues(List<T> batchNextStates, Tensor nonFinalMask)
        {
            Tensor[] nonFinalNextStatesArray = batchNextStates.Where(s => s != null).Select(s => StateToTensor(s)).ToArray();

            if (nonFinalNextStatesArray.Length > 0)
            {
                Tensor nonFinalNextStates = stack(nonFinalNextStatesArray).to(myDevice);

                using (no_grad())
                {
                    // Use the target network to compute the distribution of Q-values for each next state
                    Tensor nextQDistributions = myTargetNet.forward(nonFinalNextStates); // Shape: [batch_size, num_heads, num_actions, num_atoms]

                    // Compute the expected Q-value for each action by taking the sum of the distribution multiplied by the support
                    Tensor expectedQValues = (nextQDistributions * torch.linspace(_vMin, _vMax, steps: _numAtoms).to(myDevice)).sum(dim: -1); // Shape: [batch_size, num_heads, num_actions]

                    // Select the best action for each head based on the expected Q-values
                    Tensor bestActions = expectedQValues.argmax(dim: -1); // Shape: [batch_size, num_heads]

                    // Gather the corresponding Q-value distributions for the best actions
                    Tensor bestQDistributions = nextQDistributions.gather(2, bestActions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, _numAtoms)).squeeze(2); // Shape: [batch_size, num_heads, num_atoms]
                    return bestQDistributions;
                }
            }
            else
            {
                // All steps are final, likely this means all episodes are 1-step long.
                return zeros(new long[] { myOptions.BatchSize, myEnvironments[0].actionSize.Length, _numAtoms }).to(myDevice);
            }
        }

        protected override Tensor ComputeExpectedStateActionValues(Tensor nextStateValues, Tensor rewardBatch, Tensor nonFinalMask)
        {
            // nextStateValues: [batch_size_non_final, num_heads, num_atoms]
            // rewardBatch: [batch_size]
            // nonFinalMask: [batch_size]
            Tensor maskedDist = zeros(new long[] { myOptions.BatchSize, myEnvironments[0].actionSize.Count(), _numAtoms }).to(myDevice); // [batch_size, num_heads, num_atoms]

            if (nonFinalMask.sum().item<long>() > 0)
            {
                // Handle non-terminal states
                Tensor projectedDist = ProjectDistribution(nextStateValues); // [batch_size_non_final, num_heads, num_atoms]
                maskedDist.index_copy_(0, nonFinalMask.nonzero().squeeze(), projectedDist); // [batch_size, num_heads, num_atoms]
            }

            // Handle terminal states
            Tensor terminalMask = nonFinalMask.logical_not();
            Tensor terminalRewards = rewardBatch[terminalMask].unsqueeze(-1).unsqueeze(-1);
            Tensor terminalDist = zeros(new long[] { terminalMask.sum().item<long>(), myEnvironments[0].actionSize.Count(), _numAtoms }).to(myDevice);

            Tensor atomIndices = ((terminalRewards - _vMin) / _deltaZ).round().to(torch.int64).clamp(0, _numAtoms - 1).to(myDevice);

            // Cast the source value to the same data type as terminalDist
            Tensor scatterSource = ones_like(terminalDist).to(terminalDist.dtype);

            terminalDist.scatter_(2, atomIndices.expand(-1, myEnvironments[0].actionSize.Count(), -1).to(myDevice), scatterSource);

            if (terminalMask.sum().item<long>() > 0)
            {
                maskedDist.index_copy_(0, terminalMask.nonzero().squeeze(), terminalDist);
            }

            return maskedDist;
        }

        private Tensor ProjectDistribution(Tensor distribution)
        {
            // distribution: [batch_size_non_final, num_heads, num_atoms]
            Tensor projected = zeros_like(distribution); // [batch_size_non_final, num_heads, num_atoms]
            Tensor zValues = arange(_vMin, _vMax + _deltaZ, _deltaZ).to(myDevice); // [num_atoms]

            long batchSizeNonFinal = distribution.size(0);
            Tensor rewardBatchNonFinal = CreateRewardBatch(batchSizeNonFinal); // [batch_size_non_final]

            Tensor zTilde = (rewardBatchNonFinal.view(-1, 1, 1) + myOptions.GAMMA * zValues.view(1, 1, -1)).clamp(_vMin, _vMax); // [batch_size_non_final, 1, num_atoms]
            Tensor bj = (zTilde - _vMin) / _deltaZ; // [batch_size_non_final, 1, num_atoms]
            Tensor lj = bj.floor(); // [batch_size_non_final, 1, num_atoms]
            Tensor uj = bj.ceil(); // [batch_size_non_final, 1, num_atoms]

            Tensor ljMask = (lj >= 0) & (lj < _numAtoms); // [batch_size_non_final, 1, num_atoms]
            Tensor ujMask = (uj >= 0) & (uj < _numAtoms); // [batch_size_non_final, 1, num_atoms]

            Tensor lj_masked = lj.to(ScalarType.Int64) * ljMask.to(ScalarType.Int64); // [batch_size_non_final, 1, num_atoms]
            Tensor uj_masked = uj.to(ScalarType.Int64) * ujMask.to(ScalarType.Int64); // [batch_size_non_final, 1, num_atoms]

            Tensor lower_part = distribution.gather(2, lj_masked) * (uj - bj) * ljMask; // [batch_size_non_final, num_heads, num_atoms]
            Tensor upper_part = distribution.gather(2, uj_masked) * (bj - lj) * ujMask; // [batch_size_non_final, num_heads, num_atoms]

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
    }
}