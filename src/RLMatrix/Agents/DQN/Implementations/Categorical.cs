using RLMatrix.Agents.DQN.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;
using RLMatrix.Agents.Common;
using RLMatrix.Dashboard;

namespace RLMatrix.Agents.DQN.Implementations.C51
{
    public class CategoricalComputeQValues : IComputeQValues
    {
        int[] ActionSizes;
        int numAtoms;

        public CategoricalComputeQValues(int[] ActionSizes, int numAtoms)
        {
            this.ActionSizes = ActionSizes;
            this.numAtoms = numAtoms;
        }


        public Tensor ComputeQValues(Tensor stateBatch, Module<Tensor, Tensor> policyNet)
        {
            var result = policyNet.forward(stateBatch);
            return result.view(stateBatch.shape[0], ActionSizes.Length, ActionSizes[0], numAtoms); // Shape: [batch_size, num_heads, num_actions, num_atoms]
        }
    }

    public class CategoricalExtractStateActionValues : IExtractStateActionValues
    {
        private readonly int _numAtoms;

        public CategoricalExtractStateActionValues(int numAtoms)
        {
            _numAtoms = numAtoms;
        }

        public Tensor ExtractStateActionValues(Tensor qValues, Tensor actions)
        {
            Tensor expandedActionBatch = actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, _numAtoms);
            Tensor selectedActionDistributions = qValues.gather(2, expandedActionBatch).squeeze(2);
            return selectedActionDistributions; // Shape: [batch_size, num_heads, num_atoms]
        }
    }

    public class CategoricalComputeExpectedStateActionValues<T> : IComputeExpectedStateActionValues<T>
    {
        BaseComputeNStepReturns<T> computeNStepReturns = new BaseComputeNStepReturns<T>();
        private readonly float _vMin;
        private readonly float _vMax;
        private readonly int _numAtoms;
        private readonly float _deltaZ;
        private readonly Tensor _support;
        Device myDevice;

        public CategoricalComputeExpectedStateActionValues(float vMin, float vMax, int numAtoms, Device device, Tensor support)
        {
            _vMin = vMin;
            _vMax = vMax;
            _numAtoms = numAtoms;
            _deltaZ = (_vMax - _vMin) / (_numAtoms - 1);
            _support = support;
            myDevice = device;
        }
        public Tensor ComputeExpectedStateActionValues(Tensor nextStateValues, Tensor rewardBatch, Tensor nonFinalMask, DQNAgentOptions opts, IList<TransitionInMemory<T>> transitions, int[] ActionCount, Device device)
        {
            Tensor maskedDist = zeros(new long[] { opts.BatchSize, ActionCount.Count(), _numAtoms }).to(myDevice); // [batch_size, num_heads, num_atoms]

            if (nonFinalMask.sum().item<long>() > 0)
            {
                Tensor nStepRewardBatch;
                if (opts.NStepReturn > 1)
                {

                    nStepRewardBatch = computeNStepReturns.ComputeNStepReturns(transitions, opts, device)[nonFinalMask];
                }
                else
                {
                    nStepRewardBatch = rewardBatch[nonFinalMask];
                }
                Tensor projectedDist = ProjectDistribution(nextStateValues, nStepRewardBatch, opts); // [batch_size_non_final, num_heads, num_atoms]
                maskedDist.index_copy_(0, nonFinalMask.nonzero().squeeze(), projectedDist); // [batch_size, num_heads, num_atoms]
            }

            Tensor terminalMask = nonFinalMask.logical_not();
            Tensor terminalRewards = rewardBatch[terminalMask].unsqueeze(-1).unsqueeze(-1);
            Tensor terminalDist = zeros(new long[] { terminalMask.sum().item<long>(), ActionCount.Count(), _numAtoms }).to(myDevice);

            Tensor atomIndices = ((terminalRewards - _vMin) / _deltaZ).round().to(torch.int64).clamp(0, _numAtoms - 1).to(myDevice);
            Tensor scatterSource = ones_like(terminalDist).to(terminalDist.dtype);

            terminalDist.scatter_(2, atomIndices.expand(-1, ActionCount.Count(), -1).to(myDevice), scatterSource);

            if (terminalMask.sum().item<long>() > 0)
            {
                maskedDist.index_copy_(0, terminalMask.nonzero().squeeze(), terminalDist);
            }

            return maskedDist;
        }
        private Tensor ProjectDistribution(Tensor distribution, Tensor rewards, DQNAgentOptions opts)
        {
            Tensor projected = zeros_like(distribution); // [batch_size_non_final, num_heads, num_atoms]
            Tensor zValues = arange(_vMin, _vMax + _deltaZ, _deltaZ).to(myDevice); // [num_atoms]

            Tensor zTilde = (rewards.view(-1, 1, 1) + opts.GAMMA * zValues.view(1, 1, -1)).clamp(_vMin, _vMax); // [batch_size_non_final, 1, num_atoms]
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

    }

    public class C51ComputeNextStateValues : IComputeNextStateValues
    {
        private readonly int _numAtoms;

        public C51ComputeNextStateValues(int numAtoms)
        {
            _numAtoms = numAtoms;
        }

        public Tensor ComputeNextStateValues(Tensor nonFinalNextStates, Module<Tensor, Tensor> targetNet, Module<Tensor, Tensor> policyNet, DQNAgentOptions opts, int[] ActionSize, Device device)
        {
            Tensor nextStateDistributions;

            using (no_grad())
            {
                if (nonFinalNextStates.shape[0] > 0)
                {
                    if (opts.DoubleDQN)
                    {
                        // Using policyNet to select the best action for each next state based on current policy
                        Tensor policyDistributions = policyNet.forward(nonFinalNextStates);
                        Tensor meanQValues = policyDistributions.mean(new long[] { 3}); // Average across atoms
                        Tensor nextActions = meanQValues.max(2, keepdim: true).indexes; // Select best action for each head

                        // Evaluating the selected actions' Q-value distributions using targetNet
                        Tensor allNextStateDistributions = targetNet.forward(nonFinalNextStates);

                        // Gather the distributions corresponding to the selected actions
                        nextStateDistributions = allNextStateDistributions.gather(2, nextActions.unsqueeze(3).expand(-1, -1, -1, _numAtoms)).squeeze(2);
                    }
                    else
                    {
                        // Compute the Q-value distributions for all actions in the next states using targetNet
                        nextStateDistributions = targetNet.forward(nonFinalNextStates);

                        // Select the Q-value distributions corresponding to the actions with the highest mean Q-value
                        nextStateDistributions = nextStateDistributions.max(2).values;
                    }
                }
                else
                {
                    nextStateDistributions = zeros(new long[] { opts.BatchSize, _numAtoms }, device: device);
                }
            }

            return nextStateDistributions;
        }
    }


    public class CategoricalComputeLoss : IComputeLoss
    {
        public Tensor ComputeLoss(Tensor stateActionDistributions, Tensor targetDistributions)
        {
            var criterion = torch.nn.KLDivLoss(false, reduction: nn.Reduction.None);
            var loss = criterion.forward(stateActionDistributions.log(), targetDistributions).mean(new long[] { 0, -1 }).sum();
            return loss;
        }
    }
}
