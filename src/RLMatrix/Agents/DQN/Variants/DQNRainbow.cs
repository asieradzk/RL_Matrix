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
    public class DQNAgentRainbow3<T> : DQNPerNoisy<T>
    {
        private readonly float _vMin;
        private readonly float _vMax;
        private readonly int _numAtoms;
        private readonly float _deltaZ;

        public DQNAgentRainbow3(DQNAgentOptions opts, List<IEnvironment<T>> envs, IDQNNetProvider<T> netProvider = null)
            : base(opts, envs, new RainbowNetworkProvider<T>(opts.Width, opts.Depth, opts.NumAtoms))
        {
            _vMin = opts.VMin;
            _vMax = opts.VMax;
            _numAtoms = opts.NumAtoms;
            _deltaZ = (_vMax - _vMin) / (_numAtoms - 1);

            myTargetNet = new RainbowNetworkProvider<T>(opts.Width, opts.Depth, opts.NumAtoms).CreateCriticNet(envs[0]);
            myPolicyNet = new RainbowNetworkProvider<T>(opts.Width, opts.Depth, opts.NumAtoms).CreateCriticNet(envs[0]);
        }

        public override int[] SelectAction(T state, bool isTraining = true)
        {
            if (isTraining)
            {
                foreach (var module in from module in myPolicyNet.modules()
                                       where module is NoisyLinear
                                       select module)
                {
                    ((NoisyLinear)module).ResetNoise();
                }
            }

            using (torch.no_grad())
            {
                Tensor stateTensor = StateToTensor(state); // Shape: [state_dim]
                int[] selectedActions = new int[myEnvironments[0].actionSize.Length];

                Tensor qValuesAllHeads = myPolicyNet.forward(stateTensor).view(1, myEnvironments[0].actionSize.Length, myEnvironments[0].actionSize[0], _numAtoms); // Shape: [1, num_heads, num_actions, num_atoms]

                Tensor support = torch.linspace(_vMin, _vMax, steps: _numAtoms).to(myDevice); // Shape: [num_atoms]

                for (int i = 0; i < myEnvironments[0].actionSize.Length; i++)
                {
                    Tensor qValueDistribution = qValuesAllHeads[0, i]; // Shape: [num_actions, num_atoms]
                    Tensor expectedQValues = (qValueDistribution * support).sum(dim: -1); // Shape: [num_actions]
                    selectedActions[i] = (int)expectedQValues.argmax().item<long>();
                }

                return selectedActions;
            }
        }

        public override void OptimizeModel()
        {
            if (myReplayBuffer.Length < myOptions.BatchSize)
                return;

            List<Transition<T>> transitions = myReplayBuffer.Sample();
            List<int> sampledIndices = null;

            if (myReplayBuffer is PrioritizedReplayMemory<T> prioritizedReplayBuffer)
            {
                sampledIndices = prioritizedReplayBuffer.GetSampledIndices();
            }

            List<T> batchStates = transitions.Select(t => t.state).ToList();
            List<int[]> batchMultiActions = transitions.Select(t => t.discreteActions).ToList();
            List<float> batchRewards = transitions.Select(t => t.reward).ToList();
            List<T> batchNextStates = transitions.Select(t => t.nextState).ToList();

            Tensor nonFinalMask = CreateNonFinalMask(batchNextStates);
            Tensor nextStateValues = ComputeNextStateValues(batchNextStates, nonFinalMask);
            Tensor stateBatch = CreateStateBatch(batchStates);
            Tensor actionBatch = CreateActionBatch(batchMultiActions);
            Tensor rewardBatch = CreateRewardBatch(batchRewards);

            Tensor qValuesAllHeads = ComputeQValues(stateBatch);
            Tensor stateActionValues = ExtractStateActionValues(qValuesAllHeads, actionBatch);

            Tensor expectedStateActionValues = ComputeExpectedStateActionValues(nextStateValues, rewardBatch, nonFinalMask);

            Tensor loss = ComputeLoss(stateActionValues, expectedStateActionValues);

            UpdateModel(loss);

            if (sampledIndices != null)
            {
                UpdatePrioritizedReplayMemory(stateActionValues, expectedStateActionValues.detach(), sampledIndices);
            }
        }

        protected override Tensor ComputeQValues(Tensor stateBatch)
        {
            return myPolicyNet.forward(stateBatch);
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
            Tensor projectedDist = ProjectDistribution(nextStateValues); // [batch_size_non_final, num_heads, num_atoms]
            Tensor maskedDist = zeros(new long[] { myOptions.BatchSize, myEnvironments[0].actionSize.Count(), _numAtoms }).to(myDevice); // [batch_size, num_heads, num_atoms]

            // Handle non-terminal states
            maskedDist.index_copy_(0, nonFinalMask.nonzero().squeeze(), projectedDist); // [batch_size, num_heads, num_atoms]

            // Handle terminal states
            Tensor terminalMask = nonFinalMask.logical_not();
            Tensor terminalRewards = rewardBatch[terminalMask].unsqueeze(-1).unsqueeze(-1);
            Tensor terminalDist = zeros(new long[] { terminalMask.sum().item<long>(), myEnvironments[0].actionSize.Count(), _numAtoms }).to(myDevice);

            Tensor atomIndices = ((terminalRewards - _vMin) / _deltaZ).round().to(torch.int64).clamp(0, _numAtoms - 1).to(myDevice);

            // Cast the source value to the same data type as terminalDist
            Tensor scatterSource = ones_like(terminalDist).to(terminalDist.dtype);

            terminalDist.scatter_(2, atomIndices.expand(-1, myEnvironments[0].actionSize.Count(), -1).to(myDevice), scatterSource);

            maskedDist.index_copy_(0, terminalMask.nonzero().squeeze(), terminalDist);

            return maskedDist;
        }

        private Tensor ProjectDistribution(Tensor distribution)
        {
            // distribution: [batch_size_non_final, num_heads, num_atoms]
            Tensor projected = zeros_like(distribution); // [batch_size_non_final, num_heads, num_atoms]
            Tensor zValues = arange(_vMin, _vMax + _deltaZ, _deltaZ).to(myDevice); // [num_atoms]

            long batchSizeNonFinal = distribution.size(0);
            Tensor rewardBatchNonFinal = CreateRewardBatch(batchSizeNonFinal); // [batch_size_non_final]

            for (int i = 0; i < _numAtoms; i++)
            {
                Tensor zTilde = (rewardBatchNonFinal.unsqueeze(-1) + myOptions.GAMMA * zValues[i]).clamp(_vMin, _vMax); // [batch_size_non_final, num_heads]
                Tensor bj = (zTilde - _vMin) / _deltaZ; // [batch_size_non_final, num_heads]
                Tensor lj = bj.floor(); // [batch_size_non_final, num_heads]
                Tensor uj = bj.ceil(); // [batch_size_non_final, num_heads]

                Tensor ljMask = (lj >= 0) & (lj < _numAtoms); // [batch_size_non_final, num_heads]
                Tensor ujMask = (uj >= 0) & (uj < _numAtoms); // [batch_size_non_final, num_heads]

                Tensor lj_masked = lj.to(ScalarType.Int64) * ljMask.to(ScalarType.Int64); // [batch_size_non_final, num_heads]
                Tensor uj_masked = uj.to(ScalarType.Int64) * ujMask.to(ScalarType.Int64); // [batch_size_non_final, num_heads]

                Tensor lower_part = distribution.gather(2, lj_masked.unsqueeze(-1)).squeeze(-1) * (uj - bj) * ljMask; // [batch_size_non_final, num_heads]
                Tensor upper_part = distribution.gather(2, uj_masked.unsqueeze(-1)).squeeze(-1) * (bj - lj) * ujMask; // [batch_size_non_final, num_heads]

                projected.scatter_add_(2, lj_masked.unsqueeze(2), lower_part.unsqueeze(2)); // [batch_size_non_final, num_heads, num_atoms]
                projected.scatter_add_(2, uj_masked.unsqueeze(2), upper_part.unsqueeze(2)); // [batch_size_non_final, num_heads, num_atoms]
            }

            return projected;
        }

        protected override Tensor ComputeLoss(Tensor stateActionDistributions, Tensor targetDistributions)
        {
            var criterion = torch.nn.KLDivLoss(false, reduction: nn.Reduction.None);
            var loss = criterion.forward(stateActionDistributions.log(), targetDistributions.log()).mean(new long[] { 2 }).sum();

            loss.print();

            return loss;
        }

        protected override void UpdateModel(Tensor loss)
        {
            myOptimizer.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_norm_(myPolicyNet.parameters(), 100);
            myOptimizer.step();
        }
    }
}