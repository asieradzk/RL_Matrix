using RLMatrix.Common;

namespace RLMatrix;

public class CategoricalExpectedStateActionValuesComputer<TState> : IExpectedStateActionValuesComputer<TState>
    where TState : notnull
{
    private readonly BaseLookAheadStepsComputer<TState> _lookAheadStepsComputer = new();
    private readonly float _vMin;
    private readonly float _vMax;
    private readonly int _numAtoms;
    private readonly float _deltaZ;
    //private readonly Tensor _support; TODO: unused field
    private readonly Device _device;

    public CategoricalExpectedStateActionValuesComputer(float vMin, float vMax, int numAtoms, Device device/*, Tensor support*/)
    {
        _vMin = vMin;
        _vMax = vMax;
        _numAtoms = numAtoms;
        _deltaZ = (_vMax - _vMin) / (_numAtoms - 1);
        //_support = support;
        _device = device;
    }
    
    public Tensor ComputeExpectedStateActionValues(Tensor nextStateValues, Tensor rewardBatch, Tensor nonFinalMask, DQNAgentOptions opts, IList<MemoryTransition<TState>> transitions, int[] discreteActions, Device device)
    {
        var maskedDist = torch.zeros(new long[] { opts.BatchSize, discreteActions.Length, _numAtoms }).to(_device); // [batch_size, num_heads, num_atoms]

        if (nonFinalMask.sum().item<long>() > 0)
        {
            Tensor nStepRewardBatch;
            if (opts.LookAheadSteps > 1)
            {
                nStepRewardBatch = _lookAheadStepsComputer.ComputeLookAheadSteps(transitions, opts, device)[nonFinalMask];
            }
            else
            {
                nStepRewardBatch = rewardBatch[nonFinalMask];
            }
            
            var projectedDist = ProjectDistribution(nextStateValues, nStepRewardBatch, opts); // [batch_size_non_final, num_heads, num_atoms]
            maskedDist.index_copy_(0, nonFinalMask.nonzero().squeeze(), projectedDist); // [batch_size, num_heads, num_atoms]
        }

        var terminalMask = nonFinalMask.logical_not();
        var terminalRewards = rewardBatch[terminalMask].unsqueeze(-1).unsqueeze(-1);
        var terminalDist = torch.zeros(new[] { terminalMask.sum().item<long>(), discreteActions.Length, _numAtoms }).to(_device);

        var atomIndices = ((terminalRewards - _vMin) / _deltaZ).round().to(torch.int64).clamp(0, _numAtoms - 1).to(_device);
        var scatterSource = torch.ones_like(terminalDist).to(terminalDist.dtype);

        terminalDist.scatter_(2, atomIndices.expand(-1, discreteActions.Length, -1).to(_device), scatterSource);

        if (terminalMask.sum().item<long>() > 0)
        {
            maskedDist.index_copy_(0, terminalMask.nonzero().squeeze(), terminalDist);
        }

        return maskedDist;
    }
    
    private Tensor ProjectDistribution(Tensor distribution, Tensor rewards, DQNAgentOptions opts)
    {
        var projected = torch.zeros_like(distribution); // [batch_size_non_final, num_heads, num_atoms]
        var zValues = torch.arange(_vMin, _vMax + _deltaZ, _deltaZ).to(_device); // [num_atoms]

        var zTilde = (rewards.view(-1, 1, 1) + opts.Gamma * zValues.view(1, 1, -1)).clamp(_vMin, _vMax); // [batch_size_non_final, 1, num_atoms]
        var bj = (zTilde - _vMin) / _deltaZ; // [batch_size_non_final, 1, num_atoms]
        var lj = bj.floor(); // [batch_size_non_final, 1, num_atoms]
        var uj = bj.ceil(); // [batch_size_non_final, 1, num_atoms]

        var ljMask = (lj >= 0) & (lj < _numAtoms); // [batch_size_non_final, 1, num_atoms]
        var ujMask = (uj >= 0) & (uj < _numAtoms); // [batch_size_non_final, 1, num_atoms]

        var ljMasked = lj.to(ScalarType.Int64) * ljMask.to(ScalarType.Int64); // [batch_size_non_final, 1, num_atoms]
        var ujMasked = uj.to(ScalarType.Int64) * ujMask.to(ScalarType.Int64); // [batch_size_non_final, 1, num_atoms]

        var lowerPart = distribution.gather(2, ljMasked.expand(-1, distribution.size(1), -1)) * (uj - bj) * ljMask; // [batch_size_non_final, num_heads, num_atoms]
        var upperPart = distribution.gather(2, ujMasked.expand(-1, distribution.size(1), -1)) * (bj - lj) * ujMask; // [batch_size_non_final, num_heads, num_atoms]

        projected.scatter_add_(2, ljMasked.expand(-1, distribution.size(1), -1), lowerPart); // [batch_size_non_final, num_heads, num_atoms]
        projected.scatter_add_(2, ujMasked.expand(-1, distribution.size(1), -1), upperPart); // [batch_size_non_final, num_heads, num_atoms]

        return projected;
    }
}