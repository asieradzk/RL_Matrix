using RLMatrix.Common;

namespace RLMatrix;

public class QOptimizer<TState> : IOptimizer<TState>
	where TState: notnull
{
	private readonly TensorModule _policyNet;
	private readonly TensorModule _targetNet;
	private OptimizerHelper _optimizerHelper;
	private readonly IQValuesComputer _qValuesCalculator;
	private readonly IStateActionValuesExtractor _stateActionValueExtractor;
	private readonly INextStateValuesComputer _nextStateValuesCalculator;
	private readonly IExpectedStateActionValuesComputer<TState> _expectedStateActionValuesComputerCalculator;
	private readonly ILossComputer _lossComputerCalculator;
	private readonly DQNAgentOptions _options;
	private Device _device;
	private readonly int[] _discreteActionDimensions;
	private LRScheduler? _scheduler;
	private readonly IGAIL<TState>? _gail;
	private int _updateCounter;

	public QOptimizer(TensorModule policyNet, TensorModule targetNet, OptimizerHelper optimizerHelper, 
		IQValuesComputer qValuesCalculator, IStateActionValuesExtractor stateActionValueExtractor, 
		INextStateValuesComputer nextStateValuesCalculator, IExpectedStateActionValuesComputer<TState> expectedStateActionValuesComputerCalculator,
		ILossComputer lossComputerCalculator, DQNAgentOptions options, Device device, int[] discreteActionDimensions, 
		LRScheduler? scheduler = null, IGAIL<TState>? gail = null)
	{
		_policyNet = policyNet;
		_targetNet = targetNet;
		_optimizerHelper = optimizerHelper;
		_qValuesCalculator = qValuesCalculator;
		_stateActionValueExtractor = stateActionValueExtractor;
		_nextStateValuesCalculator = nextStateValuesCalculator;
		_expectedStateActionValuesComputerCalculator = expectedStateActionValuesComputerCalculator;
		_lossComputerCalculator = lossComputerCalculator;
		_options = options;
		_device = device;
		_discreteActionDimensions = discreteActionDimensions;
		
		_scheduler = scheduler;
		_gail = gail;
	}

	public ValueTask UpdateOptimizersAsync(LRScheduler? scheduler)
	{
		_optimizerHelper = torch.optim.Adam(_policyNet.parameters(), lr: _options.LearningRate, amsgrad: true);
		scheduler ??= new CyclicLR(_optimizerHelper, _options.LearningRate * 0.5f, _options.LearningRate * 2f, step_size_up: 500, step_size_down: 2000, cycle_momentum: false);
		_scheduler = scheduler;
		return new();
	}

	public async ValueTask OptimizeAsync(IMemory<TState> replayBuffer)
	{
		using (torch.NewDisposeScope())
		{
			if (_gail != null && replayBuffer.Length > 0)
			{
				_gail.OptimiseDiscriminator(replayBuffer);
			}

			if (replayBuffer.Length < _options.BatchSize)
				return;

			var transitions = replayBuffer.Sample(_options.BatchSize);
			Utilities<TState>.CreateTensorsFromTransitions(_device, transitions, out var nonFinalMask, out var stateBatch, out var nonFinalNextStates, out var actionBatch, out var rewardBatch);
			if (_gail != null)
			{
				_gail.OptimiseDiscriminator(replayBuffer);
				rewardBatch = _gail.AugmentRewardBatch(stateBatch, actionBatch, rewardBatch);
			}
			
			var qValuesAllHeads = _qValuesCalculator.ComputeQValues(stateBatch, _policyNet);
			var stateActionValues = _stateActionValueExtractor.ExtractStateActionValues(qValuesAllHeads, actionBatch);
			var nextStateValues = _nextStateValuesCalculator.ComputeNextStateValues(nonFinalNextStates, _targetNet, _policyNet, _options, _discreteActionDimensions, _device);
			var expectedStateActionValues = _expectedStateActionValuesComputerCalculator.ComputeExpectedStateActionValues(nextStateValues, rewardBatch, nonFinalMask, _options, transitions, _discreteActionDimensions, _device);

			var loss = _lossComputerCalculator.ComputeLoss(stateActionValues, expectedStateActionValues);
			UpdateModel(loss);
			_scheduler!.step();

			var dashboard = await DashboardProvider.Instance.GetDashboardAsync();
			dashboard.UpdateLoss(loss.item<float>());
			dashboard.UpdateLearningRate((float) _scheduler!.get_last_lr().FirstOrDefault());

			if (replayBuffer is PrioritizedReplayMemory<TState> prioritizedReplayBuffer)
			{
				var sampledIndices = prioritizedReplayBuffer.GetSampledIndices();
				UpdatePrioritizedReplayMemory(prioritizedReplayBuffer, stateActionValues, expectedStateActionValues.detach(), sampledIndices);
			}
		}
	}
	
	protected virtual void UpdatePrioritizedReplayMemory(PrioritizedReplayMemory<TState> prioritizedReplayBuffer, Tensor stateActionValues, Tensor detachedExpectedStateActionValues, Span<int> sampledIndices)
	{
		var tdErrors = (stateActionValues - detachedExpectedStateActionValues).abs();
		var errors = Utilities<TState>.ExtractTensorData(tdErrors);

		for (var i = 0; i < sampledIndices.Length; i++)
		{
			prioritizedReplayBuffer.UpdatePriority(sampledIndices[i], errors[i] + 0.001f);
		}
	}

	private void UpdateModel(Tensor loss)
	{
		_optimizerHelper.zero_grad();
		loss.backward();
		torch.nn.utils.clip_grad_value_(_policyNet.parameters(), 100);
		_optimizerHelper.step();
		_updateCounter++;
		
		if (_updateCounter >= _options.SoftUpdateInterval)
		{
			SoftUpdateTargetNetwork();
			_updateCounter = 0;
		}
	}

	/// <summary>
	///		Performs an optimized soft update of the target network parameters using vectorized operations.
	/// </summary>
	private void SoftUpdateTargetNetwork()
	{
		using (torch.no_grad())
		{
			var targetParams = _targetNet.parameters().ToList();
			var policyParams = _policyNet.parameters().ToList();

			// Combine all parameters into single tensors
			var targetTensor = torch.cat(targetParams.Select(p => p.flatten()).ToList());
			var policyTensor = torch.cat(policyParams.Select(p => p.flatten()).ToList());

			// Perform the soft update in a single operation
			targetTensor.mul_(1 - _options.Tau).add_(policyTensor, alpha: _options.Tau);

			// Update the target network parameters
			var index = 0;
			foreach (var param in targetParams)
			{
				var flatSize = (int)param.numel();
				param.copy_(targetTensor.narrow(0, index, flatSize).view_as(param));
				index += flatSize;
			}
		}
	}
}