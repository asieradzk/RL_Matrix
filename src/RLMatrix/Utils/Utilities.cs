using RLMatrix.Common;

namespace RLMatrix;

public static class Utilities<TState>
    where TState : notnull
{
	public static TState DeepCopy(TState input)
	{
		if (input is float[] array1D)
		{
			return (TState)(object)array1D.ToArray();
		}
        
		if (input is float[,] array2D)
		{
			var rows = array2D.GetLength(0);
			var cols = array2D.GetLength(1);
			var clone = new float[rows, cols];
			Buffer.BlockCopy(array2D, 0, clone, 0, array2D.Length * sizeof(float));
			return (TState)(object)clone;
		}
        
		throw new InvalidOperationException("This method can only be used with float[] or float[,].");
	}
	
    //TODO: Not optimised for multi episode batches
    //TODO: This can be multi-threaded optimised
    /// <summary>
    ///		Converts a collection of portable transitions to a list of memory transitions, preserving episode boundaries.
    /// </summary>
    /// <typeparam name="TState">The type of the state in the transitions.</typeparam>
    /// <param name="originalTransitions">The collection of transitions to convert.</param>
    /// <returns>A list of memory transitions, with episode boundaries preserved.</returns>
    public static IList<MemoryTransition<TState>> ConvertToMemoryTransitions(IEnumerable<Transition<TState>> originalTransitions)
    {
        var transitions = originalTransitions as IList<Transition<TState>> ?? originalTransitions.ToList();
        
        // Create TransitionInMemory objects
        var transitionMap = transitions.ToDictionary(x => x.Id, x => (MemoryTransition<TState>) x);

        // Link transitions and set next states
        foreach (var t in transitions)
        {
            if (!t.NextTransitionId.HasValue)
                continue;
            
            var transition = transitionMap[t.Id];
            var nextTransition = transitionMap[t.NextTransitionId.Value];

            transitionMap[t.Id] = new MemoryTransition<TState>(
                transition.State, transition.Actions, transition.Reward, nextTransition.State, nextTransition, transition.PreviousTransition);
            
            transitionMap[t.NextTransitionId.Value] = new MemoryTransition<TState>(
                nextTransition.State, nextTransition.Actions, nextTransition.Reward, nextTransition.NextState, nextTransition.NextTransition, transition);
        }
        
        // Find all first transitions (start of episodes)
        var firstTransitions = transitionMap.Values.Where(t => t.PreviousTransition == null).ToList();
        
        var memoryTransitions = new List<MemoryTransition<TState>>();
        
        // Process each episode
        foreach (var firstTransition in firstTransitions)
        {
            var currentTransition = firstTransition;
            while (currentTransition != null)
            {
                memoryTransitions.Add(currentTransition);
                currentTransition = currentTransition.NextTransition;
            }
        }

        return memoryTransitions;
    }
    
    // TODO: this is DQN specific so maybe should be moved to Q namespace/class
	internal static void CreateTensorsFromTransitions(Device device, IList<MemoryTransition<TState>> transitions, out Tensor nonFinalMask, out Tensor stateBatch, out Tensor nonFinalNextStates, out Tensor actionBatch, out Tensor rewardBatch)
	{
		var length = transitions.Count;
		var fixedActionSize = transitions[0].Actions.DiscreteActions.Length;

		// Pre-allocate arrays based on the known batch size
		var nonFinalMaskArray = new bool[length];
		var batchRewards = new float[length];
		var flatMultiActions = new long[length, fixedActionSize];
		var batchStates = new TState[length];
		var batchNextStates = new TState?[length];

		for (var i = 0; i < length; i++)
		{
			var transition = transitions[i];
			nonFinalMaskArray[i] = transition.NextState != null;
			batchRewards[i] = transition.Reward;

			for (var j = 0; j < fixedActionSize; j++)
			{
				flatMultiActions[i, j] = transition.Actions.DiscreteActions[j];
			}

			batchStates[i] = transition.State;
			batchNextStates[i] = transition.NextState;
		}

		stateBatch = StateBatchToTensor(batchStates, device);
		nonFinalMask = torch.tensor(nonFinalMaskArray, device: device);
		rewardBatch = torch.tensor(batchRewards, device: device);
		actionBatch = torch.tensor(flatMultiActions, dtype: torch.int64, device: device);

		var nonFinalNextStatesCount = batchNextStates.Count(state => state != null);
		if (nonFinalNextStatesCount > 0)
		{
			var nonFinalNextStatesArray = new TState[nonFinalNextStatesCount];
			var index = 0;
			
			for (var i = 0; i < length; i++)
			{
				if (batchNextStates[i] is not null)
				{
					nonFinalNextStatesArray[index++] = batchNextStates[i]!;
				}
			}
			
			nonFinalNextStates = StateBatchToTensor(nonFinalNextStatesArray, device);
		}
		else
		{
			nonFinalNextStates = torch.zeros([1, stateBatch.shape[1]], device: device);
		}
	}
	
	public static void PrintTState(TState state)
	{
		switch (state)
		{
			case float[] stateArray:
				PrintFloatArrayState(stateArray);
				break;
			case float[,] stateMatrix:
				PrintFloatMatrixState(stateMatrix);
				break;
			default:
				throw new InvalidCastException("State must be either float[] or float[,]");
		}
		
		static void PrintFloatArrayState(float[] stateArray)
		{
			Console.WriteLine("Float Array State:");
			Console.WriteLine(string.Join(", ", stateArray));
		}

		static void PrintFloatMatrixState(float[,] stateMatrix)
		{
			Console.WriteLine("Float Matrix State:");
			var rows = stateMatrix.GetLength(0);
			var cols = stateMatrix.GetLength(1);
			for (var i = 0; i < rows; i++)
			{
				for (var j = 0; j < cols; j++)
				{
					Console.Write(stateMatrix[i, j] + " ");
				}
				Console.WriteLine();
			}
		}
	}

	public static float[] ExtractTensorData(Tensor tensor)
	{
		tensor = tensor.cpu();

		var data = new float[tensor.NumberOfElements];
		tensor.data<float>().CopyTo(data);
		return data;
	}

	/// <summary>
	///		Converts the state to a tensor representation for TorchSharp. Only float[] and float[,] states are supported.
	/// </summary>
	/// <param name="state">The state to convert.</param>
	/// <param name="device">The device to operate with.</param>
	/// <returns>The state as a tensor.</returns>
	public static Tensor StateToTensor(TState state, Device device)
	{
		switch (state)
		{
			case float[] stateArray:
				return torch.tensor(stateArray, device: device);
			case float[,] stateMatrix:
				return torch.tensor(stateMatrix, device: device);
			default:
				throw new InvalidCastException("State must be either float[] or float[,]");
		}
	}

	public static Tensor StateBatchToTensor(TState[] states, Device device)
	{
		// Assume the first element determines the type for all
		if (states.Length == 0)
		{
			throw new ArgumentException("States array cannot be empty.");
		}

		return states[0] switch
		{
			float[] when states is float[][] array => HandleFloatArrayStates(array, device),
			float[,] when states is float[][,] matrix => HandleFloatMatrixStates(matrix, device),
			_ => throw new ArgumentException("States must be arrays of either float[] or float[,].")
		};
	}

	public static Tensor HandleFloatArrayStates(float[][] states, Device device)
	{
		var totalSize = states.Length * states[0].Length;
		var batchData = new float[totalSize];
		var offset = 0;
		foreach (var state in states)
		{
			Buffer.BlockCopy(state, 0, batchData, offset * sizeof(float), state.Length * sizeof(float));
			offset += state.Length;
		}
		
		var batchShape = new long[] { states.Length, states[0].Length };
		return torch.tensor(batchData, batchShape, device: device);
	}

	public static Tensor HandleFloatMatrixStates(float[][,] states, Device device)
	{
		var d1 = states[0].GetLength(0);
		var d2 = states[0].GetLength(1);
		var batchData = new float[states.Length * d1 * d2];
		var offset = 0;

		foreach (var matrix in states)
		{
			for (var i = 0; i < d1; i++)
			{
				Buffer.BlockCopy(matrix, i * d2 * sizeof(float), batchData, offset, d2 * sizeof(float));
				offset += d2 * sizeof(float);
			}
		}

		var batchShape = new long[] { states.Length, d1, d2 };
		var result = torch.tensor(batchData, batchShape, device: device);
		return result;
	}
}