using TorchSharp;
using static TorchSharp.torch;

namespace RLMatrix.Agents.DQN.Domain
{
    public static class QOptimizerUtils<T>
    {
        public static void CreateTensorsFromTransitions(ref Device device, ref ReadOnlySpan<TransitionInMemory<T>> transitions, out Tensor nonFinalMask, out Tensor stateBatch, out Tensor nonFinalNextStates, out Tensor actionBatch, out Tensor rewardBatch)
        {
            int length = transitions.Length;
            var fixedActionSize = transitions[0].discreteActions.Length;
            

            // Pre-allocate arrays based on the known batch size
            bool[] nonFinalMaskArray = new bool[length];
            float[] batchRewards = new float[length];
            int[] flatMultiActions = new int[length * fixedActionSize];
            T[] batchStates = new T[length];
            T?[] batchNextStates = new T?[length];

            int flatMultiActionsIndex = 0;
            int nonFinalNextStatesCount = 0;


            for (int i = 0; i < length; i++)
            {
                var transition = transitions[i];
                nonFinalMaskArray[i] = transition.nextState != null;
                batchRewards[i] = transition.reward;
                Array.Copy(transition.discreteActions, 0, flatMultiActions, flatMultiActionsIndex, transition.discreteActions.Length);
                flatMultiActionsIndex += transition.discreteActions.Length; // Assuming a fixed length for all actions

                batchStates[i] = transition.state;
                batchNextStates[i] = transition.nextState;

                if (transition.nextState != null)
                {
                    nonFinalNextStatesCount++;
                }
            }

            stateBatch = StateBatchToTensor(batchStates, device);
            nonFinalMask = torch.tensor(nonFinalMaskArray, device: device);
            rewardBatch = torch.tensor(batchRewards, device: device);
            actionBatch = torch.tensor(flatMultiActions, new long[] { length, fixedActionSize }, torch.int64, device: device); // Reshape based on actual action size


            if (nonFinalNextStatesCount > 0)
            {
                T[] nonFinalNextStatesArray = new T[nonFinalNextStatesCount];
                int index = 0;
                for (int i = 0; i < length; i++)
                {
                    if (batchNextStates[i] is not null)
                    {
                        nonFinalNextStatesArray[index++] = batchNextStates[i];
                    }
                }
                nonFinalNextStates = StateBatchToTensor(nonFinalNextStatesArray, device);
            }
            else
            {
                nonFinalNextStates = torch.zeros(new long[] { 1, stateBatch.shape[1] }, device: device);
            }
        }

        public static void PrintTState(T state)
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
                int rows = stateMatrix.GetLength(0);
                int cols = stateMatrix.GetLength(1);
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
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

            float[] data = new float[tensor.NumberOfElements];
            tensor.data<float>().CopyTo(data, 0);
            return data;
        }

        /// <summary>
        /// Converts the state to a tensor representation for torchsharp. Only float[] and float[,] states are supported.
        /// </summary>
        /// <param name="state">The state to convert.</param>
        /// <returns>The state as a tensor.</returns>
        public static Tensor StateToTensor(T state, Device device)
        {
            switch (state)
            {
                case float[] stateArray:
                    return tensor(stateArray, device: device);
                case float[,] stateMatrix:
                    return tensor(stateMatrix, device: device);
                default:
                    throw new InvalidCastException("State must be either float[] or float[,]");
            }
        }

        public static Tensor StateBatchToTensor(T[] states, Device device)
        {
            // Assume the first element determines the type for all
            if (states.Length == 0)
            {
                throw new ArgumentException("States array cannot be empty.");
            }

            if (states[0] is float[])
            {
                // Handling arrays of float arrays (float[][]).
                return HandleFloatArrayStates(states as float[][], device);
            }
            else if (states[0] is float[,])
            {
                // Handling arrays of float matrices (float[][,]).
                return HandleFloatMatrixStates(states as float[][,], device);
            }
            else
            {
                throw new InvalidCastException("States must be arrays of either float[] or float[,].");
            }
        }

        public static Tensor HandleFloatArrayStates(float[][] states, Device device)
        {
            int totalSize = states.Length * states[0].Length;
            float[] batchData = new float[totalSize];
            int offset = 0;
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
         
            int d1 = states[0].GetLength(0);
            int d2 = states[0].GetLength(1);
            float[] batchData = new float[states.Length * d1 * d2];
            int offset = 0;

            foreach (var matrix in states)
            {
                for (int i = 0; i < d1; i++)
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
}