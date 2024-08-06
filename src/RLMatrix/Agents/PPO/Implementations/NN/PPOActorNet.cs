using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch.nn.utils.rnn;

namespace RLMatrix
{
    public abstract class PPOActorNet : Module<Tensor, Tensor>
    {
        public PPOActorNet(string name) : base(name)
        {
        }

        public override abstract Tensor forward(Tensor x);
        public abstract (Tensor, Tensor, Tensor) forward(Tensor x, Tensor? state, Tensor? state2);
        public abstract Tensor forward(PackedSequence x);

        public static long[] GetIndices(long length)
        {
            if (length < 0)
                throw new ArgumentOutOfRangeException(nameof(length), "Length must be non-negative.");

            long[] indices = new long[length];
            for (int i = 0; i < length; i++)
            {
                indices[i] = i;
            }
            return indices;
        }

        public Tensor get_log_prob<StateTensor>(StateTensor states, Tensor actions, int discreteActions, int contActions)
        {
            Tensor logits;
            switch (states)
            {
                case Tensor state:
                    logits = forward(state);
                    break;
                case PackedSequence state:
                    logits = forward(state);
                    break;
                default:
                    throw new ArgumentException("Invalid state type");
            }

            var discreteLogProbs = new List<Tensor>();
            var continuousLogProbs = new List<Tensor>();

            // Discrete action log probabilities
            for (int i = 0; i < discreteActions; i++)
            {
                using (var actionLogits = logits.select(1, i))
                using (var actionTaken = actions.select(1, i).to(ScalarType.Int64).unsqueeze(-1))
                {
                    var res = torch.nn.functional.log_softmax(actionLogits, dim: 1).gather(dim: 1, index: actionTaken);
                    discreteLogProbs.Add(res);
                }
            }

            // Continuous action log probabilities
            for (int i = 0; i < contActions; i++)
            {
#if NET8_0_OR_GREATER
                using (var mean = logits[.., discreteActions + i, 0].unsqueeze(1))
                using (var logStd = logits[.., discreteActions + contActions + i, 0].unsqueeze(1))
                using (var actionTaken = actions.select(1, discreteActions + i).unsqueeze(1))
#else
        using (var mean = logits[GetIndices(logits.size(0)), discreteActions + i, 0].unsqueeze(1))
        using (var logStd = logits[GetIndices(logits.size(0)), discreteActions + contActions + i, 0].unsqueeze(1))
        using (var actionTaken = actions.select(1, discreteActions + i).unsqueeze(1))
#endif
                {
                    var std = torch.exp(logStd);
                    var diff = actionTaken - mean;
                    var squared_diff = torch.pow(diff / std, 2);
                    var log_prob = (-0.5f * squared_diff - logStd - 0.5f * (float)Math.Log(2 * Math.PI));
                    continuousLogProbs.Add(log_prob);
                }
            }

            // Combine discrete and continuous log probabilities
            using (var combinedLogProbs = torch.cat(discreteLogProbs.Concat(continuousLogProbs).ToArray(), dim: 1))
            {
                var res = combinedLogProbs.squeeze();
                return res;
            }
        }

        public (Tensor logprobs, Tensor entropy) get_log_prob_entropy<StateTensor>(StateTensor states, Tensor actions, int discreteActions, int contActions)
        {
            Tensor logits;
            switch (states)
            {
                case Tensor state:
                    logits = forward(state);
                    break;
                case PackedSequence state:
                    logits = forward(state);
                    break;
                default:
                    throw new ArgumentException("Invalid state type");
            }

            var discreteLogProbs = new List<Tensor>();
            var continuousLogProbs = new List<Tensor>();
            var discreteEntropies = new List<Tensor>();
            var continuousEntropies = new List<Tensor>();

            // Discrete action log probabilities and entropy
            for (int i = 0; i < discreteActions; i++)
            {
                using (var actionLogits = logits.select(1, i))
                using (var actionProbs = torch.nn.functional.softmax(actionLogits, dim: 1))
                using (var actionTaken = actions.select(1, i).to(ScalarType.Int64).unsqueeze(-1))
                {
                    discreteLogProbs.Add(torch.log(actionProbs + 1e-10).gather(dim: 1, index: actionTaken));
                    discreteEntropies.Add(-(actionProbs * torch.log(actionProbs + 1e-10)).sum(1, keepdim: true));
                }
            }

            // Continuous action log probabilities and entropy
            for (int i = 0; i < contActions; i++)
            {
#if NET8_0_OR_GREATER
                using (var mean = logits[.., discreteActions + i, 0].unsqueeze(1))
                using (var logStd = logits[.., discreteActions + contActions + i, 0].unsqueeze(1))
                using (var actionTaken = actions.select(1, discreteActions + i).unsqueeze(1))
#else
        using (var mean = logits[GetIndices(logits.size(0)), discreteActions + i, 0].unsqueeze(1))
        using (var logStd = logits[GetIndices(logits.size(0)), discreteActions + contActions + i, 0].unsqueeze(1))
        using (var actionTaken = actions.select(1, discreteActions + i).unsqueeze(1))
#endif
                {
                    var std = torch.exp(logStd);
                    var diff = actionTaken - mean;
                    var squared_diff = torch.pow(diff / std, 2);
                    var log_prob = -0.5f * squared_diff - logStd - 0.5f * (float)Math.Log(2 * Math.PI);
                    continuousLogProbs.Add(log_prob);
                    var entropy = (logStd + 0.5f * (1 + (float)Math.Log(2 * Math.PI)));
                    continuousEntropies.Add(entropy);
                }
            }

            // Combine discrete and continuous log probabilities and entropies
            using (var combinedLogProbs = torch.cat(discreteLogProbs.Concat(continuousLogProbs).ToArray(), dim: 1))
            using (var combinedEntropies = torch.cat(discreteEntropies.Concat(continuousEntropies).ToArray(), dim: 1))
            {
                var logProbs = combinedLogProbs.squeeze();
                var totalEntropy = combinedEntropies.mean(new long[] { 1 }, true);
                return (logProbs, totalEntropy);
            }
        }



    }


    public class PPOActorNet1D : PPOActorNet
    {
        private readonly ModuleList<Module<Tensor, Tensor>> fcModules = new();
        private readonly ModuleList<Module<Tensor, Tensor>> discreteHeads = new();
        private readonly ModuleList<Module<Tensor, Tensor>> continuousHeadsMean = new();
        private readonly ModuleList<Module<Tensor, Tensor>> continuousHeadsLogStd = new();
        private LSTM lstmLayer;
        //private readonly GRU lstmLayer;
        private readonly int hiddenSize;
        private readonly bool useRnn;
        private readonly int headSize = 1;
        private readonly (float, float)[] continuousActionBounds;

        public PPOActorNet1D(string name, long inputs, int width, int[] discreteActions, (float, float)[] continuousActionBounds, int depth = 3, bool useRNN = false) : base(name)
        {
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");
            this.continuousActionBounds = continuousActionBounds;
            this.useRnn = useRNN;
            this.hiddenSize = width;

            this.headSize = discreteActions.Count() > 0 ? discreteActions[0] : 1;

            if (useRnn)
            {
                // Initialize LSTM layer if useRnn is true
                lstmLayer = nn.LSTM(inputs, hiddenSize, depth, batchFirst: true, dropout: 0.05f);
                // width = hiddenSize; // The output of LSTM layer is now the input for the heads

            }

            // Base layers
            if (useRnn)
            {
                fcModules.Add(Linear(width, width));
            }
            else
            {
                fcModules.Add(Linear(inputs, width));
            }


            for (int i = 1; i < depth; i++)
            {
                fcModules.Add(Linear(width, width));

            }


            // Discrete Heads
            foreach (var actionSize in discreteActions)
            {
                discreteHeads.Add(Linear(width, actionSize));
            }

            // Continuous Heads for means and log std deviations
            foreach (var actionBounds in continuousActionBounds)
            {
                continuousHeadsMean.Add(Linear(width, 1));  // Assuming one output per continuous action dimension for mean
                continuousHeadsLogStd.Add(Linear(width, 1));  // Assuming one output per continuous action dimension for log std deviation
            }

            RegisterComponents();
        }

        private Tensor ApplyHeads(Tensor x)
        {
            var outputs = new List<Tensor>();

            // Process discrete heads
            foreach (var head in discreteHeads)
            {
                outputs.Add(functional.softmax(head.forward(x), 1));
            }

            // Process continuous heads (mean) and apply tanh to bound them within [-1, 1]
            for (int i = 0; i < continuousHeadsMean.Count; i++)
            {
                var continuousOutput = continuousHeadsMean[i].forward(x);  // Apply tanh to bound within [-1, 1]

                // Remap continuous means from [-1, 1] to the desired action bounds
                // var low = continuousActionBounds[i].Item1;
                //  var high = continuousActionBounds[i].Item2;
                //  continuousOutput = (continuousOutput + 1) / 2 * (high - low) + low;  // Remap to [low, high]

                var paddedOutput = torch.zeros(continuousOutput.size(0), headSize, device: x.device);
#if NET8_0_OR_GREATER
                paddedOutput[.., 0] = continuousOutput.squeeze(-1);
#else
                paddedOutput[GetIndices(paddedOutput.size(0)), 0] = continuousOutput.squeeze(-1);
#endif

                outputs.Add(paddedOutput);
            }

            // Process continuous heads (log std)
            for (int i = 0; i < continuousHeadsLogStd.Count; i++)
            {
                var continuousOutput = continuousHeadsLogStd[i].forward(x);
                var paddedOutput = torch.zeros(continuousOutput.size(0), headSize, device: x.device);
#if NET8_0_OR_GREATER
                paddedOutput[.., 0] = continuousOutput.squeeze(-1);
#else
                paddedOutput[GetIndices(paddedOutput.size(0)), 0] = continuousOutput.squeeze(-1);
#endif
                outputs.Add(paddedOutput);
            }

            var result = stack(outputs.ToArray(), dim: 1);
            return result;
        }


        public override Tensor forward(Tensor x)
        {
            if (x.dim() == 1)
            {
                x = x.unsqueeze(0);
            }

            if (useRnn && x.dim() == 2)
            {
                x = x.unsqueeze(0);
            }


            // Apply the first fc module
            if (useRnn)
            {

                var res = lstmLayer.forward(x);
                x = res.Item1;
                x = x.reshape(new long[] { -1, x.size(2) });
                x = functional.tanh(fcModules.First().forward(x));

            }
            else
            {
                x = functional.tanh(fcModules.First().forward(x));
            }



            // Apply the rest of the fc modules
            foreach (var module in fcModules.Skip(1))
            {
                x = functional.tanh(module.forward(x));
            }

            var result = ApplyHeads(x);
            return result;
        }
        /*
        public override (Tensor, Tensor, Tensor) forward(Tensor x, Tensor? state, Tensor? state2)
        {
            if (x.dim() == 1)
            {
                x = x.unsqueeze(0);
            }

            // Create a PackedSequence with the input tensor
            var packedSequence = pack_sequence(new[] { x });

            (Tensor, Tensor)? stateTuple = null;
            if (state is not null && state2 is not null)
            {
                stateTuple = (state, state2);
            }

            // Call the LSTM layer with the PackedSequence and state tuple
            var res = lstmLayer.call(packedSequence, stateTuple);

            // Unpack the output PackedSequence
            var lstmOutput = pad_packed_sequence(res.Item1, batch_first: true);
            x = lstmOutput.Item1;

            x = x.squeeze(0);

            // Extract the hidden state and cell state from the LSTM output
            var hiddenState = res.Item2;
            var cellState = res.Item3;

            x = functional.tanh(fcModules.First().forward(x));

            // Apply the rest of the fc modules
            foreach (var module in fcModules.Skip(1))
            {
                x = functional.tanh(module.forward(x));
            }

            // Apply the heads
            var result = ApplyHeads(x);
        
            // Return the result tensor, hidden state, and cell state
            return (result, hiddenState, cellState);
        }
        */


        public override (Tensor, Tensor, Tensor) forward(Tensor x, Tensor? state, Tensor? state2)
        {
            if (x.dim() == 1)
            {
                x = x.unsqueeze(0);
            }

            if (state is null)
            {
            }


            x = x.unsqueeze(0);
            (Tensor, Tensor, Tensor) lstmResult;
            (Tensor, Tensor)? stateTuple = null;

            if (state is not null && state2 is not null)
            {
                stateTuple = (state, state2);
            }

            lstmResult = lstmLayer.forward(x, stateTuple);
            x = lstmResult.Item1;
            var resultHiddenState = (lstmResult.Item2, lstmResult.Item3);


            x = x.squeeze(0);
            x = functional.tanh(fcModules.First().forward(x));

            // Apply the rest of the fc modules
            foreach (var module in fcModules.Skip(1))
            {
                x = functional.tanh(module.forward(x));
            }

            var result = ApplyHeads(x);
            return (result, resultHiddenState.Item1, resultHiddenState.Item2);
        }

        public override Tensor forward(PackedSequence x)
        {
            // Unpack the PackedSequence
            var unpackedData = x.data;
            var batchSizes = x.batch_sizes;

            if (useRnn)
            {
                (var lstmOutput, _, _) = lstmLayer.call(x, null);
                unpackedData = lstmOutput.data;
                unpackedData = unpackedData.reshape(new long[] { -1, unpackedData.size(1) });
                unpackedData = functional.tanh(fcModules.First().forward(unpackedData));
            }
            else
            {
                // Adjust for a single input
                if (unpackedData.dim() == 1)
                {
                    unpackedData = unpackedData.unsqueeze(0);
                }
                if (unpackedData.dim() == 2)
                {
                    unpackedData = unpackedData.unsqueeze(0);
                }
                unpackedData = functional.tanh(fcModules.First().forward(unpackedData));
            }

            // Apply the rest of the fc modules
            foreach (var module in fcModules.Skip(1))
            {
                unpackedData = functional.tanh(module.forward(unpackedData));
            }

            var result = ApplyHeads(unpackedData);

            return result;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                // Dispose modules in fcModules
                foreach (var module in fcModules)
                {
                    module.Dispose();
                }

                lstmLayer?.Dispose();

                // Dispose discrete heads
                foreach (var head in discreteHeads)
                {
                    head.Dispose();
                }

                // Dispose continuous heads for mean values
                foreach (var head in continuousHeadsMean)
                {
                    head.Dispose();
                }

                // Dispose continuous heads for log standard deviation values
                foreach (var head in continuousHeadsLogStd)
                {
                    head.Dispose();
                }

                // Clear internal module list
                ClearModules();
            }

            base.Dispose(disposing);
        }
    }




    public class PPOActorNet2D : PPOActorNet
    {
        private Module<Tensor, Tensor> conv1, flatten;
        private readonly ModuleList<Module<Tensor, Tensor>> fcModules = new();
        private readonly ModuleList<Module<Tensor, Tensor>> discreteHeads = new();
        private readonly ModuleList<Module<Tensor, Tensor>> continuousHeadsMean = new();
        private readonly ModuleList<Module<Tensor, Tensor>> continuousHeadsLogStd = new();
        private readonly LSTM LSTMLayer;
        private int width;
        private long linear_input_size;
        private bool useRNN;
        private readonly int headSize = 1;
        private readonly (float, float)[] continuousActionBounds;

        public long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
        {
            return ((inputSize - kernelSize + 2 * padding) / stride) + 1;
        }

        public PPOActorNet2D(string name, long h, long w, int[] discreteActionSizes, (float, float)[] continuousActionBounds, int width, int depth = 3, bool useRNN = false) : base(name)
        {
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");

            this.width = width;
            this.useRNN = useRNN;
            this.continuousActionBounds = continuousActionBounds;
            this.headSize = discreteActionSizes.Length > 0 ? discreteActionSizes[0] : 1;

            var smallestDim = Math.Min(h, w);
            var padding = smallestDim / 2;

            conv1 = Conv2d(1, width, kernelSize: (smallestDim, smallestDim), stride: (1, 1), padding: (padding, padding));

            long output_height = CalculateConvOutputSize(h, smallestDim, stride: 1, padding: padding);
            long output_width = CalculateConvOutputSize(w, smallestDim, stride: 1, padding: padding);
            linear_input_size = output_height * output_width * width;

            flatten = Flatten();
            if (useRNN)
            {
                LSTMLayer = nn.LSTM(linear_input_size, width, depth, batchFirst: true);
                fcModules.Add(Linear(width, width));
            }
            else
            {
                fcModules.Add(Linear(linear_input_size, width));
            }

            for (int i = 1; i < depth; i++)
            {
                fcModules.Add(Linear(width, width));
            }

            foreach (var discreteActionSize in discreteActionSizes)
            {
                discreteHeads.Add(Linear(width, discreteActionSize));
            }

            foreach (var actionBound in continuousActionBounds)
            {
                continuousHeadsMean.Add(Linear(width, 1));
                continuousHeadsLogStd.Add(Linear(width, 1));
            }

            RegisterComponents();
        }

        private Tensor ApplyHeads(Tensor x)
        {
            var outputs = new List<Tensor>();

            // Process discrete heads
            foreach (var head in discreteHeads)
            {
                outputs.Add(functional.softmax(head.forward(x), 1));
            }

            // Process continuous heads (mean) and apply tanh to bound them within [-1, 1]
            for (int i = 0; i < continuousHeadsMean.Count; i++)
            {
                var continuousOutput = continuousHeadsMean[i].forward(x);  // Apply tanh to bound within [-1, 1]

                // Remap continuous means from [-1, 1] to the desired action bounds
                //   var low = continuousActionBounds[i].Item1;
                //   var high = continuousActionBounds[i].Item2;
                //  continuousOutput = (continuousOutput + 1) / 2 * (high - low) + low;  // Remap to [low, high]

                var paddedOutput = torch.zeros(continuousOutput.size(0), headSize, device: x.device);
#if NET8_0_OR_GREATER
                paddedOutput[.., 0] = continuousOutput.squeeze(-1);
#else
                paddedOutput[GetIndices(paddedOutput.size(0)), 0] = continuousOutput.squeeze(-1);
#endif
                outputs.Add(paddedOutput);
            }

            // Process continuous heads (log std)
            for (int i = 0; i < continuousHeadsLogStd.Count; i++)
            {
                var continuousOutput = continuousHeadsLogStd[i].forward(x);
                var paddedOutput = torch.zeros(continuousOutput.size(0), headSize, device: x.device);
#if NET8_0_OR_GREATER
                paddedOutput[.., 0] = continuousOutput.squeeze(-1);
#else
                paddedOutput[GetIndices(paddedOutput.size(0)), 0] = continuousOutput.squeeze(-1);
#endif
                outputs.Add(paddedOutput);
            }

            var result = stack(outputs.ToArray(), dim: 1);
            return result;
        }

        public override Tensor forward(Tensor x)
        {
            if (x.dim() == 2)
            {
                x = x.unsqueeze(0).unsqueeze(0);
            }
            else if (x.dim() == 3)
            {
                x = x.unsqueeze(1);
            }

            var batchLength = x.size(0);
            var sequenceLength = x.size(1);

            if (useRNN && x.dim() == 4)
            {
                x = x.reshape(new long[] { batchLength * sequenceLength, 1, x.size(2), x.size(3) });
            }

            x = functional.tanh(conv1.forward(x));
            x = flatten.forward(x);
            if (useRNN)
            {
                x = x.reshape(new long[] { batchLength, sequenceLength, x.size(1) });
                x = LSTMLayer.forward(x, null).Item1;
                x = x.reshape(new long[] { -1, x.size(2) });
            }

            foreach (var module in fcModules)
            {
                x = functional.tanh(module.forward(x));
            }

            return ApplyHeads(x);
        }

        public override (Tensor, Tensor, Tensor) forward(Tensor x, Tensor? state, Tensor? state2)
        {
            if (x.dim() == 2)
            {
                x = x.unsqueeze(0).unsqueeze(0);
            }
            else if (x.dim() == 3)
            {
                x = x.unsqueeze(1);
            }
            x = functional.tanh(conv1.forward(x));
            x = flatten.forward(x);
            x = x.unsqueeze(0);

            (Tensor, Tensor)? stateTuple = null;
            if (state is not null && state2 is not null)
            {
                stateTuple = (state, state2);
            }

            var result = LSTMLayer.forward(x, stateTuple);
            x = result.Item1.squeeze(0);
            var stateRes = (result.Item2, result.Item3);

            foreach (var module in fcModules)
            {
                x = functional.tanh(module.forward(x));
            }

            return (ApplyHeads(x), stateRes.Item1, stateRes.Item2);
        }

        public override Tensor forward(PackedSequence x)
        {
            throw new NotImplementedException();
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                LSTMLayer?.Dispose();
                conv1.Dispose();
                foreach (var module in fcModules.Concat(discreteHeads).Concat(continuousHeadsMean).Concat(continuousHeadsLogStd))
                {
                    module.Dispose();
                }

                ClearModules();
            }

            base.Dispose(disposing);
        }
    }


}
