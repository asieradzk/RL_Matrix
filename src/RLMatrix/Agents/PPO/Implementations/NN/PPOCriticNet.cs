using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using static TorchSharp.torch.nn.utils.rnn;

namespace RLMatrix
{
    public abstract class PPOCriticNet : Module<Tensor, Tensor>
    {
        public PPOCriticNet(string name) : base(name)
        {
        }

        public override abstract Tensor forward(Tensor x);

        public abstract Tensor forward(PackedSequence x); 
    }

    public class PPOCriticNet1D : PPOCriticNet
    {
        private readonly ModuleList<Module<Tensor, Tensor>> fcModules = new();
        private readonly Module<Tensor, Tensor> head;
        private LSTM lstmLayer;
        private int hiddenSize;
        private bool useRnn;


        public PPOCriticNet1D(string name, long inputs, int width, int depth = 3, bool useRNN = false) : base(name)
        {
            // Ensure depth is at least 1.
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");

            this.useRnn = useRNN;
            this.hiddenSize = width;


            if (useRnn)
            {
                // Initialize LSTM layer if useRnn is true
                lstmLayer = nn.LSTM(inputs, hiddenSize, depth, batchFirst: true, dropout: 0.05f);
                //width = hiddenSize; // The output of LSTM layer is now the input for the heads
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


            // Final layer to produce the value estimate.
            head = Linear(width, 1);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
           
            // Adjust for a single input.
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
                // Apply LSTM layer if useRnn is true
                x = lstmLayer.forward(x, null).Item1;
                x = x.reshape(new long[] { x.size(0) * x.size(1), x.size(2) });
                x = functional.tanh(fcModules.First().forward(x));
            }
            else
            {
                x = functional.tanh(fcModules.First().forward(x));
            }


            foreach (var module in fcModules.Skip(1))
            {
                x = functional.tanh(module.forward(x));
            }


            var result = head.forward(x);
            return result;
        }

        public override Tensor forward(PackedSequence x)
        {
            // Unpack the PackedSequence
            var unpackedData = x.data;
            var batchSizes = x.batch_sizes;

            if (useRnn)
            {
                // Apply LSTM layer if useRnn is true
                (var lstmOutput, _, _) = lstmLayer.call(x);
                unpackedData = lstmOutput.data;
                unpackedData = unpackedData.reshape(new long[] { unpackedData.size(0), unpackedData.size(1) });
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

            foreach (var module in fcModules.Skip(1))
            {
                unpackedData = functional.tanh(module.forward(unpackedData));
            }

            var result = head.forward(unpackedData);

            return result;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
 
                foreach (var module in fcModules)
                {
                    module.Dispose();
                }
                lstmLayer?.Dispose();
                head.Dispose();
            }

            base.Dispose(disposing);
        }
    }


    public class PPOCriticNet2D : PPOCriticNet
    {
        private Module<Tensor, Tensor> conv1, flatten, head;
        private readonly ModuleList<Module<Tensor, Tensor>> fcModules = new();
        private LSTM LSTMLayer; // GRU layer
        private int width;
        private long linear_input_size;
        private bool useRnn;
        private int hiddenSize;

        // Calculates the output size for convolutional layers.
        private long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
        {
            return ((inputSize - kernelSize + 2 * padding) / stride) + 1;
        }

        public PPOCriticNet2D(string name, long h, long w, int width, int depth = 3, bool useRNN = false) : base(name)
        {
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");

            this.width = width;
            this.useRnn = useRNN;
            this.hiddenSize = width; // Assuming hidden size to be same as width for simplicity.

            var smallestDim = Math.Min(h, w);
            var padding = smallestDim / 2;

            // Convolutional layer to process 2D input.
            conv1 = Conv2d(1, width, kernelSize: (smallestDim, smallestDim), stride: (1, 1), padding: (padding, padding));

            // Calculate input size for the fully connected layers after convolution and flattening.
            long output_height = CalculateConvOutputSize(h, smallestDim, stride: 1, padding: padding);
            long output_width = CalculateConvOutputSize(w, smallestDim, stride: 1, padding: padding);
            linear_input_size = output_height * output_width * width;

            if (useRnn)
            {
                // Initialize GRU layer if useRnn is true
                LSTMLayer = nn.LSTM(linear_input_size, hiddenSize, depth, batchFirst: true, dropout: 0.1f);
                linear_input_size = hiddenSize; // The output of GRU layer is now the input for the fully connected layers.
            }

            flatten = Flatten();

            // Define the fully connected layers.
            fcModules.Add(Linear(linear_input_size, width));
            for (int i = 1; i < depth; i++)
            {
                fcModules.Add(Linear(width, width));
            }

            // Final layer to produce the value estimate.
            head = Linear(width, 1);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            // Adjust for input dimensions.
            if (x.dim() == 2)
            {
                x = x.unsqueeze(0).unsqueeze(0);
            }
            else if (x.dim() == 3)
            {
                x = x.unsqueeze(1);
            }

            var batchength = x.size(0);
            var sequencLength = x.size(1);

            if (useRnn && x.dim() == 4)
            {
                x = x.reshape(new long[] { batchength * sequencLength, 1, x.size(2), x.size(3) });
            }



            // Process through convolutional layer.
            x = functional.tanh(conv1.forward(x));
            x = flatten.forward(x);

            if (useRnn)
            {
                x = x.reshape(new long[] { batchength, sequencLength, x.size(1) });
                x = LSTMLayer.forward(x, null).Item1;
                x = x.reshape(new long[] { -1, x.size(2) });
            }

            // Process through fully connected layers.
            foreach (var module in fcModules)
            {
                x = functional.tanh(module.forward(x));
            }

            return head.forward(x);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                // Dispose all created modules.
                conv1.Dispose();
                flatten.Dispose();
                LSTMLayer?.Dispose();
                foreach (var module in fcModules)
                {
                    module.Dispose();
                }
                head.Dispose();
                ClearModules();
            }

            base.Dispose(disposing);
        }

        public override Tensor forward(PackedSequence x)
        {
            throw new NotImplementedException();
        }
    }



}
