using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;
using TorchSharp.Modules;

namespace RLMatrix.GAILNET
{

    public abstract class GAILNET : Module<Tensor, Tensor>
    {
        public GAILNET(string name) : base(name)
        {

        }

        public override abstract Tensor forward(Tensor x);

    }


    public class GAILDiscriminator1D : GAILNET
    {
        private readonly ModuleList<Module<Tensor, Tensor>> fcLayers = new();

        public GAILDiscriminator1D(string name, long stateSize, int[] discreteActionSizes, (float, float)[] continuousActionBounds, int width = 512, int depth = 3) : base(name)
        {
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");

            // Calculate total action size (sum of discrete action sizes and continuous action bounds)
            int totalActionSize = discreteActionSizes.Length + continuousActionBounds.Length;
            // First layer takes state and action sizes as input
            fcLayers.Add(Linear(stateSize + totalActionSize, width));
            for (int i = 1; i < depth; i++)
            {
                fcLayers.Add(Linear(width, width));
            }

            // Output layer
            fcLayers.Add(Linear(width, 1)); // Output is a single scalar value representing 'expertness'

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {

            // Pass through fully connected layers
            for (int i = 0; i < fcLayers.Count - 1; i++)
            {
                x = functional.relu(fcLayers[i].forward(x));
            }

            // Apply the last layer without ReLU activation, then apply sigmoid
            // No need to use squeeze() since we want the output shape to be [batchSize, 1]
            return functional.sigmoid(fcLayers.Last().forward(x)).squeeze(1);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var layer in fcLayers)
                {
                    layer.Dispose();
                }

                ClearModules();
            }

            base.Dispose(disposing);
        }
    }


    public class GAILDiscriminator2D : GAILNET
    {
        private Module<Tensor, Tensor> conv1, flatten;
        private readonly ModuleList<Module<Tensor, Tensor>> fcLayers = new();
        private long linear_input_size;

        public GAILDiscriminator2D(string name, long h, long w, int[] discreteActionSizes, (float, float)[] continuousActionBounds, int width = 128, int depth = 3) : base(name)
        {
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");

            // Calculate total action size (sum of discrete action sizes and continuous action bounds)
            int totalActionSize = discreteActionSizes.Length + continuousActionBounds.Length;

            var smallestDim = Math.Min(h, w);

            conv1 = Conv2d(1, width, kernelSize: (smallestDim, smallestDim), stride: (1, 1));

            long output_height = CalculateConvOutputSize(h, smallestDim);
            long output_width = CalculateConvOutputSize(w, smallestDim);
            linear_input_size = output_height * output_width * width;

            flatten = Flatten();

            // First FC layer takes flattened conv output and action sizes as input
            fcLayers.Add(Linear(linear_input_size + totalActionSize, width));
            for (int i = 1; i < depth; i++)
            {
                fcLayers.Add(Linear(width, width));
            }

            // Output layer
            fcLayers.Add(Linear(width, 1)); // Output is a single scalar value representing 'expertness'

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            x = functional.relu(conv1.forward(x));
            x = flatten.forward(x);

            foreach (var module in fcLayers)
            {
                x = functional.relu(module.forward(x));
            }

            // Sigmoid activation for output
            return functional.sigmoid(fcLayers.Last().forward(x)).squeeze();
        }

        private long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
        {
            return ((inputSize - kernelSize + 2 * padding) / stride) + 1;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                conv1.Dispose();
                flatten.Dispose();
                foreach (var module in fcLayers)
                {
                    module.Dispose();
                }
            }

            base.Dispose(disposing);
        }
    }
}
