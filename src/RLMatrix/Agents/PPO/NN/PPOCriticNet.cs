using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace RLMatrix
{
    public abstract class PPOCriticNet : Module<Tensor, Tensor>
    {
        public PPOCriticNet(string name) : base(name)
        {
        }

        public override abstract Tensor forward(Tensor x);

    }

    public class PPOCriticNet1D : PPOCriticNet
    {
        private readonly ModuleList<Module<Tensor, Tensor>> fcModules = new();
        private readonly Module<Tensor, Tensor> head;

        public PPOCriticNet1D(string name, long inputs, int width, int depth = 3) : base(name)
        {
            // Ensure depth is at least 1.
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");

            // Add base layers.
            fcModules.Add(Linear(inputs, width));
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

            // Pass through base layers.
            foreach (var module in fcModules)
            {
                x = functional.tanh(module.forward(x));
            }

            // Output the value estimate.
            return head.forward(x);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                // Dispose all created modules.
                foreach (var module in fcModules)
                {
                    module.Dispose();
                }
                head.Dispose();
            }

            base.Dispose(disposing);
        }
    }

    public class PPOCriticNet2D : PPOCriticNet
    {
        private Module<Tensor, Tensor> conv1, flatten, head;
        private readonly ModuleList<Module<Tensor, Tensor>> fcModules = new();
        private int width;
        private long linear_input_size;

        // Calculates the output size for convolutional layers.
        private long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
        {
            return ((inputSize - kernelSize + 2 * padding) / stride) + 1;
        }

        public PPOCriticNet2D(string name, long h, long w, int width, int depth = 3) : base(name)
        {
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");

            this.width = width;

            var smallestDim = Math.Min(h, w);

            // Convolutional layer to process 2D input.
            conv1 = Conv2d(1, width, kernelSize: (smallestDim, smallestDim), stride: (1, 1));

            // Calculate input size for the fully connected layers after convolution and flattening.
            long output_height = CalculateConvOutputSize(h, smallestDim);
            long output_width = CalculateConvOutputSize(w, smallestDim);
            linear_input_size = output_height * output_width * width;

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

            // Process through layers.
            x = functional.tanh(conv1.forward(x));
            x = flatten.forward(x);
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
                foreach (var module in fcModules)
                {
                    module.Dispose();
                }
                head.Dispose();
                ClearModules();
            }

            base.Dispose(disposing);
        }
    }


}
