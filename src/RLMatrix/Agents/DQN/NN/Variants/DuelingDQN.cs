using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace RLMatrix
{
    public sealed class DuelingDQN : DQNNET
    {
        private readonly ModuleList<Module<Tensor, Tensor>> sharedModules = new ModuleList<Module<Tensor, Tensor>>();
        private readonly Module<Tensor, Tensor> valueHead;
        private readonly ModuleList<Module<Tensor, Tensor>> advantageHeads = new ModuleList<Module<Tensor, Tensor>>();

        public DuelingDQN(string name, int obsSize, int width, int[] actionSizes, int depth = 4) : base(name)
        {
            if (obsSize < 1)
            {
                throw new ArgumentException("Number of observations can't be less than 1");
            }

            // Shared layers configuration
            sharedModules.Add(Linear(obsSize, width)); // First shared layer
            for (int i = 1; i < depth; i++)
            {
                sharedModules.Add(Linear(width, width)); // Additional shared layers
            }

            // Separate value stream (only one head because state value doesn't depend on action)
            valueHead = Linear(width, 1);
            // Separate advantage streams (multiple heads for different action sizes)
            foreach (var actionSize in actionSizes)
            {
                advantageHeads.Add(Linear(width, actionSize));
            }

            // Register the modules for the network
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            if (x.dim() == 1)
            {
                x = x.unsqueeze(0); // Ensure input is at least 2D, now [batch_size, feature_size]
            }

            // Forward through shared layers
            foreach (var module in sharedModules)
            {
                x = functional.relu(module.forward(x)); // After this, shape should still be [batch_size, feature_size]
            }

            // Forward through value stream
            var value = valueHead.forward(x).unsqueeze(1); // This should make shape [batch_size, 1, 1] to prepare for broadcasting

            // Forward through each advantage stream (head)
            var advantageList = new List<Tensor>();
            foreach (var head in advantageHeads)
            {
                var advantage = head.forward(x); // Expected shape [batch_size, actionSize]
                                               
                advantage = advantage.unsqueeze(1); 
                advantageList.Add(advantage);
            }

            // Combine value and advantages to get Q values for each head/action size
            var qValuesList = new List<Tensor>();
            foreach (var advantage in advantageList)
            {
                // Broadcast value across advantage's actions, should align shapes
                var qValues = value + (advantage - advantage.mean(dimensions: new long[] { 2 }, keepdim: true)); // Subtract mean along the correct dimension
                qValuesList.Add(qValues.squeeze(1)); // Remove the extra dimension to get [batch_size, actionSize] before stacking
            }

        
            var finalOutput = torch.stack(qValuesList, dim: 1);
            return finalOutput; // Expected final shape [batch_size, num_heads, actionSize]
        }



        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var module in sharedModules)
                {
                    module.Dispose();
                }
                valueHead.Dispose();
                foreach (var head in advantageHeads)
                {
                    head.Dispose();
                }
            }

            base.Dispose(disposing);
        }
    }


    public sealed class DuelingDQN2D : DQNNET
    {
        private readonly Module<Tensor, Tensor> conv1, flatten;
        private readonly ModuleList<Module<Tensor, Tensor>> sharedModules = new ModuleList<Module<Tensor, Tensor>>();
        private readonly Module<Tensor, Tensor> valueHead;
        private readonly ModuleList<Module<Tensor, Tensor>> advantageHeads = new ModuleList<Module<Tensor, Tensor>>();
        private readonly int width;
        private long linear_input_size;

        public DuelingDQN2D(string name, long h, long w, int[] actionSizes, int width, int depth = 3) : base(name)
        {
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");
            this.width = width;

            var smallestDim = Math.Min(h, w);
            conv1 = Conv2d(1, width, kernelSize: (smallestDim, smallestDim), stride: (1, 1));

            long output_height = CalculateConvOutputSize(h, smallestDim);
            long output_width = CalculateConvOutputSize(w, smallestDim);
            linear_input_size = output_height * output_width * width;
            flatten = Flatten();

            // First FC layer (always present)
            sharedModules.Add(Linear(linear_input_size, width));

            // Additional depth-1 FC layers
            for (int i = 1; i < depth; i++)
            {
                sharedModules.Add(Linear(width, width));
            }

            // Separate value stream (only one head because state value doesn't depend on action)
            valueHead = Linear(width, 1);

            // Separate advantage streams (multiple heads for different action sizes)
            foreach (var actionSize in actionSizes)
            {
                advantageHeads.Add(Linear(width, actionSize));
            }

            RegisterComponents();
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

            x = functional.relu(conv1.forward(x));
            x = flatten.forward(x);

            foreach (var module in sharedModules)
            {
                x = functional.relu(module.forward(x));
            }

            // Forward through value stream
            var value = valueHead.forward(x).unsqueeze(1);

            // Forward through each advantage stream (head)
            var advantageList = new List<Tensor>();
            foreach (var head in advantageHeads)
            {
                var advantage = head.forward(x).unsqueeze(1);
                advantageList.Add(advantage);
            }

            // Combine value and advantages to get Q values for each head/action size
            var qValuesList = new List<Tensor>();
            foreach (var advantage in advantageList)
            {
                var qValues = value + (advantage - advantage.mean(dimensions: new long[] { 2 }, keepdim: true));
                qValuesList.Add(qValues.squeeze(1));
            }

            // Stack along a new dimension for the heads
            var finalOutput = torch.stack(qValuesList, dim: 1);
            return finalOutput; // Expected final shape [batch_size, num_heads, actionSize]
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                conv1.Dispose();
                flatten.Dispose();
                foreach (var module in sharedModules)
                {
                    module.Dispose();
                }
                valueHead.Dispose();
                foreach (var head in advantageHeads)
                {
                    head.Dispose();
                }
            }

            base.Dispose(disposing);
        }

        private long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
        {
            return ((inputSize - kernelSize + 2 * padding) / stride) + 1;
        }
    }
}
