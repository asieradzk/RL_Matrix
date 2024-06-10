using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace RLMatrix
{
    public abstract class DQNNET : Module<Tensor, Tensor>
    {
        public DQNNET(string name) : base(name)
        {
        }

        public override abstract Tensor forward(Tensor x);
    }

    public sealed class DQN1D : DQNNET
    {
        private readonly ModuleList<Module<Tensor, Tensor>> modules = new();
        private readonly ModuleList<Module<Tensor, Tensor>> heads = new();

        public DQN1D(string name, int obsSize, int width, int[] actionSizes, int depth = 4, bool noisyLayers = false, float noiseScale = 0.0001f) : base(name)
        {
            if (obsSize < 1)
            {
                throw new ArgumentException("Number of observations can't be less than 1");
            }

            modules.Add(noisyLayers ? new NoisyLinear(obsSize, width, std_init: noiseScale) : Linear(obsSize, width));

            for (int i = 1; i < depth; i++)
            {
                modules.Add(noisyLayers ? new NoisyLinear(width, width, std_init: noiseScale) : Linear(width, width));
            }

            foreach (var actionSize in actionSizes)
            {
                heads.Add(noisyLayers ? new NoisyLinear(width, actionSize, std_init: noiseScale) : Linear(width, actionSize));
            }

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            if (x.dim() == 1)
            {
                x = x.unsqueeze(0);
            }

            foreach (var module in modules)
            {
                x = functional.relu(module.forward(x));
            }

            var outputs = new List<Tensor>();
            foreach (var head in heads)
            {
                outputs.Add(head.forward(x));
            }

            return stack(outputs, dim: 1);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var module in modules)
                {
                    module.Dispose();
                }
                foreach (var head in heads)
                {
                    head.Dispose();
                }
            }

            base.Dispose(disposing);
        }
    }

    public sealed class DQN2D : DQNNET
    {
        private readonly Module<Tensor, Tensor> conv1, flatten;
        private readonly ModuleList<Module<Tensor, Tensor>> fcModules = new();
        private readonly ModuleList<Module<Tensor, Tensor>> heads = new();
        private readonly int width;
        private long linear_input_size;

        private long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
        {
            return ((inputSize - kernelSize + 2 * padding) / stride) + 1;
        }

        public DQN2D(string name, long h, long w, int[] actionSizes, int width, int depth = 3, bool noisyLayers = false, float noiseScale = 0.0001f) : base(name)
        {
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");

            this.width = width;

            var smallestDim = Math.Min(h, w);
            var padding = smallestDim / 2;

            conv1 = Conv2d(1, width, kernelSize: (smallestDim, smallestDim), stride: (1, 1), padding: (padding, padding));

            long output_height = CalculateConvOutputSize(h, smallestDim, padding: padding);
            long output_width = CalculateConvOutputSize(w, smallestDim, padding: padding);

            linear_input_size = output_height * output_width * width;

            flatten = Flatten();

            fcModules.Add(noisyLayers ? new NoisyLinear(linear_input_size, width, std_init: noiseScale) : Linear(linear_input_size, width));

            for (int i = 1; i < depth; i++)
            {
                fcModules.Add(noisyLayers ? new NoisyLinear(width, width, std_init: noiseScale) : Linear(width, width));
            }

            foreach (var actionSize in actionSizes)
            {
                heads.Add(noisyLayers ? new NoisyLinear(width, actionSize, std_init: noiseScale) : Linear(width, actionSize));
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
            foreach (var module in fcModules)
            {
                x = functional.relu(module.forward(x));
            }

            var outputs = new List<Tensor>();
            foreach (var head in heads)
            {
                outputs.Add(head.forward(x));
            }

            return stack(outputs, dim: 1);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                conv1.Dispose();
                flatten.Dispose();
                foreach (var module in fcModules)
                {
                    module.Dispose();
                }
                foreach (var head in heads)
                {
                    head.Dispose();
                }
            }

            base.Dispose(disposing);
        }
    }
}