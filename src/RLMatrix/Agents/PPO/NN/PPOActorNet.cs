using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;
using TorchSharp.Modules;

namespace RLMatrix
{
    public abstract class PPOActorNet : Module<Tensor, Tensor>
    {
        public PPOActorNet(string name) : base(name)
        {
        }

        public override abstract Tensor forward(Tensor x);

        public Tensor get_log_prob(Tensor states, Tensor actions)
        {
            Tensor logits = forward(states);
            actions = actions.to(ScalarType.Int64);
            return torch.nn.functional.log_softmax(logits, dim: 1).gather(dim: 1, index: actions);
        }

        public Tensor get_entropy(Tensor states)
        {
            Tensor logits = forward(states);
            Tensor probabilities = torch.nn.functional.softmax(logits, dim: 1);
            return -(probabilities * torch.nn.functional.log_softmax(logits, dim: 1)).sum(dim: 1);
        }
    }

    public class PPOActorNet1D : PPOActorNet
    {
        private readonly ModuleList<Module<Tensor, Tensor>> fcModules = new();
        private readonly Module<Tensor, Tensor> head;

        public PPOActorNet1D(string name, long inputs, long outputs, int width, int depth = 3) : base(name)
        {
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");

            // First FC layer (always present)
            fcModules.Add(Linear(inputs, width));

            // Additional depth-1 FC layers
            for (int i = 1; i < depth; i++)
            {
                fcModules.Add(Linear(width, width));
            }

            head = Linear(width, outputs);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            if (x.dim() == 1)
            {
                x = x.unsqueeze(0);
            }

            foreach (var module in fcModules)
            {
                x = functional.relu(module.forward(x));
            }

            return functional.softmax(head.forward(x), 1);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var module in fcModules)
                {
                    module.Dispose();
                }
                head.Dispose();
            }

            base.Dispose(disposing);
        }
    }


    public class PPOActorNet2D : PPOActorNet
    {
        private Module<Tensor, Tensor> conv1, flatten, head;
        private readonly ModuleList<Module<Tensor, Tensor>> fcModules = new();
        private int width;
        private long linear_input_size;

        public long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
        {
            return ((inputSize - kernelSize + 2 * padding) / stride) + 1;
        }

        public PPOActorNet2D(string name, long h, long w, long outputs, int width, int depth = 3) : base(name)
        {
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");

            this.width = width;

            var smallestDim = Math.Min(h, w);

            conv1 = Conv2d(1, width, kernelSize: (smallestDim, smallestDim), stride: (1, 1));

            long output_height = CalculateConvOutputSize(h, smallestDim); // output size of conv1
            long output_width = CalculateConvOutputSize(w, smallestDim);
            linear_input_size = output_height * output_width * width;

            flatten = Flatten();

            // First FC layer (always present)
            fcModules.Add(Linear(linear_input_size, width));

            // Additional depth-1 FC layers
            for (int i = 1; i < depth; i++)
            {
                fcModules.Add(Linear(width, width));
            }

            head = Linear(width, outputs);

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

            return functional.softmax(head.forward(x), 1);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                conv1.Dispose();
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
