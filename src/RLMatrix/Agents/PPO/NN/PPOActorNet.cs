using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;

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
        private Module<Tensor, Tensor> fc1, fc2, fc3, head;

        public PPOActorNet1D(string name, long inputs, long outputs, int width) : base(name)
        {
            fc1 = Linear(inputs, width);
            fc2 = Linear(width, width);
            fc3 = Linear(width, width);
            head = Linear(width, outputs);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            if (x.dim() == 1)
            {
                x = x.unsqueeze(0);
            }

            x = functional.relu(fc1.forward(x));
            x = functional.relu(fc2.forward(x));
            x = functional.relu(fc3.forward(x));
            return functional.softmax(head.forward(x), 1);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                fc1.Dispose();
                fc2.Dispose();
                fc3.Dispose();
                head.Dispose();
                ClearModules();
            }

            base.Dispose(disposing);
        }
    }

    public class PPOActorNet2D : PPOActorNet
    {
        private Module<Tensor, Tensor> conv1, flatten, fc1, fc2, fc3, head;
        private int width;
        private long linear_input_size;

        public long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
        {
            return ((inputSize - kernelSize + 2 * padding) / stride) + 1;
        }

        public PPOActorNet2D(string name, long h, long w, long outputs, int width) : base(name)
        {
            this.width = width;

            conv1 = Conv2d(1, width, kernelSize: (3, 3), stride: (1, 1));

            long output_height = CalculateConvOutputSize(h, 3); // output size of conv1
            long output_width = CalculateConvOutputSize(w, 3);

            linear_input_size = output_height * output_width * width;

            flatten = Flatten();
            fc1 = Linear(linear_input_size, width);
            fc2 = Linear(width, width);
            fc3 = Linear(width, width);
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
            x = functional.relu(fc1.forward(x));
            x = functional.relu(fc2.forward(x));
            x = functional.relu(fc3.forward(x));
            return functional.softmax(head.forward(x), 1);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                conv1.Dispose();
                fc1.Dispose();
                fc2.Dispose();
                fc3.Dispose();
                head.Dispose();
                ClearModules();
            }

            base.Dispose(disposing);
        }
    }
}
