using System;
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
        private readonly Module<Tensor, Tensor> fc1, fc2, fc3;
        private readonly Module<Tensor, Tensor> head;

        /// <summary>
        /// Initializes a new instance of the DQN1D class.
        /// </summary>
        /// <param name="name">The name of the network.</param>
        /// <param name="obsSize">The size of the observation space, i.e., the number of inputs to the network.</param>
        /// <param name="width">The number of neurons in each hidden layer.</param>
        /// <param name="actionSize">The size of the action space, i.e., the number of outputs from the network.</param>
        public DQN1D(string name, int obsSize, int width, int actionSize) : base(name)
        {
            fc1 = Linear(obsSize, width);
            fc2 = Linear(width, width);
            fc3 = Linear(width, width);
            head = Linear(width, actionSize);

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
            return head.forward(x);
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



    public sealed class DQN2D : DQNNET
    {
        private Module<Tensor, Tensor> conv1, flatten, fc1, fc2, fc3;
        private readonly Module<Tensor, Tensor> head;
        private readonly int width;

        private long linear_input_size;

        private long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
        {
            return ((inputSize - kernelSize + 2 * padding) / stride) + 1;
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="DQN2D"/> class.
        /// </summary>
        /// <param name="name">The name of the network.</param>
        /// <param name="h">The height of the input tensor.</param>
        /// <param name="w">The width of the input tensor.</param>
        /// <param name="outputs">The number of outputs.</param>
        /// <param name="width">The width of the hidden layers. This parameter specifies the number of neurons in each hidden layer.</param>
        /// <exception cref="ArgumentNullException">Thrown when the 'name' parameter is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the 'h', 'w', 'outputs' or 'width' parameters are less than or equal to zero.</exception>
        public DQN2D(string name, long h, long w, long outputs, int width) : base(name)
        {
            this.width = width;
            var smallestDim = Math.Min(h, w);

            conv1 = Conv2d(1, width, kernelSize: (smallestDim, smallestDim), stride: (1, 1));

            long output_height = CalculateConvOutputSize(h, smallestDim); // output size of conv1
            long output_width = CalculateConvOutputSize(w, smallestDim);

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
            return head.forward(x);
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