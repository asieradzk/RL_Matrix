using System;
using TorchSharp;
using TorchSharp.Modules;
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
        private readonly Module<Tensor, Tensor> head;

        /// <summary>
        /// Initializes a new instance of the DQN1D class.
        /// </summary>
        /// <param name="name">The name of the network.</param>
        /// <param name="obsSize">The size of the observation space, i.e., the number of inputs to the network.</param>
        /// <param name="width">The number of neurons in each hidden layer.</param>
        /// <param name="actionSize">The size of the action space, i.e., the number of outputs from the network.</param>
        /// <param name="depth">The number of hidden layers.</param>
        public DQN1D(string name, int obsSize, int width, int actionSize, int depth = 4) : base(name)
        {

            // First layer
            modules.Add(Linear(obsSize, width));

            // Additional depth-1 hidden layers
            for (int i = 1; i < depth; i++)
            {
                modules.Add(Linear(width, width));
            }

            head = Linear(width, actionSize);

            // Register the modules and head for the network
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

            return head.forward(x);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var module in modules)
                {
                    module.Dispose();
                }
                head.Dispose();
            }

            base.Dispose(disposing);
        }
    }




    public sealed class DQN2D : DQNNET
{
    private readonly Module<Tensor, Tensor> conv1, flatten, head;
    private readonly ModuleList<Module<Tensor, Tensor>> fcModules = new();
    private readonly int width;
    private long linear_input_size;

    private long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
    {
        return ((inputSize - kernelSize + 2 * padding) / stride) + 1;
    }

    public DQN2D(string name, long h, long w, long outputs, int width, int depth = 3) : base(name)
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

        return head.forward(x);
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
            head.Dispose();
        }

        base.Dispose(disposing);
    }
}

}