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



        public Tensor get_log_prob(Tensor states, Tensor actions, int discreteActions, int contActions)
        {
            Tensor logits = forward(states);

            var discreteLogProbs = new List<Tensor>();
            var continuousLogProbs = new List<Tensor>();
            
            int numDiscreteActions = discreteActions;
            for (int i = 0; i < numDiscreteActions; i++)
            {
                Tensor actionLogits = logits.select(1, i);
                Tensor actionTaken = actions.select(1, i).to(ScalarType.Int64).unsqueeze(-1);
                discreteLogProbs.Add(torch.nn.functional.log_softmax(actionLogits, dim: 1).gather(dim: 1, index: actionTaken));
            }

            int numContinuousActions = contActions; // assuming continuousStds and continuousMeans have the same count
            for (int i = 0; i < numContinuousActions; i++)
            {
                Tensor mean = logits.select(1, numDiscreteActions + i);
                Tensor log_std = logits.select(1, numDiscreteActions + numContinuousActions + i);
                Tensor actionTaken = actions.select(1, numDiscreteActions + i);
                Tensor log_prob = (-0.5 * torch.pow((actionTaken - mean) / torch.exp(log_std), 2) - log_std - 0.5 * Math.Log(2 * Math.PI)).sum(1, keepdim: true);
                continuousLogProbs.Add(log_prob);
            }

            return torch.cat(discreteLogProbs.Concat(continuousLogProbs).ToArray(), dim: 1);
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
        private readonly ModuleList<Module<Tensor, Tensor>> discreteHeads = new();
        private readonly ModuleList<Module<Tensor, Tensor>> continuousHeadsMean = new();
        private readonly ModuleList<Module<Tensor, Tensor>> continuousHeadsLogStd = new();

        public PPOActorNet1D(string name, long inputs, int width, int[] discreteActions, (float, float)[] continuousActionBounds, int depth = 3) : base(name)
        {
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");

            // Base layers
            fcModules.Add(Linear(inputs, width));
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

            // Apply discrete heads and continuous heads
            var outputs = new List<Tensor>();
            foreach (var head in discreteHeads)
            {
                outputs.Add(functional.softmax(head.forward(x), 1));
            }
            foreach (var head in continuousHeadsMean)
            {
                outputs.Add(head.forward(x));  // mean values
            }
            foreach (var head in continuousHeadsLogStd)
            {
                outputs.Add(functional.softplus(head.forward(x)));  // Convert to log std deviation values
            }

            return stack(outputs, dim: 1);
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
        private readonly ModuleList<Module<Tensor, Tensor>> continuousMeans = new();
        private readonly ModuleList<Module<Tensor, Tensor>> continuousStds = new();
        private int width;
        private long linear_input_size;

        public long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
        {
            return ((inputSize - kernelSize + 2 * padding) / stride) + 1;
        }

        public PPOActorNet2D(string name, long h, long w, int[] discreteActionSizes, (float, float)[] continuousActionBounds, int width, int depth = 3) : base(name)
        {
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");

            this.width = width;

            var smallestDim = Math.Min(h, w);

            conv1 = Conv2d(1, width, kernelSize: (smallestDim, smallestDim), stride: (1, 1));

            long output_height = CalculateConvOutputSize(h, smallestDim);
            long output_width = CalculateConvOutputSize(w, smallestDim);
            linear_input_size = output_height * output_width * width;

            flatten = Flatten();

            fcModules.Add(Linear(linear_input_size, width));
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
                continuousMeans.Add(Linear(width, 1));  // Assuming each continuous action is 1-dimensional
                continuousStds.Add(Linear(width, 1));
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

            var discreteOutputs = new List<Tensor>();
            var continuousOutputs = new List<Tensor>();

            foreach (var head in discreteHeads)
            {
                discreteOutputs.Add(functional.softmax(head.forward(x), 1));
            }

            foreach (var meanModule in continuousMeans)
            {
                continuousOutputs.Add(meanModule.forward(x));
            }

            foreach (var stdModule in continuousStds)
            {
                continuousOutputs.Add(functional.softplus(stdModule.forward(x)));
            }

            return stack(discreteOutputs.Concat(continuousOutputs).ToArray(), dim: 1);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                conv1.Dispose();
                foreach (var module in fcModules.Concat(discreteHeads).Concat(continuousMeans).Concat(continuousStds))
                {
                    module.Dispose();
                }
                ClearModules();
            }

            base.Dispose(disposing);
        }
    }


}
