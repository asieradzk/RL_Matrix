using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLMatrix.Agents.DQN.NN
{
    using System;
    using System.Collections.Generic;
    using System.Runtime.ExceptionServices;
    using TorchSharp;
    using TorchSharp.Modules;
    using static TorchSharp.torch;
    using static TorchSharp.torch.nn;

    namespace RLMatrix
    {
        public sealed class HDQN : DQNNET
        {
            private readonly ModuleList<Module<Tensor, Tensor>> modules = new();
            private readonly ModuleList<Module<Tensor, Tensor>> heads = new();
            private readonly ModuleList<Module<Tensor, Tensor>> hiddenLayers = new();

            public HDQN(string name, int obsSize, int width, int[] actionSizes, int depth = 4) : base(name)
            {
                if (obsSize < 1)
                {
                    throw new ArgumentException("Number of observations can't be less than 1");
                }

                // First layer
                modules.Add(Linear(obsSize, width));

                // Additional depth-1 hidden layers
                for (int i = 1; i < depth; i++)
                {
                    modules.Add(Linear(width, width));
                }

                // Multiple heads for multi-head actions
                foreach (var actionSize in actionSizes)
                {
                    heads.Add(Linear(width, actionSize));
                }

                //Hidden layers for hierarchical processing
                //these go before heads
                int inputSize = width;
                for (int i = 0; i < actionSizes.Length; i++)
                {
                    hiddenLayers.Add(Linear(inputSize + i * actionSizes[i], width));
                    inputSize = width;
                }

                // Register the modules, hidden layers, and heads for the network
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
                

                for (int i = 0; i < heads.Count; i++)
                {
                    var head = heads[i];
                    var hiddenLayer = hiddenLayers[i];
                    var hiddenInput = i == 0 ? x : cat(new[] { x }.Concat(outputs.Take(i)).ToList(), dim: 1);
                    var hiddenOutput = functional.relu(hiddenLayer.forward(hiddenInput));
                    outputs.Add(head.forward(hiddenOutput));
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
                    foreach (var hiddenLayer in hiddenLayers)
                    {
                        hiddenLayer.Dispose();
                    }
                }

                base.Dispose(disposing);
            }
        }


        public sealed class HDuelingDQN : DQNNET
        {
            private readonly ModuleList<Module<Tensor, Tensor>> sharedModules = new ModuleList<Module<Tensor, Tensor>>();
            private readonly Module<Tensor, Tensor> valueHead;
            private readonly Module<Tensor, IEnumerable<Tensor>> hierarchicalMultiHead;


            public HDuelingDQN(string name, int obsSize, int width, int[] actionSizes, int depth = 4) : base(name)
            {
                if (obsSize < 1)
                {
                    throw new ArgumentException("Number of observations can't be less than 1");
                }

                //hidden layers for hierarchical processing
               
                // Shared layers configuration
                sharedModules.Add(Linear(obsSize, width)); // First shared layer
                for (int i = 1; i < depth - 1; i++)
                {
                    sharedModules.Add(Linear(width, width)); // Additional shared layers
                }
                sharedModules.Add(Linear(width, width)); // Last shared layer

                // Separate value stream (only one head because state value doesn't depend on action)
                valueHead = Linear(width, 1);
                // Separate advantage streams (multiple heads for different action sizes)
               hierarchicalMultiHead = new HierarchicalMultiHead("hmultihead", width, width/2, actionSizes, 4);

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
                var advantageList = hierarchicalMultiHead.forward(x);

                // Combine value and advantages to get Q values for each head/action size
                var qValuesList = new List<Tensor>();


                foreach (var advantage in advantageList)
                {
                    var adv = advantage.unsqueeze(1); 
                    // Broadcast value across advantage's actions, should align shapes
                    var qValues = value + (adv - adv.mean(dimensions: new long[] { 2 }, keepdim: true)); // Subtract mean along the correct dimension
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
                    hierarchicalMultiHead.Dispose();
                }
                base.Dispose(disposing);
            }
        }
    }
}
