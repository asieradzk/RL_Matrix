using System;
using System.Collections.Generic;
using TorchSharp.Modules;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace RLMatrix
{
    public sealed class DuelingDQNoisy : DQNNET
    {
        private readonly ModuleList<Module<Tensor, Tensor>> sharedModules = new ModuleList<Module<Tensor, Tensor>>();
        private readonly Module<Tensor, Tensor> valueHead;
        private readonly ModuleList<Module<Tensor, Tensor>> advantageHeads = new ModuleList<Module<Tensor, Tensor>>();

        public DuelingDQNoisy(string name, int obsSize, int width, int[] actionSizes, int depth = 4) : base(name)
        {
            if (obsSize < 1)
            {
                throw new ArgumentException("Number of observations can't be less than 1");
            }

            // Shared layers configuration
            sharedModules.Add(new NoisyLinear(obsSize, width)); // First shared layer
            for (int i = 1; i < depth; i++)
            {
                sharedModules.Add(new NoisyLinear(width, width)); // Additional shared layers
            }

            // Separate value stream (only one head because state value doesn't depend on action)
            valueHead = Linear(width, 1);

            // Separate advantage streams (multiple heads for different action sizes)
            foreach (var actionSize in actionSizes)
            {
                advantageHeads.Add(new NoisyLinear(width, actionSize));
            }

            // Register the modules for the network
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            if (x.dim() == 1)
            {
                x = x.unsqueeze(0);
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

                advantage = advantage.unsqueeze(1); // This changes shape to [batch_size, 1, actionSize] for proper broadcasting with value
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

            // Stack along a new dimension for the heads, should result in [batch_size, num_heads, actionSize]
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

}
