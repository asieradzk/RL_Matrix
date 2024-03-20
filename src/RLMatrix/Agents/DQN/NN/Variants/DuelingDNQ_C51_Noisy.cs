using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace RLMatrix
{
    public sealed class DuelingDQN_C51_Noisy : DQNNET
    {
        private readonly ModuleList<Module<Tensor, Tensor>> _sharedModules = new ModuleList<Module<Tensor, Tensor>>();
        private readonly Module<Tensor, Tensor> _valueHead;
        private readonly ModuleList<Module<Tensor, Tensor>> _advantageHeads = new ModuleList<Module<Tensor, Tensor>>();
        private readonly int _numAtoms;
        private readonly int[] _actionSizes;

        public DuelingDQN_C51_Noisy(string name, int obsSize, int width, int[] actionSizes, int depth, int numAtoms) : base(name)
        {
            var device = torch.cuda_is_available() ? torch.CUDA : torch.CPU;

            if (obsSize < 1)
            {
                throw new ArgumentException("Number of observations can't be less than 1");
            }
            _numAtoms = numAtoms;
            _actionSizes = actionSizes;

            // Shared layers configuration
            _sharedModules.Add(new NoisyLinear(obsSize, width)); // First shared layer
            for (int i = 1; i < depth; i++)
            {
                _sharedModules.Add(new NoisyLinear(width, width)); // Additional shared layers
            }
         
            // Separate value stream
            _valueHead = Linear(width, _numAtoms, device: device); // C51 modifies this to output a distribution


            // Separate advantage streams for different action sizes
            foreach (var actionSize in actionSizes)
            {
                _advantageHeads.Add(new NoisyLinear(width, actionSize * _numAtoms)); // C51 modifications for distributional outputs
            }

       
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            if (x.dim() == 1)
            {
                x = x.unsqueeze(0); // Ensure input is at least 2D, now [batch_size, feature_size]
            }

            // Forward through shared layers
            foreach (var module in _sharedModules)
            {
                x = functional.relu(module.forward(x)); // After this, shape should still be [batch_size, feature_size]
            }

            // Forward through value stream
            var value = _valueHead.forward(x); // Shape [batch_size, 1, _numAtoms] for C51
            value = value.view(-1, 1, _numAtoms);
            // Forward through each advantage stream (head)
            var advantageList = new List<Tensor>();
            foreach (var head in _advantageHeads)
            {
                var advantage = head.forward(x); // Expected shape [batch_size, actionSize * _numAtoms]
                int numActions = _actionSizes[0];
                advantage = advantage.view(-1, numActions, _numAtoms); // Change to [batch_size, numActions, _numAtoms]
                advantageList.Add(advantage);
            }

            // Combine value and advantages using dueling architecture
            var qDistributionsList = new List<Tensor>();
            foreach (var advantage in advantageList)
            {
                var qDistributions = value + (advantage - advantage.mean(dimensions: new long[] { 1 }, keepdim: true)); // Dueling architecture
                qDistributions = functional.softmax(qDistributions, dim: -1); // Apply softmax to normalize the distributions
                qDistributionsList.Add(qDistributions); // Now [batch_size, numActions, _numAtoms]
            }

            // Stack along a new dimension for the heads
            var finalOutput = torch.stack(qDistributionsList, dim: 1); // [batch_size, num_heads, numActions, _numAtoms]
            return finalOutput;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var module in _sharedModules)
                {
                    module.Dispose();
                }
                _valueHead.Dispose();
                foreach (var head in _advantageHeads)
                {
                    head.Dispose();
                }
            }
            base.Dispose(disposing);
        }
    }
}
