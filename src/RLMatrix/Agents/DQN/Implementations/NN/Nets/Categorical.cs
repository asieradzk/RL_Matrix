using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace RLMatrix
{
    public sealed class CategoricalDQN1D : DQNNET
    {
        private readonly ModuleList<Module<Tensor, Tensor>> _modules = new ModuleList<Module<Tensor, Tensor>>();
        private readonly ModuleList<Module<Tensor, Tensor>> _heads = new ModuleList<Module<Tensor, Tensor>>();
        private readonly int _numAtoms;
        private readonly int[] _actionSizes;

        public CategoricalDQN1D(string name, int obsSize, int width, int[] actionSizes, int depth, int numAtoms, bool noisyLayers = false, float noiseScale = 0.01f) : base(name)
        {
            noiseScale *= 0.2f;

            if (obsSize < 1)
            {
                throw new ArgumentException("Number of observations can't be less than 1");
            }
            _numAtoms = numAtoms;
            _actionSizes = actionSizes;

            // Hidden layers configuration
            _modules.Add(noisyLayers ? new NoisyLinear(obsSize, width, std_init: noiseScale) : Linear(obsSize, width)); // First hidden layer
            for (int i = 1; i < depth; i++)
            {
                _modules.Add(noisyLayers ? new NoisyLinear(width, width, std_init: noiseScale) : Linear(width, width)); // Additional hidden layers
            }

            // Output heads for different action sizes
            foreach (var actionSize in actionSizes)
            {
                _heads.Add(noisyLayers ? new NoisyLinear(width, actionSize * _numAtoms, std_init: noiseScale) : Linear(width, actionSize * _numAtoms)); // C51 modifications for distributional outputs
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

            // Forward through hidden layers
            foreach (var module in _modules)
            {
                x = functional.relu(module.forward(x)); // After this, shape should still be [batch_size, feature_size]
            }

            // Forward through each output head
            var qDistributionsList = new List<Tensor>();
            foreach (var head in _heads)
            {
                var qDistributions = head.forward(x); // Expected shape [batch_size, actionSize * _numAtoms]
                int numActions = _actionSizes[0];
                qDistributions = qDistributions.view(-1, numActions, _numAtoms); // Change to [batch_size, numActions, _numAtoms]
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
                foreach (var module in _modules)
                {
                    module.Dispose();
                }
                foreach (var head in _heads)
                {
                    head.Dispose();
                }
            }
            base.Dispose(disposing);
        }
    }

    public sealed class CategoricalDQN2D : DQNNET
    {
        private readonly Module<Tensor, Tensor> _conv1, _flatten;
        private readonly ModuleList<Module<Tensor, Tensor>> _fcModules = new ModuleList<Module<Tensor, Tensor>>();
        private readonly ModuleList<Module<Tensor, Tensor>> _heads = new ModuleList<Module<Tensor, Tensor>>();
        private readonly int _width;
        private long _linearInputSize;
        private readonly int _numAtoms;
        private readonly int[] _actionSizes;

        public CategoricalDQN2D(string name, long h, long w, int[] actionSizes, int width, int depth, int numAtoms, bool noisyLayers = false, float noiseScale = 0.01f) : base(name)
        {
            noiseScale *= 0.2f;
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");
            _width = width;
            _numAtoms = numAtoms;
            _actionSizes = actionSizes;

            var smallestDim = Math.Min(h, w);
            var padding = smallestDim / 2;

            _conv1 = Conv2d(1, _width, kernelSize: (smallestDim, smallestDim), stride: (1, 1), padding: (padding, padding));

            long outputHeight = CalculateConvOutputSize(h, smallestDim, stride: 1, padding: padding);
            long outputWidth = CalculateConvOutputSize(w, smallestDim, stride: 1, padding: padding);
            _linearInputSize = outputHeight * outputWidth * _width;
            _flatten = Flatten();

            // First FC layer (always present)
            _fcModules.Add(noisyLayers ? new NoisyLinear(_linearInputSize, _width, std_init: noiseScale) : Linear(_linearInputSize, _width));

            // Additional depth-1 FC layers
            for (int i = 1; i < depth; i++)
            {
                _fcModules.Add(noisyLayers ? new NoisyLinear(_width, _width, std_init: noiseScale) : Linear(_width, _width));
            }

            // Output heads for different action sizes
            foreach (var actionSize in actionSizes)
            {
                _heads.Add(noisyLayers ? new NoisyLinear(_width, actionSize * _numAtoms, std_init: noiseScale) : Linear(_width, actionSize * _numAtoms));
            }

            RegisterComponents();
        }

        private long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
        {
            return ((inputSize - kernelSize + 2 * padding) / stride) + 1;
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

            x = functional.relu(_conv1.forward(x));
            x = _flatten.forward(x);

            foreach (var module in _fcModules)
            {
                x = functional.relu(module.forward(x));
            }

            // Forward through each output head
            var qDistributionsList = new List<Tensor>();
            foreach (var head in _heads)
            {
                var qDistributions = head.forward(x); // Expected shape [batch_size, actionSize * _numAtoms]
                int numActions = _actionSizes[0];
                qDistributions = qDistributions.view(-1, numActions, _numAtoms); // Change to [batch_size, numActions, _numAtoms]
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
                _conv1.Dispose();
                _flatten.Dispose();
                foreach (var module in _fcModules)
                {
                    module.Dispose();
                }
                foreach (var head in _heads)
                {
                    head.Dispose();
                }
            }
            base.Dispose(disposing);
        }


    }
}