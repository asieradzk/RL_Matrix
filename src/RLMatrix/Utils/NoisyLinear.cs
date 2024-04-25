using System;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.Modules
{
    public class NoisyLinear : torch.nn.Module<Tensor, Tensor>
    {
        private Linear _linear;
        private Tensor _weight_epsilon;
        private Tensor _bias_epsilon;
        private float _std_init;
        private Device _device;

        public NoisyLinear(long in_features, long out_features, bool bias = true, Device? device = null, ScalarType? dtype = null, float std_init = 0.0001f)
            : base(nameof(NoisyLinear))
        {
            _std_init = std_init;
            _device = device ?? (cuda.is_available() ? CUDA : CPU);
            _linear = nn.Linear(in_features, out_features, bias, _device, dtype);

            var factorizedNoiseShape = new long[] { out_features, 1 };
            _weight_epsilon = torch.randn(factorizedNoiseShape, device: _device, dtype: dtype) * _std_init;
            if (bias)
            {
                _bias_epsilon = torch.randn(out_features, device: _device, dtype: dtype) * _std_init;
            }
        }

        public override Tensor forward(Tensor input)
        {
            if (this.training)
            {
                var noisy_weight = _linear.weight + _weight_epsilon.expand(_linear.weight.shape);
                var noisy_bias = _linear.bias + _bias_epsilon;
                return nn.functional.linear(input, noisy_weight, noisy_bias);
            }
            else
            {
                return _linear.forward(input);
            }
        }

        public void ResetNoise()
        {
            _weight_epsilon.normal_(0, _std_init);
            if (_bias_epsilon is not null)
            {
                _bias_epsilon.normal_(0, _std_init);
            }
        }
    }
}