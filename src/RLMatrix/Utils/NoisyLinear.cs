using System;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.Modules
{
    public class NoisyLinear : torch.nn.Module<Tensor, Tensor>
    {
        private Linear _linear;
        private Tensor _weight_sigma;
        private Tensor _weight_epsilon;
        private float _std_init;
        private Device _device;

        public NoisyLinear(long in_features, long out_features, bool bias = true, Device? device = null, ScalarType? dtype = null, float std_init = 0.025f)
            : base(nameof(NoisyLinear))
        {
            _std_init = std_init;
            _device = device ?? (cuda.is_available() ? CUDA : CPU);
            _linear = nn.Linear(in_features, out_features, bias, _device, dtype);

            _weight_sigma = Parameter(torch.full(_linear.weight.shape, std_init, device: _device, dtype: dtype));
            _weight_epsilon = torch.empty_like(_linear.weight, device: _device, dtype: dtype);

            ResetNoise();
        }

        public override Tensor forward(Tensor input)
        {
            if (this.training)
            {
                var noisy_weight = _linear.weight + _weight_sigma * _weight_epsilon;
                return nn.functional.linear(input, noisy_weight, _linear.bias);
            }
            else
            {
                return _linear.forward(input);
            }
        }

        public void ResetNoise()
        {
            _weight_epsilon.copy_(ScaleNoise(_linear.weight.shape));
        }

        private Tensor ScaleNoise(long[] size)
        {
            var x = torch.randn(size, device: _device);
            return x.sign() * x.abs().sqrt();
        }
    }
}