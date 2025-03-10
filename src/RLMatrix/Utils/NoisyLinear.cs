namespace RLMatrix;

public class NoisyLinear : TensorModule
{
    private readonly Linear _linear;
    private readonly Tensor _epsilonWeight;
    private readonly Tensor? _epsilonBias;
    private readonly float _initStandardDeviation;

    public NoisyLinear(long inFeatures, long outFeatures, bool bias = true, Device? device = null, ScalarType? dType = null, float initStandardDeviation = 0.0001f)
        : base(nameof(NoisyLinear))
    {
        _initStandardDeviation = initStandardDeviation;
         device ??= torch.cuda.is_available() ? torch.CUDA : torch.CPU;
        _linear = torch.nn.Linear(inFeatures, outFeatures, bias, device, dType);

        var factorizedNoiseShape = new[] { outFeatures, 1 };
        _epsilonWeight = torch.randn(factorizedNoiseShape, device: device, dtype: dType) * _initStandardDeviation;
        
        if (bias)
        {
            _epsilonBias = torch.randn(outFeatures, device: device, dtype: dType) * _initStandardDeviation;
        }
    }

    public override Tensor forward(Tensor input)
    {
        if (training)
        {
            var noisyWeight = _linear.weight + _epsilonWeight.expand(_linear.weight.shape);
            var noisyBias = _linear.bias! + _epsilonBias!;
            return torch.nn.functional.linear(input, noisyWeight, noisyBias);
        }
        
        return _linear.forward(input);
    }

    public void ResetNoise()
    {
        _epsilonWeight.normal_(0, _initStandardDeviation);
        _epsilonBias?.normal_(0, _initStandardDeviation);
    }
}