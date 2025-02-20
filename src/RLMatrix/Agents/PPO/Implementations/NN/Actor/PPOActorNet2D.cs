using RLMatrix.Common;

namespace RLMatrix;

public class PPOActorNet2D : PPOActorNet
{
    private readonly TensorModule _conv1;
    private readonly TensorModule _flatten;
    private readonly ModuleList<TensorModule> _fcModules = new();
    private readonly ModuleList<TensorModule> _discreteHeads = new();
    private readonly ModuleList<TensorModule> _continuousHeadsMean = new();
    private readonly ModuleList<TensorModule> _continuousHeadsLogStd = new();
    private readonly LSTM? _lstm;
    private readonly bool _useRNN;
    private readonly int _headSize;

    public PPOActorNet2D(string name, long h, long w, int[] discreteActionSizes, ContinuousActionDimensions[] continuousActionDimensions, int width, int depth = 3, bool useRNN = false) 
        : base(name)
    {
        if (depth < 1) 
            throw new ArgumentOutOfRangeException(nameof(depth), "Depth must be 1 or greater.");

        //_width = width;
        _useRNN = useRNN;
        //_continuousActionDimensions = continuousActionDimensions;
        _headSize = discreteActionSizes.Length > 0 ? discreteActionSizes[0] : 1;

        var smallestDim = Math.Min(h, w);
        var padding = smallestDim / 2;

        _conv1 = torch.nn.Conv2d(1, width, kernel_size: (smallestDim, smallestDim), stride: (1, 1), padding: (padding, padding));

        var outputHeight = CalculateConvOutputSize(h, smallestDim, stride: 1, padding: padding);
        var outputWidth = CalculateConvOutputSize(w, smallestDim, stride: 1, padding: padding);
        var linearInputSize = outputHeight * outputWidth * width;

        _flatten = torch.nn.Flatten();
        if (useRNN)
        {
            _lstm = torch.nn.LSTM(linearInputSize, width, depth, batchFirst: true);
            _fcModules.Add(torch.nn.Linear(width, width));
        }
        else
        {
            _fcModules.Add(torch.nn.Linear(linearInputSize, width));
        }

        for (var i = 1; i < depth; i++)
        {
            _fcModules.Add(torch.nn.Linear(width, width));
        }

        foreach (var discreteActionSize in discreteActionSizes)
        {
            _discreteHeads.Add(torch.nn.Linear(width, discreteActionSize));
        }

        foreach (var _ in continuousActionDimensions)
        {
            _continuousHeadsMean.Add(torch.nn.Linear(width, 1));   // Assuming one output per continuous action dimension for mean
            _continuousHeadsLogStd.Add(torch.nn.Linear(width, 1)); // Assuming one output per continuous action dimension for log std deviation
        }

        RegisterComponents();
    }

    private Tensor ApplyHeads(Tensor x)
    {
        var outputs = new List<Tensor>();

        // Process discrete heads
        foreach (var head in _discreteHeads)
        {
            outputs.Add(torch.nn.functional.softmax(head.forward(x), 1));
        }

        // Process continuous heads (mean) and apply tanh to bound them within [-1, 1]
        for (var i = 0; i < _continuousHeadsMean.Count; i++)
        {
            var continuousOutput = _continuousHeadsMean[i].forward(x);  // Apply tanh to bound within [-1, 1]

            // Remap continuous means from [-1, 1] to the desired action bounds
            //   var low = continuousActionBounds[i].Item1;
            //   var high = continuousActionBounds[i].Item2;
            //  continuousOutput = (continuousOutput + 1) / 2 * (high - low) + low;  // Remap to [low, high]

            var paddedOutput = torch.zeros(continuousOutput.size(0), _headSize, device: x.device);
#if NET8_0_OR_GREATER
            paddedOutput[.., 0] = continuousOutput.squeeze(-1);
#else
            paddedOutput[GetIndices(paddedOutput.size(0)), 0] = continuousOutput.squeeze(-1);
#endif
            outputs.Add(paddedOutput);
        }

        // Process continuous heads (log std)
        for (var i = 0; i < _continuousHeadsLogStd.Count; i++)
        {
            var continuousOutput = _continuousHeadsLogStd[i].forward(x);
            var paddedOutput = torch.zeros(continuousOutput.size(0), _headSize, device: x.device);
#if NET8_0_OR_GREATER
            paddedOutput[.., 0] = continuousOutput.squeeze(-1);
#else
            paddedOutput[GetIndices(paddedOutput.size(0)), 0] = continuousOutput.squeeze(-1);
#endif
            outputs.Add(paddedOutput);
        }

        var result = torch.stack(outputs.ToArray(), dim: 1);
        return result;
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

        var batchLength = x.size(0);
        var sequenceLength = x.size(1);

        if (_useRNN && x.dim() == 4)
        {
            x = x.reshape(batchLength * sequenceLength, 1, x.size(2), x.size(3));
        }

        x = torch.nn.functional.tanh(_conv1.forward(x));
        x = _flatten.forward(x);
        if (_useRNN)
        {
            x = x.reshape(batchLength, sequenceLength, x.size(1));
            x = _lstm!.forward(x).Item1;
            x = x.reshape(-1, x.size(2));
        }

        foreach (var module in _fcModules)
        {
            x = torch.nn.functional.tanh(module.forward(x));
        }

        return ApplyHeads(x);
    }

    public override (Tensor, Tensor, Tensor) forward(Tensor x, Tensor? state, Tensor? state2)
    {
        if (x.dim() == 2)
        {
            x = x.unsqueeze(0).unsqueeze(0);
        }
        else if (x.dim() == 3)
        {
            x = x.unsqueeze(1);
        }
        
        x = torch.nn.functional.tanh(_conv1.forward(x));
        x = _flatten.forward(x);
        x = x.unsqueeze(0);

        (Tensor, Tensor)? stateTuple = null;
        if (state is not null && state2 is not null)
        {
            stateTuple = (state, state2);
        }

        var result = _lstm!.forward(x, stateTuple);
        x = result.Item1.squeeze(0);
        var stateRes = (result.Item2, result.Item3);

        foreach (var module in _fcModules)
        {
            x = torch.nn.functional.tanh(module.forward(x));
        }

        return (ApplyHeads(x), stateRes.Item1, stateRes.Item2);
    }

    // TODO: not implemented?
    public override Tensor forward(PackedSequence x)
    {
        throw new NotImplementedException();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _lstm?.Dispose();
            _conv1.Dispose();
            
            foreach (var module in _fcModules.Concat(_discreteHeads).Concat(_continuousHeadsMean).Concat(_continuousHeadsLogStd))
            {
                module.Dispose();
            }

            ClearModules();
        }

        base.Dispose(disposing);
    }
    
    private static long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
    {
        return (inputSize - kernelSize + 2 * padding) / stride + 1;
    }
}