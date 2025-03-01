namespace RLMatrix;

public class PPOCriticNet2D : PPOCriticNet
{
    private readonly TensorModule _conv1;
    private readonly TensorModule _flatten;
    private readonly TensorModule _head;
    private readonly ModuleList<TensorModule> _fcModules = new();
    private readonly LSTM? _lstm; // GRU layer
    private readonly bool _useRnn;

    public PPOCriticNet2D(string name, long h, long w, int width, int depth = 3, bool useRNN = false) : base(name)
    {
        if (depth < 1) 
            throw new ArgumentOutOfRangeException(nameof(depth), "Depth must be 1 or greater.");

        _useRnn = useRNN;
        //_hiddenSize = width; // Assuming hidden size to be same as width for simplicity.

        var smallestDim = Math.Min(h, w);
        var padding = smallestDim / 2;

        // Convolutional layer to process 2D input.
        _conv1 = torch.nn.Conv2d(1, width, kernel_size: (smallestDim, smallestDim), stride: (1, 1), padding: (padding, padding));

        // Calculate input size for the fully connected layers after convolution and flattening.
        var outputHeight = CalculateConvOutputSize(h, smallestDim, stride: 1, padding: padding);
        var outputWidth = CalculateConvOutputSize(w, smallestDim, stride: 1, padding: padding);
        var linearInputSize = outputHeight * outputWidth * width;

        if (_useRnn)
        {
            // Initialize GRU layer if useRnn is true
            // hiddenSize = width
            _lstm = torch.nn.LSTM(linearInputSize, width, depth, batchFirst: true, dropout: 0.1f);
            linearInputSize = width; // The output of GRU layer is now the input for the fully connected layers.
        }

        _flatten = torch.nn.Flatten();

        // Define the fully connected layers.
        _fcModules.Add(torch.nn.Linear(linearInputSize, width));
        for (var i = 1; i < depth; i++)
        {
            _fcModules.Add(torch.nn.Linear(width, width));
        }

        // Final layer to produce the value estimate.
        _head = torch.nn.Linear(width, 1);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // Adjust for input dimensions.
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

        if (_useRnn && x.dim() == 4)
        {
            x = x.reshape(batchLength * sequenceLength, 1, x.size(2), x.size(3));
        }

        // Process through convolutional layer.
        x = torch.nn.functional.tanh(_conv1.forward(x));
        x = _flatten.forward(x);

        if (_useRnn)
        {
            x = x.reshape(batchLength, sequenceLength, x.size(1));
            x = _lstm!.forward(x).Item1;
            x = x.reshape(-1, x.size(2));
        }

        // Process through fully connected layers.
        foreach (var module in _fcModules)
        {
            x = torch.nn.functional.tanh(module.forward(x));
        }

        return _head.forward(x);
    }

    // TODO: unimplemented?
    public override Tensor forward(PackedSequence x)
    {
        throw new NotImplementedException();
    }
    
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            // Dispose all created modules.
            _conv1.Dispose();
            _flatten.Dispose();
            _lstm?.Dispose();
            
            foreach (var module in _fcModules)
            {
                module.Dispose();
            }
            
            _head.Dispose();
            ClearModules();
        }

        base.Dispose(disposing);
    }
    
    // Calculates the output size for convolutional layers.
    private static long CalculateConvOutputSize(long inputSize, long kernel_size, long stride = 1, long padding = 0)
    {
        return (inputSize - kernel_size + 2 * padding) / stride + 1;
    }
}