using RLMatrix.Common;

namespace RLMatrix;

public abstract class GAILNET(string name) : TensorModule(name)
{
    public abstract override Tensor forward(Tensor x);
}


public class GAILDiscriminator1D : GAILNET
{
    private readonly ModuleList<TensorModule> _fcLayers = new();

    public GAILDiscriminator1D(string name, long stateSize, int[] discreteActionDimensions, ContinuousActionDimensions[] continuousActionDimensions, int width = 512, int depth = 3) : base(name)
    {
        if (depth < 1) 
            throw new ArgumentOutOfRangeException(nameof(depth), "Depth must be 1 or greater.");

        // Calculate total action size (sum of discrete action sizes and continuous action bounds)
        var totalActionSize = discreteActionDimensions.Length + continuousActionDimensions.Length;
        // First layer takes state and action sizes as input
        _fcLayers.Add(torch.nn.Linear(stateSize + totalActionSize, width));
        for (var i = 1; i < depth; i++)
        {
            _fcLayers.Add(torch.nn.Linear(width, width));
        }

        // Output layer
        _fcLayers.Add(torch.nn.Linear(width, 1)); // Output is a single scalar value representing 'expertness'

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // Pass through fully connected layers
        for (var i = 0; i < _fcLayers.Count - 1; i++)
        {
            x = torch.nn.functional.relu(_fcLayers[i].forward(x));
        }

        // Apply the last layer without ReLU activation, then apply sigmoid
        // No need to use squeeze() since we want the output shape to be [batchSize, 1]
        return torch.nn.functional.sigmoid(_fcLayers.Last().forward(x)).squeeze(1);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            foreach (var layer in _fcLayers)
            {
                layer.Dispose();
            }

            ClearModules();
        }

        base.Dispose(disposing);
    }
}

public class GAILDiscriminator2D : GAILNET
{
    private readonly TensorModule _conv1;
    private readonly TensorModule _flatten;
    private readonly ModuleList<TensorModule> _fcLayers = new();

    public GAILDiscriminator2D(string name, long h, long w, int[] discreteActionDimensions, ContinuousActionDimensions[] continuousActionDimensions, int width = 128, int depth = 3) : base(name)
    {
        if (depth < 1) 
            throw new ArgumentOutOfRangeException(nameof(depth), "Depth must be 1 or greater.");

        // Calculate total action size (sum of discrete action sizes and continuous action bounds)
        var totalActionSize = discreteActionDimensions.Length + continuousActionDimensions.Length;

        var smallestDim = Math.Min(h, w);

        _conv1 = torch.nn.Conv2d(1, width, kernel_size: (smallestDim, smallestDim), stride: (1, 1));

        var outputHeight = CalculateConvOutputSize(h, smallestDim);
        var outputWidth = CalculateConvOutputSize(w, smallestDim);
        var linearInputSize = outputHeight * outputWidth * width;

        _flatten = torch.nn.Flatten();

        // First FC layer takes flattened conv output and action sizes as input
        _fcLayers.Add(torch.nn.Linear(linearInputSize + totalActionSize, width));
        for (var i = 1; i < depth; i++)
        {
            _fcLayers.Add(torch.nn.Linear(width, width));
        }

        // Output layer
        _fcLayers.Add(torch.nn.Linear(width, 1)); // Output is a single scalar value representing 'expertness'

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = torch.nn.functional.relu(_conv1.forward(x));
        x = _flatten.forward(x);

        foreach (var module in _fcLayers)
        {
            x = torch.nn.functional.relu(module.forward(x));
        }

        // Sigmoid activation for output
        return torch.nn.functional.sigmoid(_fcLayers.Last().forward(x)).squeeze();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _conv1.Dispose();
            _flatten.Dispose();
            foreach (var module in _fcLayers)
            {
                module.Dispose();
            }
        }

        base.Dispose(disposing);
    }
    
    private static long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
    {
        return (inputSize - kernelSize + 2 * padding) / stride + 1;
    }
}