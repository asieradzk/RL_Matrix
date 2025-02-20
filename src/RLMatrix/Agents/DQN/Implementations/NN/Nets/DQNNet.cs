namespace RLMatrix;

public abstract class DQNNET(string name) : TensorModule(name)
{
    public abstract override Tensor forward(Tensor x);
}

public sealed class DQN1D : DQNNET
{
    private readonly ModuleList<TensorModule> _modules = new();
    private readonly ModuleList<TensorModule> _heads = new();

    public DQN1D(string name, int obsSize, int width, int[] actionSizes, int depth = 4, bool noisyLayers = false, float noiseScale = 0.0001f) 
        : base(name)
    {
        if (obsSize < 1)
        {
            throw new ArgumentException("Number of observations can't be less than 1");
        }

        _modules.Add(noisyLayers ? new NoisyLinear(obsSize, width, initStandardDeviation: noiseScale) : torch.nn.Linear(obsSize, width));

        for (var i = 1; i < depth; i++)
        {
            _modules.Add(noisyLayers ? new NoisyLinear(width, width, initStandardDeviation: noiseScale) : torch.nn.Linear(width, width));
        }

        foreach (var actionSize in actionSizes)
        {
            _heads.Add(noisyLayers ? new NoisyLinear(width, actionSize, initStandardDeviation: noiseScale) : torch.nn.Linear(width, actionSize));
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        if (x.dim() == 1)
        {
            x = x.unsqueeze(0);
        }

        foreach (var module in _modules)
        {
            x = torch.nn.functional.relu(module.forward(x));
        }

        var outputs = new List<Tensor>();
        foreach (var head in _heads)
        {
            outputs.Add(head.forward(x));
        }

        return torch.stack(outputs, dim: 1);
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

public sealed class DQN2D : DQNNET
{
    private readonly TensorModule _conv1;
    private readonly TensorModule _flatten;
    private readonly ModuleList<TensorModule> _fcModules = new();
    private readonly ModuleList<TensorModule> _heads = new();

    public DQN2D(string name, long h, long w, int[] actionSizes, int width, int depth = 3, bool noisyLayers = false, float noiseScale = 0.0001f) : base(name)
    {
        if (depth < 1) 
            throw new ArgumentOutOfRangeException(nameof(depth), "Depth must be 1 or greater.");

        var smallestDim = Math.Min(h, w);
        var padding = smallestDim / 2;

        _conv1 = torch.nn.Conv2d(1, width, kernel_size: (smallestDim, smallestDim), stride: (1, 1), padding: (padding, padding));

        var outputHeight = CalculateConvOutputSize(h, smallestDim, padding: padding);
        var outputWidth = CalculateConvOutputSize(w, smallestDim, padding: padding);

        var linearInputSize = outputHeight * outputWidth * width;

        _flatten = torch.nn.Flatten();

        _fcModules.Add(noisyLayers ? new NoisyLinear(linearInputSize, width, initStandardDeviation: noiseScale) : torch.nn.Linear(linearInputSize, width));

        for (var i = 1; i < depth; i++)
        {
            _fcModules.Add(noisyLayers ? new NoisyLinear(width, width, initStandardDeviation: noiseScale) : torch.nn.Linear(width, width));
        }

        foreach (var actionSize in actionSizes)
        {
            _heads.Add(noisyLayers ? new NoisyLinear(width, actionSize, initStandardDeviation: noiseScale) : torch.nn.Linear(width, actionSize));
        }

        RegisterComponents();
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
        x = torch.nn.functional.relu(_conv1.forward(x));
        x = _flatten.forward(x);
        foreach (var module in _fcModules)
        {
            x = torch.nn.functional.relu(module.forward(x));
        }

        var outputs = new List<Tensor>();
        foreach (var head in _heads)
        {
            outputs.Add(head.forward(x));
        }

        return torch.stack(outputs, dim: 1);
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
    
    private static long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
    {
        return (inputSize - kernelSize + 2 * padding) / stride + 1;
    }
}