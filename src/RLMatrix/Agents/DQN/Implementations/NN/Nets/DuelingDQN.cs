namespace RLMatrix;

public sealed class DuelingDQN : DQNNET
{
    private readonly ModuleList<TensorModule> sharedModules = new();
    private readonly TensorModule valueHead;
    private readonly ModuleList<TensorModule> advantageHeads = new();

    public DuelingDQN(string name, int obsSize, int width, int[] actionSizes, int depth = 4, bool noisyLayers = false, float noiseScale = 0.0001f) : base(name)
    {

        if (obsSize < 1)
        {
            throw new ArgumentException("Number of observations can't be less than 1");
        }

        sharedModules.Add(noisyLayers ? new NoisyLinear(obsSize, width, initStandardDeviation: noiseScale) : torch.nn.Linear(obsSize, width));
        for (var i = 1; i < depth; i++)
        {
            sharedModules.Add(noisyLayers ? new NoisyLinear(width, width, initStandardDeviation: noiseScale) : torch.nn.Linear(width, width));
        }

        valueHead = noisyLayers ? new NoisyLinear(width, 1, initStandardDeviation: noiseScale) : torch.nn.Linear(width, 1);
        foreach (var actionSize in actionSizes)
        {
            advantageHeads.Add(noisyLayers ? new NoisyLinear(width, actionSize, initStandardDeviation: noiseScale) : torch.nn.Linear(width, actionSize));
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        if (x.dim() == 1)
        {
            x = x.unsqueeze(0);
        }

        foreach (var module in sharedModules)
        {
            x = torch.nn.functional.relu(module.forward(x));
        }

        var value = valueHead.forward(x).unsqueeze(1);

        var advantageList = new List<Tensor>();
        foreach (var head in advantageHeads)
        {
            var advantage = head.forward(x);
            advantage = advantage.unsqueeze(1);
            advantageList.Add(advantage);
        }

        var qValuesList = new List<Tensor>();
        foreach (var advantage in advantageList)
        {
            var qValues = value + (advantage - advantage.mean(dimensions: [2], keepdim: true));
            qValuesList.Add(qValues.squeeze(1));
        }

        var finalOutput = torch.stack(qValuesList, dim: 1);
        return finalOutput;
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

public sealed class DuelingDQN2D : DQNNET
{
    private readonly TensorModule _conv1;
    private readonly TensorModule _flatten;
    private readonly ModuleList<TensorModule> _sharedModules = new();
    private readonly TensorModule _valueHead;
    private readonly ModuleList<TensorModule> _advantageHeads = new();

    public DuelingDQN2D(string name, long h, long w, int[] actionSizes, int width, int depth = 3, bool noisyLayers = false, float noiseScale = 0.0001f) : base(name)
    {
        if (depth < 1) 
            throw new ArgumentOutOfRangeException(nameof(depth), "Depth must be 1 or greater.");

        var smallestDim = Math.Min(h, w);
        var padding = smallestDim / 2;

        _conv1 = torch.nn.Conv2d(1, width, kernel_size: (smallestDim, smallestDim), stride: (1, 1), padding: (padding, padding));

        var outputHeight = CalculateConvOutputSize(h, smallestDim, stride: 1, padding: padding);
        var outputWidth = CalculateConvOutputSize(w, smallestDim, stride: 1, padding: padding);
        var linearInputSize = outputHeight * outputWidth * width;
        _flatten = torch.nn.Flatten();

        _sharedModules.Add(noisyLayers ? new NoisyLinear(linearInputSize, width, initStandardDeviation: noiseScale) : torch.nn.Linear(linearInputSize, width));

        for (var i = 1; i < depth; i++)
        {
            _sharedModules.Add(noisyLayers ? new NoisyLinear(width, width, initStandardDeviation: noiseScale) : torch.nn.Linear(width, width));
        }

        _valueHead = noisyLayers ? new NoisyLinear(width, 1, initStandardDeviation: noiseScale) : torch.nn.Linear(width, 1);

        foreach (var actionSize in actionSizes)
        {
            _advantageHeads.Add(noisyLayers ? new NoisyLinear(width, actionSize, initStandardDeviation: noiseScale) : torch.nn.Linear(width, actionSize));
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

        foreach (var module in _sharedModules)
        {
            x = torch.nn.functional.relu(module.forward(x));
        }

        var value = _valueHead.forward(x).unsqueeze(1);

        var advantageList = new List<Tensor>();
        foreach (var head in _advantageHeads)
        {
            var advantage = head.forward(x).unsqueeze(1);
            advantageList.Add(advantage);
        }

        var qValuesList = new List<Tensor>();
        foreach (var advantage in advantageList)
        {
            var qValues = value + (advantage - advantage.mean(dimensions: [2], keepdim: true));
            qValuesList.Add(qValues.squeeze(1));
        }

        var finalOutput = torch.stack(qValuesList, dim: 1);
        return finalOutput;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _conv1.Dispose();
            _flatten.Dispose();
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

    private static long CalculateConvOutputSize(long inputSize, long kernel_size, long stride = 1, long padding = 0)
    {
        return (inputSize - kernel_size + 2 * padding) / stride + 1;
    }
}