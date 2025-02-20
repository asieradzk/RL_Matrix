namespace RLMatrix;

public sealed class CategoricalDuelingDQN1D : DQNNET
{
    private readonly ModuleList<TensorModule> _sharedModules = new();
    private readonly TensorModule _valueHead;
    private readonly ModuleList<TensorModule> _advantageHeads = new();
    private readonly int _numAtoms;
    private readonly int[] _actionSizes;

    public CategoricalDuelingDQN1D(string name, int obsSize, int width, int[] actionSizes, int depth, int numAtoms, bool noisyLayers = false, float noiseScale = 0.01f) : base(name)
    {
        noiseScale *= 0.2f;

        if (obsSize < 1)
        {
            throw new ArgumentException("Number of observations can't be less than 1");
        }
        
        _numAtoms = numAtoms;
        _actionSizes = actionSizes;

        // Shared layers configuration
        _sharedModules.Add(noisyLayers ? new NoisyLinear(obsSize, width, initStandardDeviation: noiseScale) : torch.nn.Linear(obsSize, width)); // First shared layer
        for (var i = 1; i < depth; i++)
        {
            _sharedModules.Add(noisyLayers ? new NoisyLinear(width, width, initStandardDeviation: noiseScale) : torch.nn.Linear(width, width)); // Additional shared layers
        }

        // Separate value stream
        _valueHead = noisyLayers ? new NoisyLinear(width, _numAtoms, initStandardDeviation: noiseScale) : torch.nn.Linear(width, _numAtoms); // C51 modifies this to output a distribution

        // Separate advantage streams for different action sizes
        foreach (var actionSize in actionSizes)
        {
            _advantageHeads.Add(noisyLayers ? new NoisyLinear(width, actionSize * _numAtoms, initStandardDeviation: noiseScale) : torch.nn.Linear(width, actionSize * _numAtoms)); // C51 modifications for distributional outputs
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

        // Forward through shared layers
        foreach (var module in _sharedModules)
        {
            x = torch.nn.functional.relu(module.forward(x)); // After this, shape should still be [batch_size, feature_size]
        }

        // Forward through value stream
        var value = _valueHead.forward(x); // Shape [batch_size, 1, _numAtoms] for C51
        value = value.view(-1, 1, _numAtoms);

        // Forward through each advantage stream (head)
        var advantageList = new List<Tensor>();
        foreach (var head in _advantageHeads)
        {
            var advantage = head.forward(x); // Expected shape [batch_size, actionSize * _numAtoms]
            var numActions = _actionSizes[0];
            advantage = advantage.view(-1, numActions, _numAtoms); // Change to [batch_size, numActions, _numAtoms]
            advantageList.Add(advantage);
        }

        // Combine value and advantages using dueling architecture
        var qDistributionsList = new List<Tensor>();
        foreach (var advantage in advantageList)
        {
            var qDistributions = value + (advantage - advantage.mean(dimensions: new long[] { 1 }, keepdim: true)); // Dueling architecture
            qDistributions = torch.nn.functional.softmax(qDistributions, dim: -1); // Apply softmax to normalize the distributions
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

public sealed class CategoricalDuelingDQN2D : DQNNET
{
    private readonly TensorModule _conv1, _flatten;
    private readonly ModuleList<TensorModule> _sharedModules = new();
    private readonly TensorModule _valueHead;
    private readonly ModuleList<TensorModule> _advantageHeads = new();
    private readonly int _numAtoms;
    private readonly int[] _actionSizes;

    private long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
    {
        return ((inputSize - kernelSize + 2 * padding) / stride) + 1;
    }

    public CategoricalDuelingDQN2D(string name, long h, long w, int[] actionSizes, int width, int depth, int numAtoms, bool noisyLayers = false, float noiseScale = 0.01f) : base(name)
    {
        if (depth < 1) 
            throw new ArgumentOutOfRangeException(nameof(depth), "Depth must be 1 or greater.");
        
        noiseScale *= 0.2f;
        _numAtoms = numAtoms;
        _actionSizes = actionSizes;

        var smallestDim = Math.Min(h, w);
        var padding = smallestDim / 2;

        _conv1 = torch.nn.Conv2d(1, width, kernel_size: (smallestDim, smallestDim), stride: (1, 1), padding: (padding, padding));

        var outputHeight = CalculateConvOutputSize(h, smallestDim, stride: 1, padding: padding);
        var outputWidth = CalculateConvOutputSize(w, smallestDim, stride: 1, padding: padding);
        var linearInputSize = outputHeight * outputWidth * width;
        _flatten = torch.nn.Flatten();

        // First FC layer (always present)
        _sharedModules.Add(noisyLayers ? new NoisyLinear(linearInputSize, width, initStandardDeviation: noiseScale) : torch.nn.Linear(linearInputSize, width));

        // Additional depth-1 FC layers
        for (var i = 1; i < depth; i++)
        {
            _sharedModules.Add(noisyLayers ? new NoisyLinear(width, width, initStandardDeviation: noiseScale) : torch.nn.Linear(width, width));
        }

        // Separate value stream
        _valueHead = noisyLayers ? new NoisyLinear(width, _numAtoms, initStandardDeviation: noiseScale) : torch.nn.Linear(width, _numAtoms);

        // Separate advantage streams for different action sizes
        foreach (var actionSize in actionSizes)
        {
            _advantageHeads.Add(noisyLayers ? new NoisyLinear(width, actionSize * _numAtoms, initStandardDeviation: noiseScale) : torch.nn.Linear(width, actionSize * _numAtoms));
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

        // Forward through value stream
        var value = _valueHead.forward(x); // Shape [batch_size, 1, _numAtoms] for C51
        value = value.view(-1, 1, _numAtoms);

        // Forward through each advantage stream (head)
        var advantageList = new List<Tensor>();
        foreach (var head in _advantageHeads)
        {
            var advantage = head.forward(x); // Expected shape [batch_size, actionSize * _numAtoms]
            var numActions = _actionSizes[0];
            advantage = advantage.view(-1, numActions, _numAtoms); // Change to [batch_size, numActions, _numAtoms]
            advantageList.Add(advantage);
        }

        // Combine value and advantages using dueling architecture
        var qDistributionsList = new List<Tensor>();
        foreach (var advantage in advantageList)
        {
            var qDistributions = value + (advantage - advantage.mean(dimensions: new long[] { 1 }, keepdim: true)); // Dueling architecture
            qDistributions = torch.nn.functional.softmax(qDistributions, dim: -1); // Apply softmax to normalize the distributions
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