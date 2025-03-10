using RLMatrix.Common;

namespace RLMatrix;

public class PPOActorNet1D : PPOActorNet
{
    private readonly ModuleList<TensorModule> _fcModules = new();
    private readonly ModuleList<TensorModule> _discreteHeads = new();
    private readonly ModuleList<TensorModule> _continuousHeadsMean = new();
    private readonly ModuleList<TensorModule> _continuousHeadsLogStd = new();
    private readonly LSTM? _lstmLayer;
    // private readonly int _hiddenSize; TODO: went unused outside of ctor
    private readonly bool _useRNN;
    private readonly int _headSize;
    // private readonly ContinuousActionDimensions[] _continuousActionBounds; TODO: went unused outside of ctor

    public PPOActorNet1D(string name, long inputs, int width, int[] discreteActions, ContinuousActionDimensions[] continuousActionBounds, int depth = 3, bool useRNN = false) 
        : base(name)
    {
        if (depth < 1)
            throw new ArgumentOutOfRangeException(nameof(depth), "Depth must be 1 or greater.");
        
        //_continuousActionBounds = continuousActionBounds;
        _useRNN = useRNN;
        //_hiddenSize = width;

        _headSize = discreteActions.Length > 0 ? discreteActions[0] : 1;

        if (_useRNN)
        {
            // Initialize LSTM layer if useRnn is true
            //_lstmLayer = torch.nn.LSTM(inputs, _hiddenSize, depth, batchFirst: true, dropout: 0.05f);
            _lstmLayer = torch.nn.LSTM(inputs, width, depth, batchFirst: true, dropout: 0.05f);
            // width = hiddenSize; // The output of LSTM layer is now the input for the heads
        }

        // Base layers
        if (_useRNN)
        {
            _fcModules.Add(torch.nn.Linear(width, width));
        }
        else
        {
            _fcModules.Add(torch.nn.Linear(inputs, width));
        }

        for (var i = 1; i < depth; i++)
        {
            _fcModules.Add(torch.nn.Linear(width, width));

        }

        // Discrete Heads
        foreach (var actionSize in discreteActions)
        {
            _discreteHeads.Add(torch.nn.Linear(width, actionSize));
        }

        // Continuous Heads for means and log std deviations
        foreach (var _ in continuousActionBounds)
        {
            _continuousHeadsMean.Add(torch.nn.Linear(width, 1));    // Assuming one output per continuous action dimension for mean
            _continuousHeadsLogStd.Add(torch.nn.Linear(width, 1));  // Assuming one output per continuous action dimension for log std deviation
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
            // var low = continuousActionBounds[i].Item1;
            // var high = continuousActionBounds[i].Item2;
            // continuousOutput = (continuousOutput + 1) / 2 * (high - low) + low;  // Remap to [low, high]

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
        if (x.dim() == 1)
        {
            x = x.unsqueeze(0);
        }

        if (_useRNN && x.dim() == 2)
        {
            x = x.unsqueeze(0);
        }

        // Apply the first fc module
        if (_useRNN)
        {
            var res = _lstmLayer!.forward(x); // not null if _useRnn == true
            x = res.Item1;
            x = x.reshape(-1, x.size(2));
        }

        x = torch.nn.functional.tanh(_fcModules.First().forward(x));

        // Apply the rest of the fc modules
        foreach (var module in _fcModules.Skip(1))
        {
            x = torch.nn.functional.tanh(module.forward(x));
        }

        var result = ApplyHeads(x);
        return result;
    }
    
    /*
    public override (Tensor, Tensor, Tensor) forward(Tensor x, Tensor? state, Tensor? state2)
    {
        if (x.dim() == 1)
        {
            x = x.unsqueeze(0);
        }

        // Create a PackedSequence with the input tensor
        var packedSequence = pack_sequence(new[] { x });

        (Tensor, Tensor)? stateTuple = null;
        if (state is not null && state2 is not null)
        {
            stateTuple = (state, state2);
        }

        // Call the LSTM layer with the PackedSequence and state tuple
        var res = lstmLayer.call(packedSequence, stateTuple);

        // Unpack the output PackedSequence
        var lstmOutput = pad_packed_sequence(res.Item1, batch_first: true);
        x = lstmOutput.Item1;

        x = x.squeeze(0);

        // Extract the hidden state and cell state from the LSTM output
        var hiddenState = res.Item2;
        var cellState = res.Item3;

        x = functional.tanh(fcModules.First().forward(x));

        // Apply the rest of the fc modules
        foreach (var module in fcModules.Skip(1))
        {
            x = functional.tanh(module.forward(x));
        }

        // Apply the heads
        var result = ApplyHeads(x);

        // Return the result tensor, hidden state, and cell state
        return (result, hiddenState, cellState);
    }
    */

    public override (Tensor, Tensor, Tensor) forward(Tensor x, Tensor? state, Tensor? state2)
    {
        if (x.dim() == 1)
        {
            x = x.unsqueeze(0);
        }

        /* Checked later anyways
        if (state is null)
        { }
        */

        x = x.unsqueeze(0);
        (Tensor, Tensor, Tensor) lstmResult;
        (Tensor, Tensor)? stateTuple = null;

        if (state is not null && state2 is not null)
        {
            stateTuple = (state, state2);
        }

        // TODO: Guard if _useRNN == true?
        
        lstmResult = _lstmLayer!.forward(x, stateTuple);
        x = lstmResult.Item1;
        var resultHiddenState = (lstmResult.Item2, lstmResult.Item3);

        x = x.squeeze(0);
        x = torch.nn.functional.tanh(_fcModules.First().forward(x));

        // Apply the rest of the fc modules
        foreach (var module in _fcModules.Skip(1))
        {
            x = torch.nn.functional.tanh(module.forward(x));
        }

        var result = ApplyHeads(x);
        return (result, resultHiddenState.Item1, resultHiddenState.Item2);
    }

    public override Tensor forward(PackedSequence x)
    {
        // Unpack the PackedSequence
        var unpackedData = x.data;
        //var batchSizes = x.batch_sizes; TODO: unused

        if (_useRNN)
        {
            (var lstmOutput, _, _) = _lstmLayer!.call(x);
            unpackedData = lstmOutput.data;
            unpackedData = unpackedData.reshape(-1, unpackedData.size(1));
        }
        else
        {
            // Adjust for a single input
            if (unpackedData.dim() == 1)
            {
                unpackedData = unpackedData.unsqueeze(0);
            }
            
            // TODO: this is identical to the above code. Maybe I'm missing something...
            if (unpackedData.dim() == 2)
            {
                unpackedData = unpackedData.unsqueeze(0);
            }
        }

        unpackedData = torch.nn.functional.tanh(_fcModules.First().forward(unpackedData));

        // Apply the rest of the fc modules
        foreach (var module in _fcModules.Skip(1))
        {
            unpackedData = torch.nn.functional.tanh(module.forward(unpackedData));
        }

        var result = ApplyHeads(unpackedData);

        return result;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            // Dispose modules in fcModules
            foreach (var module in _fcModules)
            {
                module.Dispose();
            }

            _lstmLayer?.Dispose();

            // Dispose discrete heads
            foreach (var head in _discreteHeads)
            {
                head.Dispose();
            }

            // Dispose continuous heads for mean values
            foreach (var head in _continuousHeadsMean)
            {
                head.Dispose();
            }

            // Dispose continuous heads for log standard deviation values
            foreach (var head in _continuousHeadsLogStd)
            {
                head.Dispose();
            }

            // Clear internal module list
            ClearModules();
        }

        base.Dispose(disposing);
    }
}