namespace RLMatrix;

public class PPOCriticNet1D : PPOCriticNet
{
    private readonly ModuleList<TensorModule> _fcModules = new();
    private readonly TensorModule _head;
    private readonly LSTM? _lstmLayer;
    private readonly bool _useRnn;

    public PPOCriticNet1D(string name, long inputs, int width, int depth = 3, bool useRNN = false) : base(name)
    {
        // Ensure depth is at least 1.
        if (depth < 1) 
            throw new ArgumentOutOfRangeException(nameof(width), "Depth must be 1 or greater.");

        _useRnn = useRNN;

        if (_useRnn)
        {
            // Initialize LSTM layer if useRnn is true
            // hiddenSize = width
            _lstmLayer = torch.nn.LSTM(inputs, width, depth, batchFirst: true, dropout: 0.05f);
        }

        // Base layers
        if (_useRnn)
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

        // Final layer to produce the value estimate.
        _head = torch.nn.Linear(width, 1);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // Adjust for a single input.
        if (x.dim() == 1)
        {
            x = x.unsqueeze(0);
        }

        if (_useRnn && x.dim() == 2)
        {
            x = x.unsqueeze(0);
        }


        // Apply the first fc module
        if (_useRnn)
        {
            // Apply LSTM layer if useRnn is true
            x = _lstmLayer!.forward(x).Item1;
            x = x.reshape(x.size(0) * x.size(1), x.size(2));
        }

        x = torch.nn.functional.tanh(_fcModules.First().forward(x));


        foreach (var module in _fcModules.Skip(1))
        {
            x = torch.nn.functional.tanh(module.forward(x));
        }

        var result = _head.forward(x);
        return result;
    }

    public override Tensor forward(PackedSequence x)
    {
        // Unpack the PackedSequence
        var unpackedData = x.data;
        //var batchSizes = x.batch_sizes; // TODO: unused

        if (_useRnn)
        {
            // Apply LSTM layer if useRnn is true
            var (lstmOutput, _, _) = _lstmLayer!.call(x);
            unpackedData = lstmOutput.data;
            unpackedData = unpackedData.reshape(unpackedData.size(0), unpackedData.size(1));
        }
        else
        {
            // Adjust for a single input
            if (unpackedData.dim() == 1)
            {
                unpackedData = unpackedData.unsqueeze(0);
            }
            
            if (unpackedData.dim() == 2)
            {
                unpackedData = unpackedData.unsqueeze(0);
            }
        }

        unpackedData = torch.nn.functional.tanh(_fcModules.First().forward(unpackedData));

        foreach (var module in _fcModules.Skip(1))
        {
            unpackedData = torch.nn.functional.tanh(module.forward(unpackedData));
        }

        var result = _head.forward(unpackedData);

        return result;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            foreach (var module in _fcModules)
            {
                module.Dispose();
            }
            
            _lstmLayer?.Dispose();
            _head.Dispose();
        }

        base.Dispose(disposing);
    }
}