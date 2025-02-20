namespace RLMatrix;

public sealed class HierarchicalMultiHead : TensorEnumerableModule
{
    private readonly ModuleList<TensorModule> heads = new();
    private readonly ModuleList<ModuleList<TensorModule>> hiddenLayers = new();
    private readonly int depth;

    public HierarchicalMultiHead(string name, int inputSize, int width, int[] actionSizes, int depth = 1) : base(name)
    {
        this.depth = depth;

        foreach (var actionSize in actionSizes)
        {
            heads.Add(torch.nn.Linear(width, actionSize));
        }

        for (var i = 0; i < actionSizes.Length; i++)
        {
            var hiddenLayersList = new ModuleList<TensorModule>();
            var hiddenLayerInputSize = inputSize;

            for (var j = 0; j < depth; j++)
            {
                if (j == 0)
                {
                    hiddenLayerInputSize += i * actionSizes[i];
                }
                hiddenLayersList.Add(torch.nn.Linear(hiddenLayerInputSize, width));
                hiddenLayerInputSize = width;
            }

            hiddenLayers.Add(hiddenLayersList);
        }

        RegisterComponents();
    }

    public IEnumerable<Tensor> Process(Tensor x)
    {
        var outputs = new List<Tensor>();

        for (var i = 0; i < heads.Count; i++)
        {
            var head = heads[i];
            var hiddenLayersList = hiddenLayers[i];
            var hiddenInput = x;

            for (var j = 0; j < depth; j++)
            {
                if (j == 0)
                {
                    hiddenInput = torch.cat(new[] { hiddenInput }.Concat(outputs.Take(i)).ToList(), dim: 1);
                }
                var hiddenLayer = hiddenLayersList[j];
                hiddenInput = torch.nn.functional.relu(hiddenLayer.forward(hiddenInput));
            }

            outputs.Add(head.forward(hiddenInput));
        }

        return outputs;
    }

    public override IEnumerable<Tensor> forward(Tensor x)
    {
        return Process(x);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            foreach (var head in heads)
            {
                head.Dispose();
            }
            foreach (var hiddenLayersList in hiddenLayers)
            {
                foreach (var hiddenLayer in hiddenLayersList)
                {
                    hiddenLayer.Dispose();
                }
            }
        }

        base.Dispose(disposing);
    }
}