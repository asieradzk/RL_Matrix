using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

public sealed class HierarchicalMultiHead : Module<Tensor, IEnumerable<Tensor>>
{
    private readonly ModuleList<Module<Tensor, Tensor>> heads = new();
    private readonly ModuleList<ModuleList<Module<Tensor, Tensor>>> hiddenLayers = new();
    private readonly int depth;

    public HierarchicalMultiHead(string name, int inputSize, int width, int[] actionSizes, int depth = 1) : base(name)
    {
        this.depth = depth;

        foreach (var actionSize in actionSizes)
        {
            heads.Add(Linear(width, actionSize));
        }

        for (int i = 0; i < actionSizes.Length; i++)
        {
            var hiddenLayersList = new ModuleList<Module<Tensor, Tensor>>();
            int hiddenLayerInputSize = inputSize;

            for (int j = 0; j < depth; j++)
            {
                if (j == 0)
                {
                    hiddenLayerInputSize += i * actionSizes[i];
                }
                hiddenLayersList.Add(Linear(hiddenLayerInputSize, width));
                hiddenLayerInputSize = width;
            }

            hiddenLayers.Add(hiddenLayersList);
        }

        RegisterComponents();
    }

    public IEnumerable<Tensor> Process(Tensor x)
    {
        var outputs = new List<Tensor>();

        for (int i = 0; i < heads.Count; i++)
        {
            var head = heads[i];
            var hiddenLayersList = hiddenLayers[i];
            var hiddenInput = x;

            for (int j = 0; j < depth; j++)
            {
                if (j == 0)
                {
                    hiddenInput = cat(new[] { hiddenInput }.Concat(outputs.Take(i)).ToList(), dim: 1);
                }
                var hiddenLayer = hiddenLayersList[j];
                hiddenInput = functional.relu(hiddenLayer.forward(hiddenInput));
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