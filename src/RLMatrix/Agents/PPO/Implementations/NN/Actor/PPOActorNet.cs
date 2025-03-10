namespace RLMatrix;

public abstract class PPOActorNet(string name) : TensorModule(name)
{
    public abstract override Tensor forward(Tensor x);
    
    public abstract (Tensor, Tensor, Tensor) forward(Tensor x, Tensor? state, Tensor? state2);
    
    public abstract Tensor forward(PackedSequence x);

#if !NET8_0_OR_GREATER
    public static long[] GetIndices(long length)
    {
        if (length < 0)
            throw new ArgumentOutOfRangeException(nameof(length), "Length must be non-negative.");

        var indices = new long[length];
        for (var i = 0; i < length; i++)
        {
            indices[i] = i;
        }
        
        return indices;
    }
#endif

    public Tensor get_log_prob<TTensorState>(TTensorState state, Tensor actions, int discreteActions, int contActions)
    {
        var logits = state switch
        {
            Tensor st => forward(st),
            PackedSequence st => forward(st),
            _ => throw new ArgumentException("Invalid state type.")
        };

        var discreteLogProbabilities = new List<Tensor>();
        var continuousLogProbabilities = new List<Tensor>();

        // Discrete action log probabilities
        for (var i = 0; i < discreteActions; i++)
        {
            using var actionLogits = logits.select(1, i);
            using var actionTaken = actions.select(1, i).to(ScalarType.Int64).unsqueeze(-1);
            
            var softMaxRes = torch.nn.functional.log_softmax(actionLogits, dim: 1).gather(dim: 1, index: actionTaken);
            discreteLogProbabilities.Add(softMaxRes);
        }

        // Continuous action log probabilities
        for (var i = 0; i < contActions; i++)
        {
#if NET8_0_OR_GREATER
            using var mean = logits[.., discreteActions + i, 0].unsqueeze(1);
            using var logStd = logits[.., discreteActions + contActions + i, 0].unsqueeze(1);
            using var actionTaken = actions.select(1, discreteActions + i).unsqueeze(1);
#else
            using var mean = logits[GetIndices(logits.size(0)), discreteActions + i, 0].unsqueeze(1);
            using var logStd = logits[GetIndices(logits.size(0)), discreteActions + contActions + i, 0].unsqueeze(1);
            using var actionTaken = actions.select(1, discreteActions + i).unsqueeze(1);
#endif
            var std = torch.exp(logStd);
            var diff = actionTaken - mean;
            var squaredDiff = torch.pow(diff / std, 2);
            var logProbability = -0.5f * squaredDiff - logStd - 0.5f * (float)Math.Log(2 * Math.PI);
            continuousLogProbabilities.Add(logProbability);
        }

        // Combine discrete and continuous log probabilities
        using var combinedLogProbabilities = torch.cat(discreteLogProbabilities.Concat(continuousLogProbabilities).ToArray(), dim: 1);
        
        var squeezeResult = combinedLogProbabilities.squeeze();
        return squeezeResult;
    }

    public (Tensor LogProbabilities, Tensor Entropy) get_log_prob_entropy<TTensorState>(TTensorState state, Tensor actions, int discreteActions, int contActions)
    {
        var logits = state switch
        {
            Tensor st => forward(st),
            PackedSequence st => forward(st),
            _ => throw new ArgumentException("Invalid state type.")
        };

        var discreteLogProbabilities = new List<Tensor>();
        var continuousLogProbabilities = new List<Tensor>();
        var discreteEntropies = new List<Tensor>();
        var continuousEntropies = new List<Tensor>();

        // Discrete action log probabilities and entropy
        for (var i = 0; i < discreteActions; i++)
        {
            using var actionLogits = logits.select(1, i);
            using var actionProbabilities = torch.nn.functional.softmax(actionLogits, dim: 1);
            using var actionTaken = actions.select(1, i).to(ScalarType.Int64).unsqueeze(-1);
            
            discreteLogProbabilities.Add(torch.log(actionProbabilities + 1e-10).gather(dim: 1, index: actionTaken));
            discreteEntropies.Add(-(actionProbabilities * torch.log(actionProbabilities + 1e-10)).sum(1, keepdim: true));
        }

        // Continuous action log probabilities and entropy
        for (var i = 0; i < contActions; i++)
        {
#if NET8_0_OR_GREATER
            using var mean = logits[.., discreteActions + i, 0].unsqueeze(1);
            using var logStd = logits[.., discreteActions + contActions + i, 0].unsqueeze(1);
            using var actionTaken = actions.select(1, discreteActions + i).unsqueeze(1);
#else
            using var mean = logits[GetIndices(logits.size(0)), discreteActions + i, 0].unsqueeze(1);
            using var logStd = logits[GetIndices(logits.size(0)), discreteActions + contActions + i, 0].unsqueeze(1);
            using var actionTaken = actions.select(1, discreteActions + i).unsqueeze(1);
#endif
            var std = torch.exp(logStd);
            var diff = actionTaken - mean;
            var squaredDiff = torch.pow(diff / std, 2);
            var logProbability = -0.5f * squaredDiff - logStd - 0.5f * (float)Math.Log(2 * Math.PI);
            continuousLogProbabilities.Add(logProbability);
            var entropy = logStd + 0.5f * (1 + (float)Math.Log(2 * Math.PI));
            continuousEntropies.Add(entropy);
        }

        // Combine discrete and continuous log probabilities and entropies
        using var combinedLogProbabilities = torch.cat(discreteLogProbabilities.Concat(continuousLogProbabilities).ToArray(), dim: 1);
        using var combinedEntropies = torch.cat(discreteEntropies.Concat(continuousEntropies).ToArray(), dim: 1);
        
        var logProbabilities = combinedLogProbabilities.squeeze();
        var totalEntropy = combinedEntropies.mean([1], true);
        return (logProbabilities, totalEntropy);
    }
}