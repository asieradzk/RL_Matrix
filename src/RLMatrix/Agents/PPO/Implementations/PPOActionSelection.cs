using RLMatrix.Common;

namespace RLMatrix;

public static class PPOActionSelection
{
    public static int[] SelectDiscreteActionsFromProbabilities(Tensor result, int[] actionSize)
    {
        // Assuming discrete action heads come first
        var actions = new List<int>();
        for (var i = 0; i < actionSize.Length; i++)
        {
            var actionProbabilities = result.select(1, i);
            var action = torch.multinomial(actionProbabilities, 1);
            
            actions.Add((int)action.item<long>());
        }
        
        return actions.ToArray();
    }
    
    public static double SampleFromStandardNormal(Random random)
    {
        var u1 = 1.0d - random.NextDouble(); //uniform(0,1] random doubles
        var u2 = 1.0d - random.NextDouble();
        var randStdNormal = Math.Sqrt(-2.0d * Math.Log(u1)) *
                               Math.Sin(2.0d * Math.PI * u2); //random normal(0,1)
        return randStdNormal;
    }


    public static float[] SampleContinuousActions(Tensor result, int[] actionSize, ContinuousActionDimensions[] continuousActions)
    {
        var actions = new List<float>();
        var discreteHeads = actionSize.Length;
        var continuousHeads = continuousActions.Length;
        
        for (var i = 0; i < continuousHeads; i++)
        {
            var mean = result[0, discreteHeads + i, 0].item<float>();
            var logStd = result[0, discreteHeads + continuousHeads + i, 0].item<float>();
            var std = (float)Math.Exp(logStd);
            var noise = (float)SampleFromStandardNormal(new Random());
            var action = mean + std * noise;  
            
            actions.Add(action);
        }
        
        return actions.ToArray();
    }

    public static int[] SelectGreedyDiscreteActions(Tensor result, int[] actionSize)
    {
        var actions = new List<int>();
        
        for (var i = 0; i < actionSize.Length; i++)
        {
            var actionProbabilities = result.select(1, i);
            var action = actionProbabilities.argmax();
            
            actions.Add((int)action.item<long>());
        }
        
        return actions.ToArray();
    }

    public static float[] SelectMeanContinuousActions(Tensor result, int[] actionSize, ContinuousActionDimensions[] continuousActions)
    {
        var actions = new List<float>();
        var discreteHeads = actionSize.Length;
        var continuousHeads = continuousActions.Length;
        
        for (var i = 0; i < continuousHeads; i++)
        {
            var mean = result[0, discreteHeads + i, 0].item<float>();
            actions.Add(mean);
        }
        
        return actions.ToArray();
    }
}