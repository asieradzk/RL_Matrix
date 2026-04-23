using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;

namespace RLMatrix.Agents.PPO.Implementations
{
    public static class PPOActionSelection<T>
    {

        #region helpers
        public static int[] SelectDiscreteActionsFromProbs(Tensor result, int[] actionSize, int[][] maskForState = null)
        {
            // Assuming discrete action heads come first
            List<int> actions = new List<int>();
            for (int i = 0; i < actionSize.Count(); i++)
            {
                var actionProbs = result.select(1, i);
                if (maskForState != null)
                {
                    var mask = maskForState[i];
                    if (mask != null)
                    {
                        using var _ = torch.NewDisposeScope();
                        var maskTensor = torch.tensor(mask.Select(v => v == 0 ? 0f : 1f).ToArray(), dtype: ScalarType.Float32, device: actionProbs.device);
                        var masked = actionProbs * maskTensor;
                        var sum = masked.sum();
                        if (sum.item<float>() <= 0f)
                        {
                            // Fallback: if no valid actions (or masked to zero), use uniform over allowed indices; if none allowed, uniform over all
                            int allowed = mask.Count(v => v != 0);
                            Tensor fallback = allowed > 0
                                ? (maskTensor / allowed).unsqueeze(0)
                                : torch.full_like(actionProbs, 1f / actionProbs.size(1));
                            actionProbs = fallback.MoveToOuterDisposeScope();
                        }
                        else
                        {
                            actionProbs = (masked / (sum + 1e-12f)).MoveToOuterDisposeScope();
                        }
                    }
                }
                // Numeric stability: ensure strictly positive, renormalize
                actionProbs = torch.clamp(actionProbs, 1e-10f, 1.0f);
                actionProbs = actionProbs / actionProbs.sum();
                var action = torch.multinomial(actionProbs, 1, true);
                actions.Add((int)action.item<long>());
            }
            return actions.ToArray();
        }
        public static double SampleFromStandardNormal(Random random)
        {
            double u1 = 1.0 - random.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                   Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return randStdNormal;
        }


        public static float[] SampleContinuousActions(Tensor result, int[] actionSize, (float min, float max)[] continuousActions)
        {
            List<float> actions = new List<float>();
            int discreteHeads = actionSize.Length;
            int continuousHeads = continuousActions.Length;
            for (int i = 0; i < continuousHeads; i++)
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

        public static int[] SelectGreedyDiscreteActions(Tensor result, int[] actionSize, int[][] maskForState = null)
        {
            List<int> actions = new List<int>();
            for (int i = 0; i < actionSize.Count(); i++)
            {
                var actionProbs = result.select(1, i);
                if (maskForState != null)
                {
                    var mask = maskForState[i];
                    if (mask != null)
                    {
                        using var _ = torch.NewDisposeScope();
                        var maskTensor = torch.tensor(mask.Select(v => v == 0 ? 0f : 1f).ToArray(), dtype: ScalarType.Float32, device: actionProbs.device);
                        var masked = actionProbs * maskTensor;
                        var sum = masked.sum();
                        if (sum.item<float>() <= 0f)
                        {
                            int firstAllowed = Array.FindIndex(mask, v => v != 0);
                            actions.Add(firstAllowed >= 0 ? firstAllowed : 0);
                            continue;
                        }
                        actionProbs = (masked / (sum + 1e-12f)).MoveToOuterDisposeScope();
                    }
                }
                var action = actionProbs.argmax();
                actions.Add((int)action.item<long>());
            }
            return actions.ToArray();
        }

        public static float[] SelectMeanContinuousActions(Tensor result, int[] actionSize, (float min, float max)[] continuousActions)
        {
            List<float> actions = new List<float>();
            int discreteHeads = actionSize.Count();
            int continuousHeads = continuousActions.Length;
            for (int i = 0; i < continuousHeads; i++)
            {
                var mean = result[0, discreteHeads + i, 0].item<float>();
                actions.Add(mean);
            }
            return actions.ToArray();
        }

        #endregion
    }
}
