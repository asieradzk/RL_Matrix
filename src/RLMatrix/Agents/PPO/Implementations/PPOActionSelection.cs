﻿using System;
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
        public static int[] SelectDiscreteActionsFromProbs(Tensor result, int[] actionSize)
        {
            // Assuming discrete action heads come first
            List<int> actions = new List<int>();
            for (int i = 0; i < actionSize.Count(); i++)
            {
                var actionProbs = result.select(1, i);
                var action = torch.multinomial(actionProbs, 1);
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

        public static float[] SampleContinuousActions(Tensor result, int[] actionSize, (float, float)[] continousActions)
        {

            List<float> actions = new List<float>();
            int offset = actionSize.Count(); // Assuming discrete action heads come first
            for (int i = 0; i < continousActions.Count(); i++)
            {
                var mean = result.select(1, offset + i * 2).item<float>();
                var logStd = result.select(1, offset + i * 2 + 1).item<float>();
                var std = (float)Math.Exp(logStd);
                var actionValue = mean + std * (float)SampleFromStandardNormal(new Random());

                // Ensuring that action value stays within given bounds (assuming you have min and max values for each action)
                actionValue = Math.Clamp(actionValue, continousActions[i].Item1, continousActions[i].Item2);

                actions.Add(actionValue);
            }
            return actions.ToArray();
        }

        public static int[] SelectGreedyDiscreteActions(Tensor result, int[] actionSize)
        {
            List<int> actions = new List<int>();
            for (int i = 0; i < actionSize.Count(); i++)
            {
                var actionProbs = result.select(1, i);
                var action = actionProbs.argmax();
                actions.Add((int)action.item<long>());
            }
            return actions.ToArray();
        }

        public static float[] SelectMeanContinuousActions(Tensor result, int[] actionSize, (float, float)[] continousActions)
        {
            List<float> actions = new List<float>();
            int offset = actionSize.Count();
            for (int i = 0; i < continousActions.Count(); i++)
            {
                var mean = result.select(1, offset + i * 2).item<float>();
                actions.Add(mean);
            }
            return actions.ToArray();
        }


        #endregion
    }
}