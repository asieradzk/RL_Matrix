using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using RLMatrix;

namespace RLMatrix
{
    /// <summary>
    /// Default implementation of PPONetProviderBase, makes network with 3 hidden layers
    /// </summary>
    /// <typeparam name="T">Type of observation space 1D or 2D, for 2D a simple conv network is created</typeparam>
    public class PPONetProviderBase<T> : IPPONetProvider<T>
    {
        int neuronsPerLayer;
        int depth = 2;
        bool useRNN;

        /// <summary>
        /// Default constructor for simple PPO NNs - crtic and actor.
        /// </summary>
        /// <param name="neuronsPerLayer">Number of neurons in hidden layers, default 256 x 3</param>
        public PPONetProviderBase(int neuronsPerLayer = 256, int depth = 2, bool useRNN = false)
        {
            this.neuronsPerLayer = neuronsPerLayer;
            this.depth = depth;
            this.useRNN = useRNN;
        }


        public PPOActorNet CreateActorNet(IEnvironment<T> env)
        {
            switch (typeof(T))
            {
                case Type t when t == typeof(float[]):
                    var obsSize = env.stateSize.Match<int>(
                        intSize => intSize,
                        tupleSize => throw new Exception("Unexpected 2D observation dimension for 1D state"));
                    var actionSize = env.actionSize;
                    return new PPOActorNet1D("1DDQN", obsSize, neuronsPerLayer, actionSize, new (float, float)[0], depth, useRNN);
                case Type t when t == typeof(float[,]):
                    var obsSize2D = env.stateSize.Match<(int, int)>(
                        intSize => throw new Exception("Unexpected 1D observation dimension for 2D state"),
                        tupleSize => tupleSize);
                    var actionSize2 = env.actionSize;
                    return new PPOActorNet2D("2DDQN", obsSize2D.Item1, obsSize2D.Item2, actionSize2, new (float, float)[0], neuronsPerLayer , depth, useRNN);
                default:
                    throw new Exception("Unexpected type");
            }
        }

        public PPOActorNet CreateActorNet(IContinuousEnvironment<T> env)
        {
            switch (typeof(T))
            {
                case Type t when t == typeof(float[]):
                    var obsSize = env.stateSize.Match<int>(
                        intSize => intSize,
                        tupleSize => throw new Exception("Unexpected 2D observation dimension for 1D state"));
                    var actionSize = env.actionSize;
                    var continuousActionBounds = env.continuousActionBounds;
                    return new PPOActorNet1D("1DDQN", obsSize, neuronsPerLayer, actionSize, continuousActionBounds, depth, useRNN);
                case Type t when t == typeof(float[,]):
                    var obsSize2D = env.stateSize.Match<(int, int)>(
                        intSize => throw new Exception("Unexpected 1D observation dimension for 2D state"),
                        tupleSize => tupleSize);
                    var actionSize2 = env.actionSize;
                    var continuousActionBounds2 = env.continuousActionBounds;
                    return new PPOActorNet2D("2DDQN", obsSize2D.Item1, obsSize2D.Item2, actionSize2, continuousActionBounds2, neuronsPerLayer, depth, useRNN);
                default:
                    throw new Exception("Unexpected type");
            }
        }

        public PPOCriticNet CreateCriticNet(IEnvironment<T> env)
        {
            switch (typeof(T))
            {
                case Type t when t == typeof(float[]):
                    var obsSize = env.stateSize.Match<int>(
                        intSize => intSize,
                        tupleSize => throw new Exception("Unexpected 2D observation dimension for 1D state"));
                    return new PPOCriticNet1D("1DDQN", obsSize, neuronsPerLayer, depth, useRNN);
                case Type t when t == typeof(float[,]):
                    var obsSize2D = env.stateSize.Match<(int, int)>(
                        intSize => throw new Exception("Unexpected 1D observation dimension for 2D state"),
                        tupleSize => tupleSize);
                    return new PPOCriticNet2D("1DDQN", obsSize2D.Item1, obsSize2D.Item2, neuronsPerLayer, depth, useRNN);
                default:
                    throw new Exception("Unexpected type");
            }
        }

        public PPOCriticNet CreateCriticNet(IContinuousEnvironment<T> env)
        {
            switch (typeof(T))
            {
                case Type t when t == typeof(float[]):
                    var obsSize = env.stateSize.Match<int>(
                        intSize => intSize,
                        tupleSize => throw new Exception("Unexpected 2D observation dimension for 1D state"));
                    return new PPOCriticNet1D("1DDQN", obsSize, neuronsPerLayer, depth, useRNN);
                case Type t when t == typeof(float[,]):
                    var obsSize2D = env.stateSize.Match<(int, int)>(
                        intSize => throw new Exception("Unexpected 1D observation dimension for 2D state"),
                        tupleSize => tupleSize);
                    return new PPOCriticNet2D("2DDQN", obsSize2D.Item1, obsSize2D.Item2, neuronsPerLayer, depth, useRNN);
                default:
                    throw new Exception("Unexpected type");
            }
        }

    }

    public interface IPPONetProvider<T>
    {
        public PPOActorNet CreateActorNet(IEnvironment<T> env);
        public PPOActorNet CreateActorNet(IContinuousEnvironment<T> env);
        public PPOCriticNet CreateCriticNet(IEnvironment<T> env);
        public PPOCriticNet CreateCriticNet(IContinuousEnvironment<T> env);

    }
}
