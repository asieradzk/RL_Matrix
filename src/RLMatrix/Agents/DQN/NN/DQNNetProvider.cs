using RLMatrix;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace RLMatrix
{
    /// <summary>
    /// Default implementation of IDQNNetProvider, makes network with 3 hidden layers
    /// </summary>
    /// <typeparam name="T">Type of observation space 1D or 2D, for 2D a simple conv network is created</typeparam>
    public sealed class DQNNetProvider<T> : IDQNNetProvider<T>
    {
        int neuronsPerLayer;

        /// <summary>
        /// Default constructor for simple DQN NN.
        /// </summary>
        /// <param name="neuronsPerLayer">Number of neurons in hidden layers, default 256 x 3</param>
        public DQNNetProvider(int neuronsPerLayer = 256)
        {
            this.neuronsPerLayer = neuronsPerLayer;
        }

        public DQNNET CreateCriticNet(IEnvironment<T> env)
        {
            switch (typeof(T))
            {
                case Type t when t == typeof(float[]):
                    var obsSize = env.stateSize.Match<int>(
                        intSize => intSize,
                        tupleSize => throw new Exception("Unexpected 2D observation dimension for 1D state"));
                    var actionSize = env.actionSize;
                    return new DQN1D("1DDQN", obsSize, neuronsPerLayer, actionSize);
                case Type t when t == typeof(float[,]):
                    var obsSize2D = env.stateSize.Match<(int, int)>(
                        intSize => throw new Exception("Unexpected 1D observation dimension for 2D state"),
                        tupleSize => tupleSize);
                    var actionSize2 = env.actionSize;
                    return new DQN2D("2DDQN", obsSize2D.Item1, obsSize2D.Item2, actionSize2, neuronsPerLayer);
                default:
                    throw new Exception("Unexpected type");
            }
        }
    }

    public interface IDQNNetProvider<T>
    {
        public DQNNET CreateCriticNet(IEnvironment<T> env);
    }
}
