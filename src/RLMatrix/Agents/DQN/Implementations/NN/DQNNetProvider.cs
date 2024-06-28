using OneOf;
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
        int depth;
        NetworkType myNetworkTYpe;

        /// <summary>
        /// Default constructor for simple DQN NN.
        /// </summary>
        /// <param name="neuronsPerLayer">Number of neurons in hidden layers, default 256 x 3</param>
        public DQNNetProvider(int neuronsPerLayer = 256, int depth = 2, bool useDueling = false, bool categorical = false)
        {
            this.neuronsPerLayer = neuronsPerLayer;
            this.depth = depth;
            switch (categorical, useDueling)
            {
                case (true, true):
                    myNetworkTYpe = NetworkType.categoricalDueling;
                    break;
                case (true, false):
                    myNetworkTYpe = NetworkType.categorical;
                    break;
                case (false, true):
                    myNetworkTYpe = NetworkType.dueling;
                    break;
                case (false, false):
                    myNetworkTYpe = NetworkType.vanilla;
                    break;
            }
        }

        public DQNNET CreateCriticNet(EnvSizeDTO<T> env, bool noisyLayers = false, float noiseScale = 0f, int numAtoms = 51)
        {
            switch (typeof(T))
            {
                case Type t when t == typeof(float[]):
                    var obsSize = env.stateSize.Match<int>(
                        intSize => intSize,
                        tupleSize => throw new Exception("Unexpected 2D observation dimension for 1D state"));
                    var actionSize = env.actionSize;
                    return myNetworkTYpe switch
                    {
                        NetworkType.vanilla => new DQN1D("1DDQN", obsSize, neuronsPerLayer, actionSize, depth, noisyLayers, noiseScale),
                        NetworkType.dueling => new DuelingDQN("1DDuelingDQN", obsSize, neuronsPerLayer, actionSize, depth, noisyLayers, noiseScale),
                        NetworkType.categorical => new CategoricalDQN1D("1DCategoricalDQN", obsSize, neuronsPerLayer, actionSize, depth, numAtoms, noisyLayers, noiseScale),
                        NetworkType.categoricalDueling => new CategoricalDuelingDQN1D("1DCategoricalDuelingDQN", obsSize, neuronsPerLayer, actionSize, depth, numAtoms, noisyLayers, noiseScale),
                        _ => throw new ArgumentOutOfRangeException(nameof(myNetworkTYpe), myNetworkTYpe, null)
                    };
                case Type t when t == typeof(float[,]):
                    var obsSize2D = env.stateSize.Match<(int, int)>(
                        intSize => throw new Exception("Unexpected 1D observation dimension for 2D state"),
                        tupleSize => tupleSize);
                    var actionSize2 = env.actionSize;
                    return myNetworkTYpe switch
                    {
                        NetworkType.vanilla => new DQN2D("2DDQN", obsSize2D.Item1, obsSize2D.Item2, actionSize2, neuronsPerLayer, depth: depth, noisyLayers, noiseScale),
                        NetworkType.dueling => new DuelingDQN2D("2DDuelingDQN", obsSize2D.Item1, obsSize2D.Item2, actionSize2, neuronsPerLayer, depth: depth, noisyLayers, noiseScale),
                        NetworkType.categorical => new CategoricalDQN2D("2DCategoricalDQN", obsSize2D.Item1, obsSize2D.Item2, actionSize2, neuronsPerLayer, depth: depth, numAtoms, noisyLayers, noiseScale),
                        NetworkType.categoricalDueling => new CategoricalDuelingDQN2D("2DCategoricalDuelingDQN", obsSize2D.Item1, obsSize2D.Item2, actionSize2, neuronsPerLayer, depth: depth, numAtoms, noisyLayers, noiseScale),
                        _ => throw new ArgumentOutOfRangeException(nameof(myNetworkTYpe), myNetworkTYpe, null)
                    };
                default:
                    throw new Exception("Unexpected type");
            }
        }

        internal enum NetworkType
        {
            vanilla,
            dueling,
            categorical,
            categoricalDueling
        }
    }

    public class EnvSizeDTO<T>
    {
#if NET8_0_OR_GREATER
        public required OneOf<int, (int, int)> stateSize;
        public required int[] actionSize;
#else
       public OneOf<int, (int, int)> stateSize;
       public int[] actionSize;
#endif
    }

    public interface IDQNNetProvider<T>
    {
        DQNNET CreateCriticNet(EnvSizeDTO<T> env, bool noisyLayers = false, float noiseScale = 0f, int numAtoms = 51);
    }
}