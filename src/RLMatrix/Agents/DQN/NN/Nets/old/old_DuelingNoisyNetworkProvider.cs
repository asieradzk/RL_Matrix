﻿namespace RLMatrix.Agents.DQN.Variants
{
    /// <summary>
    /// Implementation of IDQNNetProvider tailored for Rainbow DQN networks with Dueling, Categorical (C51), and Noisy network layers.
    /// </summary>
    /// <typeparam name="T">Type of observation space 1D or 2D, for 2D a simple conv network is created</typeparam>
    public sealed class DuelingNoisyNetworkProvider<T> : IDQNNetProvider<T>
    {
        private readonly int neuronsPerLayer;
        private readonly int depth;
        private readonly int numAtoms;

        /// <summary>
        /// Constructor for RainbowNetworkProvider.
        /// </summary>
        /// <param name="neuronsPerLayer">Number of neurons in hidden layers, default is 256 x 3.</param>
        /// <param name="depth">Depth of the network, default is 2.</param>
        /// <param name="useDueling">Flag to use Dueling network topology, default is true.</param>
        /// <param name="numAtoms">Number of atoms for C51, default is 51.</param>
        public DuelingNoisyNetworkProvider(int neuronsPerLayer = 1024, int depth = 2, int numAtoms = 51)
        {
            this.neuronsPerLayer = neuronsPerLayer;
            this.depth = depth;
            this.numAtoms = numAtoms;
        }
        

        public DQNNET CreateCriticNet(IEnvironment<T> env)
        {
            switch (typeof(T))
            {
                case Type t when t == typeof(float[]):
                    var obsSize = env.stateSize.Match<int>(
                        intSize => intSize,
                        tupleSize => throw new Exception("Unexpected 2D observation dimension for 1D state"));
                    var actionSizes = env.actionSize;
                    // For simplicity, I'm assuming actionSizes need to be an array. Adjust based on your environment's needs.
                    return new DuelingDQNoisy("1DDuelingDQN_C51_Noisy", obsSize, neuronsPerLayer, actionSizes, depth);

                case Type t when t == typeof(float[,]):
                    var obsSize2D = env.stateSize.Match<(int, int)>(
                        intSize => throw new Exception("Unexpected 1D observation dimension for 2D state"),
                        tupleSize => tupleSize);
                    var actionSizes2D = env.actionSize;
                    // Similar assumption as above.
                    //TODO: return 2d
                    return new DuelingDQNoisy("2DDuelingDQN_C51_Noisy", obsSize2D.Item1 * obsSize2D.Item2, neuronsPerLayer, actionSizes2D, depth);

                default:
                    throw new Exception("Unexpected type");
            }
        }

        public DQNNET CreateCriticNet(EnvSizeDTO<T> env)
        {
            throw new NotImplementedException();
        }

        public DQNNET CreateCriticNet(IEnvironment<T> env, bool noisyLayers = false, float noiseScale = 0)
        {
            throw new NotImplementedException();
        }

        public DQNNET CreateCriticNet(EnvSizeDTO<T> env, bool noisyLayers = false, float noiseScale = 0)
        {
            throw new NotImplementedException();
        }

        public DQNNET CreateCriticNet(IEnvironment<T> env, bool noisyLayers = false, float noiseScale = 0, int numAtoms = 51)
        {
            throw new NotImplementedException();
        }

        public DQNNET CreateCriticNet(EnvSizeDTO<T> env, bool noisyLayers = false, float noiseScale = 0, int numAtoms = 51)
        {
            throw new NotImplementedException();
        }
    }
}