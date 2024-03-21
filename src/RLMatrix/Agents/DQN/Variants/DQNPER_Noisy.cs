using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;
using TorchSharp.Modules;

namespace RLMatrix.Agents.DQN.Variants { 

    public class DQNPerNoisy<T> : DQNAgentPER<T>
{
        public DQNPerNoisy(DQNAgentOptions opts, List<IEnvironment<T>> envs, IDQNNetProvider<T> netProvider = null)
: base(opts, envs, new DuelingNoisyNetworkProvider<T>(opts.Width, opts.Depth))
        {
        }

        public override void OptimizeModel()
        {
            base.OptimizeModel();
        }

        public override int[] SelectAction(T state, bool isTraining = true)
        {
            if(isTraining)
            {
                foreach (var module in from module in myPolicyNet.modules()
                                 where module is NoisyLinear
                                 select module)
                {
                    ((NoisyLinear)module).ResetNoise();
                }
            }


            using (torch.no_grad())
            {
                Tensor stateTensor = StateToTensor(state);
                int[] selectedActions = new int[myEnvironments[0].actionSize.Length];

                // Get action predictions from policy network only once
                Tensor predictedActions = myPolicyNet.forward(stateTensor);

                for (int i = 0; i < myEnvironments[0].actionSize.Length; i++)
                {
                    selectedActions[i] = (int)predictedActions.select(1, i).argmax().item<long>();
                }

                return selectedActions;
            }
        }




    }


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
    }
}
