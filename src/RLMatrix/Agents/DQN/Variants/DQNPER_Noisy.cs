using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;
using TorchSharp.Modules;

namespace RLMatrix.Agents.DQN.Variants
{

    public class DQNPerNoisy<T> : DQNAgentPER<T>
{
        public DQNPerNoisy(DQNAgentOptions opts, List<IEnvironment<T>> envs, IDQNNetProvider<T> netProvider = null)
: base(opts, envs, netProvider ?? new DuelingNoisyNetworkProvider<T>(opts.Width, opts.Depth, opts.NumAtoms))
        {
            if (noisyLayers == null)
            {
                noisyLayers = new();
                noisyLayers.AddRange(from module in myPolicyNet.modules()
                                     where module is NoisyLinear
                                     select (NoisyLinear)module);
            }
        }

        List<NoisyLinear> noisyLayers;

        public virtual void ResetNoise()
        {
            //Parallel.ForEach(noisyLayers, module => module.ResetNoise());

            foreach (var module in noisyLayers)
            {
                module.ResetNoise();
            }
        }

        public override void OptimizeModel()
        {
            base.OptimizeModel();
        }

        public override int[] SelectAction(T state, bool isTraining = true)
        {
            if(isTraining)
            {
                ResetNoise();
            }

            return ActionsFromState(state);
        }

        public override int[] ActionsFromState(T state)
        {
            using (torch.no_grad())
            {
                Tensor stateTensor = StateToTensor(state); // Shape: [state_dim]

                Tensor qValuesAllHeads = myPolicyNet.forward(stateTensor).view(1, myEnvironments[0].actionSize.Length, myEnvironments[0].actionSize[0]); // Shape: [1, num_heads, num_actions]

                Tensor bestActions = qValuesAllHeads.argmax(dim: -1).squeeze().to(ScalarType.Int32); // Shape: [num_heads]

                return bestActions.data<int>().ToArray();
            }
        }


    }
}
