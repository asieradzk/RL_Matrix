using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;
using TorchSharp.Modules;

namespace RLMatrix
{
    public abstract class PPOActorNet : Module<Tensor, Tensor>
    {
        public PPOActorNet(string name) : base(name)
        {
        }

        public override abstract Tensor forward(Tensor x);
        public abstract (Tensor, Tensor) forward(Tensor x, Tensor? state);


        public Tensor get_log_prob(Tensor states, Tensor actions, int discreteActions, int contActions)
        {
            Tensor logits = forward(states);
            var discreteLogProbs = new List<Tensor>();
            var continuousLogProbs = new List<Tensor>();

            // Discrete action log probabilities
            for (int i = 0; i < discreteActions; i++)
            {
                Tensor actionLogits = logits.select(1, i);
                Tensor actionTaken = actions.select(1, i).to(ScalarType.Int64).unsqueeze(-1);
                discreteLogProbs.Add(torch.nn.functional.log_softmax(actionLogits, dim: 1).gather(dim: 1, index: actionTaken));
                actionLogits.Dispose();
                actionTaken.Dispose();
            }

            // Continuous action log probabilities
            for (int i = 0; i < contActions; i++)
            {
                Tensor mean = logits.select(1, discreteActions + i);
                Tensor log_std = logits.select(1, discreteActions + contActions + i);
                Tensor actionTaken = actions.select(1, discreteActions + i);
                Tensor log_prob = (-0.5 * torch.pow((actionTaken - mean) / torch.exp(log_std), 2) - log_std - 0.5 * Math.Log(2 * Math.PI)).sum(1, keepdim: true);
                continuousLogProbs.Add(log_prob);
                mean.Dispose();
                log_std.Dispose();
                actionTaken.Dispose();
                log_prob.Dispose();
            }

            // Combine discrete and continuous log probabilities
            var res = torch.cat(discreteLogProbs.Concat(continuousLogProbs).ToArray(), dim: 1).squeeze();
            foreach (var tensor in discreteLogProbs.Concat(continuousLogProbs))
            {
                tensor.Dispose();
            }
            logits.Dispose();
            return res;
        }


        public Tensor ComputeEntropy(Tensor states, int discreteActions, int continuousActions)
        {
            Tensor logits = forward(states);

            var discreteEntropies = new List<Tensor>();
            var continuousEntropies = new List<Tensor>();

            // Discrete action entropy
            for (int i = 0; i < discreteActions; i++)
            {
                Tensor actionProbs = logits.select(1, i);  // These are already probabilities
                Tensor logProbs = torch.log(actionProbs + 1e-10);  // Add a small constant for numerical stability
                Tensor entropy = -(actionProbs * logProbs).sum(1, keepdim: true);
                discreteEntropies.Add(entropy);
             //   actionProbs.Dispose();
             //   logProbs.Dispose();
             //   discreteEntropies.Add(entropy);
            }

            // Continuous action entropy
            for (int i = 0; i < continuousActions; i++)
            {
                Tensor log_std = logits.select(1, discreteActions + i);
                Tensor entropy = 0.5 + 0.5 * Math.Log(2 * Math.PI) + log_std;  // The entropy for a Gaussian distribution
                continuousEntropies.Add(entropy);
              //  log_std.Dispose();
                
            }

            // Combine the entropies
            Tensor totalEntropy = torch.tensor(0.0f).to(states.device);  // Initialize to zero tensor
            if (discreteEntropies.Count > 0)
            {
                totalEntropy += torch.cat(discreteEntropies.ToArray(), dim: 1).mean(new long[] { 1 }, true);
            }

            if (continuousEntropies.Count > 0)
            {
                totalEntropy += torch.cat(continuousEntropies.ToArray(), dim: 1).mean(new long[] { 1 }, true);
            }

            // Normalize by the number of action heads if needed
            if (discreteEntropies.Count + continuousEntropies.Count > 0)
            {
                totalEntropy /= (discreteEntropies.Count + continuousEntropies.Count);
            }

            foreach (var tensor in discreteEntropies.Concat(continuousEntropies))
            {
              //  tensor.Dispose();
            }

            return totalEntropy;
        }

        public (Tensor logprobs, Tensor entropy) get_log_prob_entropy(Tensor states, Tensor actions, int discreteActions, int contActions)
        {
            Tensor logits = forward(states);

            var discreteLogProbs = new List<Tensor>();
            var continuousLogProbs = new List<Tensor>();
            var discreteEntropies = new List<Tensor>();
            var continuousEntropies = new List<Tensor>();

            // Discrete action log probabilities and entropy
            for (int i = 0; i < discreteActions; i++)
            {
                Tensor actionLogits = logits.select(1, i);
                Tensor actionProbs = torch.nn.functional.softmax(actionLogits, dim: 1);
                Tensor actionTaken = actions.select(1, i).to(ScalarType.Int64).unsqueeze(-1);
                discreteLogProbs.Add(torch.log(actionProbs).gather(dim: 1, index: actionTaken));
                discreteEntropies.Add(-(actionProbs * torch.log(actionProbs + 1e-10)).sum(1, keepdim: true));
             //   actionLogits.Dispose();
              //  actionProbs.Dispose();
              //  actionTaken.Dispose();
            }

            // Continuous action log probabilities and entropy
            for (int i = 0; i < contActions; i++)
            {
                Tensor mean = logits.select(1, discreteActions + i);
                Tensor log_std = logits.select(1, discreteActions + contActions + i);
                Tensor actionTaken = actions.select(1, discreteActions + i);
                Tensor log_prob = (-0.5 * torch.pow((actionTaken - mean) / torch.exp(log_std), 2) - log_std - 0.5 * Math.Log(2 * Math.PI)).sum(1, keepdim: true);
                continuousLogProbs.Add(log_prob);
                continuousEntropies.Add(0.5 + 0.5 * Math.Log(2 * Math.PI) + log_std);
             //   mean.Dispose();
             //   log_std.Dispose();
             //   actionTaken.Dispose();
             //   log_prob.Dispose();
            }

            // Combine discrete and continuous log probabilities and entropies
            Tensor logProbs = torch.cat(discreteLogProbs.Concat(continuousLogProbs).ToArray(), dim: 1).squeeze();           

            Tensor totalEntropy = torch.tensor(0.0f).to(states.device);

            if (discreteEntropies.Count > 0)
            {
                totalEntropy += torch.cat(discreteEntropies.ToArray(), dim: 1).mean(new long[] { 1 }, true);
            }

            if (continuousEntropies.Count > 0)
            {
                totalEntropy += torch.cat(continuousEntropies.ToArray(), dim: 1).mean(new long[] { 1 }, true);
            }

            if (discreteEntropies.Count + continuousEntropies.Count > 0)
            {
                totalEntropy /= (discreteEntropies.Count + continuousEntropies.Count);
            }

            foreach (var tensor in discreteLogProbs.Concat(continuousLogProbs).Concat(discreteEntropies).Concat(continuousEntropies))
            {
               // tensor.Dispose();
            }

            return (logProbs, totalEntropy);
        }
    }

    public class PPOActorNet1D : PPOActorNet
    {
        private readonly ModuleList<Module<Tensor, Tensor>> fcModules = new();
        private readonly ModuleList<Module<Tensor, Tensor>> discreteHeads = new();
        private readonly ModuleList<Module<Tensor, Tensor>> continuousHeadsMean = new();
        private readonly ModuleList<Module<Tensor, Tensor>> continuousHeadsLogStd = new();
        private GRU lstmLayer;
        //private readonly GRU lstmLayer;
        private readonly int hiddenSize;
        private readonly bool useRnn;

        public PPOActorNet1D(string name, long inputs, int width, int[] discreteActions, (float, float)[] continuousActionBounds, int depth = 3, bool useRNN = false) : base(name)
        {
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");

            this.useRnn = useRNN;
            this.hiddenSize = width;
         

            if (useRnn)
            {
                // Initialize LSTM layer if useRnn is true
                lstmLayer = nn.GRU(inputs, hiddenSize, depth, batchFirst: true, dropout: 0.05f);
                // width = hiddenSize; // The output of LSTM layer is now the input for the heads
               
            }

            // Base layers
            if (useRnn)
            {
                fcModules.Add(Linear(width, width));
            }
            else
            {
                fcModules.Add(Linear(inputs, width));
            }

            
            for (int i = 1; i < depth; i++)
            {
                fcModules.Add(Linear(width, width));
                
            }
           

            // Discrete Heads
            foreach (var actionSize in discreteActions)
            {
                discreteHeads.Add(Linear(width, actionSize));
            }

            // Continuous Heads for means and log std deviations
            foreach (var actionBounds in continuousActionBounds)
            {
                continuousHeadsMean.Add(Linear(width, 1));  // Assuming one output per continuous action dimension for mean
                continuousHeadsLogStd.Add(Linear(width, 1));  // Assuming one output per continuous action dimension for log std deviation
            }

            RegisterComponents();
        }

        private Tensor ApplyHeads(Tensor x)
        {
            var outputs = new List<Tensor>();
            foreach (var head in discreteHeads)
            {
                outputs.Add(functional.softmax(head.forward(x), 1));
            }
            foreach (var head in continuousHeadsMean)
            {
                outputs.Add(head.forward(x));  // mean values
            }
            foreach (var head in continuousHeadsLogStd)
            {
                outputs.Add(functional.softplus(head.forward(x)));  // Convert to log std deviation values
            }
            return stack(outputs, dim: 1);
        }

        public override Tensor forward(Tensor x)
        {
            if (x.dim() == 1)
            {
                x = x.unsqueeze(0);
            }

            if(useRnn && x.dim() == 2)
            {
                x = x.unsqueeze(0);
            }


            // Apply the first fc module
            if (useRnn)
            {

                var res = lstmLayer.forward(x, null);
                x = res.Item1;
                x = x.reshape(new long[] { -1, x.size(2) });
                x = functional.tanh(fcModules.First().forward(x));
        
            }
            else
            {
                x = functional.tanh(fcModules.First().forward(x));
            }



            // Apply the rest of the fc modules
            foreach (var module in fcModules.Skip(1))
            {
                x = functional.tanh(module.forward(x));
            }

            var result = ApplyHeads(x);
            return result;
        }
        public override (Tensor, Tensor) forward(Tensor x, Tensor? state)
        {
            if (x.dim() == 1)
            {
                x = x.unsqueeze(0);
            }

            Tensor resultHiddenState = null;
            if (useRnn)
            {
                if (state is null)
                {
                  //  state = torch.zeros(new long[] { 4, x.size(0), hiddenSize }, requires_grad: false).to(x.device);
                }


                x = x.unsqueeze(0);
                // Apply LSTM layer if useRnn is true
                (Tensor, Tensor) lstmResult;
                lstmResult = lstmLayer.forward(x, state);              
                x = lstmResult.Item1;
                resultHiddenState = lstmResult.Item2.detach().alias();


                x = x.squeeze(0);
                x = functional.tanh(fcModules.First().forward(x));
            }
            else
            {
                x = functional.tanh(fcModules.First().forward(x));
            }

            // Apply the rest of the fc modules
            foreach (var module in fcModules.Skip(1))
            {
                x = functional.tanh(module.forward(x));
            }

            var result = ApplyHeads(x);
            return (result, resultHiddenState);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                // Dispose modules in fcModules
                foreach (var module in fcModules)
                {
                    module.Dispose();
                }

                lstmLayer?.Dispose();

                // Dispose discrete heads
                foreach (var head in discreteHeads)
                {
                    head.Dispose();
                }

                // Dispose continuous heads for mean values
                foreach (var head in continuousHeadsMean)
                {
                    head.Dispose();
                }

                // Dispose continuous heads for log standard deviation values
                foreach (var head in continuousHeadsLogStd)
                {
                    head.Dispose();
                }

                // Clear internal module list
                ClearModules();
            }

            base.Dispose(disposing);
        }
    }




    public class PPOActorNet2D : PPOActorNet
    {
        private Module<Tensor, Tensor> conv1, flatten;
        private readonly ModuleList<Module<Tensor, Tensor>> fcModules = new();
        private readonly ModuleList<Module<Tensor, Tensor>> discreteHeads = new();
        private readonly ModuleList<Module<Tensor, Tensor>> continuousMeans = new();
        private readonly ModuleList<Module<Tensor, Tensor>> continuousStds = new();
        private readonly GRU GRULayer;
        private int width;
        private long linear_input_size;
        private bool useRNN;

        public long CalculateConvOutputSize(long inputSize, long kernelSize, long stride = 1, long padding = 0)
        {
            return ((inputSize - kernelSize + 2 * padding) / stride) + 1;
        }

        public PPOActorNet2D(string name, long h, long w, int[] discreteActionSizes, (float, float)[] continuousActionBounds, int width, int depth = 3, bool useRNN = false) : base(name)
        {
            if (depth < 1) throw new ArgumentOutOfRangeException("Depth must be 1 or greater.");

            this.width = width;
            this.useRNN = useRNN;

            var smallestDim = Math.Min(h, w);
            var padding = smallestDim / 2;

            conv1 = Conv2d(1, width, kernelSize: (smallestDim, smallestDim), stride: (1, 1), padding: (padding, padding));

            long output_height = CalculateConvOutputSize(h, smallestDim, stride: 1, padding: padding);
            long output_width = CalculateConvOutputSize(w, smallestDim, stride: 1, padding: padding);
            linear_input_size = output_height * output_width * width;

            flatten = Flatten();
            if (useRNN)
            {
                GRULayer = nn.GRU(linear_input_size, width, depth, batchFirst: true);
                fcModules.Add(Linear(width, width));
            }
            else
            {
                fcModules.Add(Linear(linear_input_size, width));
            }

            for (int i = 1; i < depth; i++)
            {
                fcModules.Add(Linear(width, width));
            }

            foreach (var discreteActionSize in discreteActionSizes)
            {
                discreteHeads.Add(Linear(width, discreteActionSize));
            }

            foreach (var actionBound in continuousActionBounds)
            {
                continuousMeans.Add(Linear(width, 1));  // Assuming each continuous action is 1-dimensional
                continuousStds.Add(Linear(width, 1));
            }

            RegisterComponents();
        }
        public override Tensor forward(Tensor x)
        {
            if (x.dim() == 2)
            {
                x = x.unsqueeze(0).unsqueeze(0);
            }
            else if (x.dim() == 3)
            {
                x = x.unsqueeze(1);
            }

            var batchength = x.size(0);
            var sequencLength = x.size(1);

            if(useRNN && x.dim() == 4)
            {
                x = x.reshape(new long[] { batchength * sequencLength, 1, x.size(2), x.size(3) });
            }

            x = functional.tanh(conv1.forward(x));
            x = flatten.forward(x);
            if (useRNN)
            {
                x = x.reshape(new long[] { batchength, sequencLength, x.size(1) });
                x = GRULayer.forward(x, null).Item1;
                x = x.reshape(new long[] { -1, x.size(2) });
            }
            

            foreach (var module in fcModules)
            {
                x = functional.tanh(module.forward(x));
            }

            var discreteOutputs = new List<Tensor>();
            var continuousOutputs = new List<Tensor>();

            foreach (var head in discreteHeads)
            {
                discreteOutputs.Add(functional.softmax(head.forward(x), 1));
            }

            foreach (var meanModule in continuousMeans)
            {
                continuousOutputs.Add(meanModule.forward(x));
            }

            foreach (var stdModule in continuousStds)
            {
                continuousOutputs.Add(functional.softplus(stdModule.forward(x)));
            }

            var res = stack(discreteOutputs.Concat(continuousOutputs).ToArray(), dim: 1);
            return res;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                conv1.Dispose();
                foreach (var module in fcModules.Concat(discreteHeads).Concat(continuousMeans).Concat(continuousStds))
                {
                    module.Dispose();
                }
                
                ClearModules();
            }

            base.Dispose(disposing);
        }


        public override (Tensor, Tensor) forward(Tensor x, Tensor? state)
        {
            if (x.dim() == 2)
            {
                x = x.unsqueeze(0).unsqueeze(0);
            }
            else if (x.dim() == 3)
            {
                x = x.unsqueeze(1);
            }
            x = functional.tanh(conv1.forward(x));
            x = flatten.forward(x);
            Tensor stateRes = null;
            if (useRNN)
            {
                x = x.unsqueeze(0);
                var result = GRULayer.forward(x, state);
                x = result.Item1.squeeze(0);
                stateRes = result.Item2;
            }
            

            foreach (var module in fcModules)
            {
                x = functional.tanh(module.forward(x));
            }

            var discreteOutputs = new List<Tensor>();
            var continuousOutputs = new List<Tensor>();

            foreach (var head in discreteHeads)
            {
                discreteOutputs.Add(functional.softmax(head.forward(x), 1));
            }

            foreach (var meanModule in continuousMeans)
            {
                continuousOutputs.Add(meanModule.forward(x));
            }

            foreach (var stdModule in continuousStds)
            {
                continuousOutputs.Add(functional.softplus(stdModule.forward(x)));
            }

            return (stack(discreteOutputs.Concat(continuousOutputs).ToArray(), dim: 1), stateRes);
        }
    }


}
