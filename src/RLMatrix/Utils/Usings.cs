global using TorchSharp;
global using TorchSharp.Modules;

global using CyclicLR = TorchSharp.torch.optim.lr_scheduler.impl.CyclicLR;
global using Device = TorchSharp.torch.Device;
global using KLDivLoss = TorchSharp.Modules.KLDivLoss;
global using LRScheduler = TorchSharp.torch.optim.lr_scheduler.LRScheduler;
global using NoisyLinear = RLMatrix.NoisyLinear;
global using PackedSequence = TorchSharp.torch.nn.utils.rnn.PackedSequence;
global using Reduction = TorchSharp.torch.nn.Reduction;
global using ScalarType = TorchSharp.torch.ScalarType;
global using Tensor = TorchSharp.torch.Tensor;
global using TensorModule = TorchSharp.torch.nn.Module<TorchSharp.torch.Tensor, TorchSharp.torch.Tensor>;
global using TensorEnumerableModule = TorchSharp.torch.nn.Module<TorchSharp.torch.Tensor, System.Collections.Generic.IEnumerable<TorchSharp.torch.Tensor>>;
