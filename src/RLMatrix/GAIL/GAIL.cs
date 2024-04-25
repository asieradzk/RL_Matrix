using OneOf;
using RLMatrix;
using RLMatrix.GAILNET;
using RLMatrix.Memories;
using RLMatrix.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using static TorchSharp.torchvision;

public class GAIL<T>
{

    IMemory<T> expertDemonstrations;
    GAILOptions myOptions;
    GAILNET myNet;
    Device myDevice;
    OptimizerHelper myOptimizer;

    /// <summary>
    /// RL Matrix GAIL serves as way to augment rewards and as such should be possible to implement with any RLMatrix agent.
    /// This is still a test version, some key concepts:
    /// Expose a method to initialize network for custom state/action sizes so the agent can initialzie GAIL for its environment specs
    /// Options must accept an expert replay buffer and constructor should use it or throw exception. 
    /// User instantiates instance of GAIL with options but its the agent who properly initializes it.
    /// </summary>
    public GAIL(IMemory<T> expertDemonstrations, GAILOptions gailOptions = null)
    {
        if (gailOptions == null)
        {
            gailOptions = new GAILOptions();
        }
        if (expertDemonstrations == null)
        {
            throw new ArgumentNullException("Expert Demonstrations cannot be null");
        }   
        this.expertDemonstrations = expertDemonstrations;
        this.myOptions = gailOptions;
        
    }

    public void Initialise(OneOf<int, (int, int)> obSize, int[] actionSize, (float, float)[] continousActions, Device myDevice)
    {
        this.myDevice = myDevice;

        switch (typeof(T))
        {
            case Type t when t == typeof(float[]):
                var obsSize = obSize.Match<int>(
                                       intSize => intSize,
                                                          tupleSize => throw new Exception("Unexpected 2D observation dimension for 1D state"));
                var actionSize1D = actionSize;
                myNet = new GAILDiscriminator1D("1DDQN", obsSize, actionSize1D, continousActions, myOptions.NNWidth, myOptions.NNDepth).to(myDevice);
                break;
            case Type t when t == typeof(float[,]):
                var obsSize2D = obSize.Match<(int, int)>(
                                       intSize => throw new Exception("Unexpected 1D observation dimension for 2D state"),
                                                          tupleSize => tupleSize);
                var actionSize2D = actionSize;
                myNet = new GAILDiscriminator2D("2DDQN", obsSize2D.Item1, obsSize2D.Item2, actionSize2D, continousActions, myOptions.NNWidth, myOptions.NNDepth).to(myDevice);
                break;
            default:
                throw new Exception("Unexpected type");
        }

        this.myOptimizer = optim.Adam(myNet.parameters(), myOptions.LR);

    }

    public Tensor AugmentRewards(Tensor states, Tensor actions, Tensor rewards)
    {
        using(torch.no_grad())
        {

            // Combine states and actions for a forward pass through the network
            Tensor combinedInput = torch.cat(new Tensor[] { states, actions }, dim: 1);

            // Perform a forward pass to get the expertness score
            Tensor expertnessScore = myNet.forward(combinedInput);

            // Ensure expertnessScore is a single-dimensional tensor matching the rewards batch size
            expertnessScore = expertnessScore.squeeze();

            long numSteps = rewards.shape[0];


            //TODO: Could pick reward augmentation algorithm depending on the batch size in the agent (somehow need to capture that dependency)


            //This works with multi-espide batches, but can bias agent towards longer episodes
            Tensor rewardModification = expertnessScore * myOptions.rewardFactor;


            //This only works with single-episode long batches but does not bias agent towards longer episodes since the reward is always divided by n steps in the batch-episode
            /*
            // Calculate the reward modification
            // If expertnessScore > 0.5, add (expertnessScore - 0.5) * myOptions.rewardFactor
            // If expertnessScore < 0.5, subtract (0.5 - expertnessScore) * myOptions.rewardFactor
            Tensor rewardModification = torch.where(expertnessScore > 0.5f,
                                                    (expertnessScore - 0.5f) * myOptions.rewardFactor/ numSteps,
                                                    (0.5f - expertnessScore) * -myOptions.rewardFactor/ numSteps);

            */
            // Augment the rewards
            Tensor augmentedRewards = rewards + rewardModification;

            return augmentedRewards;
        }

    }


    public void OptimiseDiscriminator(IMemory<T> replayBuffer)
    {
        throw new NotImplementedException();
        /*
        List<TransitionReplayMemory<T>> agentTransitions;
        List<TransitionReplayMemory<T>> expertTransitions;
        Tensor agentLabels;
        Tensor expertLabels;

        var count = replayBuffer.Length;

        if(count < myOptions.BatchSize || count < myOptions.BatchSize)
        {
            agentTransitions = replayBuffer.Sample(count).ToArray().ToList();
            expertTransitions = expertDemonstrations.Sample(count).ToArray().ToList();
            agentLabels = torch.zeros(count, dtype: torch.@float).to(myDevice);
            expertLabels = torch.ones(count, dtype: torch.@float).to(myDevice);
        }
        else
        {
            //TODO: Span stuff
            agentTransitions = replayBuffer.Sample(myOptions.BatchSize).ToArray().ToList();
            expertTransitions = expertDemonstrations.Sample(myOptions.BatchSize).ToArray().ToList();
            // Labels: 0 for agent-generated, 1 for expert
            agentLabels = torch.zeros(myOptions.BatchSize, dtype: torch.@float).to(myDevice);
            expertLabels = torch.ones(myOptions.BatchSize, dtype: torch.@float).to(myDevice);
        }



        // Convert to tensors - Agent
        Tensor agentStateBatch = stack(agentTransitions.Select(t => Utilities<T>.StateToTensor(t.state, myDevice)).ToArray()).to(myDevice);
        Tensor agentDiscreteActionBatch = stack(agentTransitions.Where(t => t.discreteActions != null).Select(t => tensor(t.discreteActions)).ToArray()).to(myDevice);
        Tensor agentContinuousActionBatch = stack(agentTransitions.Where(t => t.continuousActions != null).Select(t => tensor(t.continuousActions)).ToArray()).to(myDevice);
        Tensor agentActionBatch = torch.cat(new Tensor[] { agentDiscreteActionBatch, agentContinuousActionBatch }, dim: 1);

        // Convert to tensors - Expert
        Tensor expertStateBatch = stack(expertTransitions.Select(t => Utilities<T>.StateToTensor(t.state, myDevice)).ToArray()).to(myDevice);
        Tensor expertDiscreteActionBatch = stack(expertTransitions.Where(t => t.discreteActions != null).Select(t => tensor(t.discreteActions)).ToArray()).to(myDevice);
        Tensor expertContinuousActionBatch = stack(expertTransitions.Where(t => t.continuousActions != null).Select(t => tensor(t.continuousActions)).ToArray()).to(myDevice);
        Tensor expertActionBatch = torch.cat(new Tensor[] { expertDiscreteActionBatch, expertContinuousActionBatch }, dim: 1);

        // Combine states and actions for both agent and expert
        Tensor combinedAgentInput = torch.cat(new Tensor[] { agentStateBatch, agentActionBatch }, dim: 1);
        Tensor combinedExpertInput = torch.cat(new Tensor[] { expertStateBatch, expertActionBatch }, dim: 1);



        // Combine inputs and labels
        Tensor inputs = torch.cat(new Tensor[] { combinedAgentInput, combinedExpertInput }, dim: 0).to(myDevice);
        Tensor labels = torch.cat(new Tensor[] { agentLabels, expertLabels }, dim: 0).to(myDevice);

        // Forward pass
        Tensor predictions = myNet.forward(inputs);

        var criterion = nn.BCELoss();

        // Compute loss
        var loss = criterion.forward(predictions, labels);

        // Backward and optimize
        myOptimizer.zero_grad();
        loss.backward();
        myOptimizer.step();
        */
    }



}
