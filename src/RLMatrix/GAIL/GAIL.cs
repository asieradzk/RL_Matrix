using RLMatrix.Common;

// TODO: All GAIL needs to be re-implemented. No longer works
// TODO: 2-for-1-special: Any reason this doesn't implement IGAIL<TState>?
namespace RLMatrix;

public class GAIL<TState>
    where TState : notnull
{
    private readonly GAILOptions _options;
    private readonly IMemory<TState> _expertDemonstrations;

    // These are initialized in Initialize
    private GAILNET _net = null!;
    private Device _device = null!;
    private OptimizerHelper _optimizerHelper = null!;

    /// <summary>
    ///     RL Matrix GAIL serves as way to augment rewards and as such should be possible to implement with any RLMatrix agent.
    ///     This is still a test version, some key concepts:
    ///     Exposes a method to initialize network for custom state/action sizes so the agent can initialize GAIL for its environment specs.
    ///     Options must accept an expert replay buffer and constructor should use it or throw exception. 
    ///     User instantiates an instance of GAIL with options, but it's the agent who properly initializes it.
    /// </summary>
    public GAIL(IMemory<TState> expertDemonstrations, GAILOptions? gailOptions = null)
    {
        _options = gailOptions ?? new GAILOptions();
        _expertDemonstrations = expertDemonstrations;
    }

    public void Initialize(StateDimensions observationDimensions, int[] actionSize, ContinuousActionDimensions[] continuousActionDimensions, Device myDevice)
    {
        _device = myDevice;

        if (typeof(TState) == typeof(float[]))
        {
            if (observationDimensions is not { Dimensions: var dimensions, Dimensions.Length: 1 })
                throw new Exception("Unexpected observation dimension for 1D state.");
            
            _net = new GAILDiscriminator1D("1DDQN", dimensions[0], actionSize, continuousActionDimensions, _options.NeuralNetworkWidth, _options.NeuralNetworkDepth).to(myDevice);
        }
        else if (typeof(TState) == typeof(float[,]))
        {
            if (observationDimensions is not { Dimensions: var dimensions, Dimensions.Length: 2 })
                throw new Exception("Unexpected observation dimension for 2D state.");
            
            _net = new GAILDiscriminator2D("2DDQN", dimensions[0], dimensions[1], actionSize, continuousActionDimensions, _options.NeuralNetworkWidth, _options.NeuralNetworkDepth).to(myDevice);
        }
        else
        {
            throw new Exception("Unexpected type");
        }

        _optimizerHelper = torch.optim.Adam(_net.parameters(), _options.LearningRate);
    }

    public Tensor AugmentRewards(Tensor states, Tensor actions, Tensor rewards)
    {
        using(torch.no_grad())
        {
            // Combine states and actions for a forward pass through the network
            var combinedInput = torch.cat([states, actions], dim: 1);

            // Perform a forward pass to get the expertness score
            var expertnessScore = _net.forward(combinedInput);

            // Ensure expertnessScore is a single-dimensional tensor matching the rewards batch size
            expertnessScore = expertnessScore.squeeze();

            var numSteps = rewards.shape[0];

            //TODO: Could pick reward augmentation algorithm depending on the batch size in the agent (somehow need to capture that dependency)

            //This works with multi-espide batches, but can bias agent towards longer episodes
            var rewardModification = expertnessScore * _options.RewardFactor;

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
            var augmentedRewards = rewards + rewardModification;

            return augmentedRewards;
        }
    }

    public void OptimiseDiscriminator(IMemory<TState> replayBuffer)
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