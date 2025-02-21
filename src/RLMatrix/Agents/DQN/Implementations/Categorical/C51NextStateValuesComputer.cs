using RLMatrix.Common;

namespace RLMatrix;

public class C51NextStateValuesComputer : INextStateValuesComputer
{
    private readonly int _numAtoms;

    public C51NextStateValuesComputer(int numAtoms)
    {
        _numAtoms = numAtoms;
    }

    public Tensor ComputeNextStateValues(Tensor nonFinalNextStates, TensorModule targetNet, TensorModule policyNet, DQNAgentOptions opts, int[] discreteActionDimensions, Device device)
    {
        Tensor nextStateDistributions;

        using (torch.no_grad())
        {
            if (nonFinalNextStates.shape[0] > 0)
            {
                if (opts.UseDoubleDQN)
                {
                    // Using policyNet to select the best action for each next state based on current policy
                    var policyDistributions = policyNet.forward(nonFinalNextStates);
                    var meanQValues = policyDistributions.mean([3]); // Average across atoms
                    var nextActions = meanQValues.max(2, keepdim: true).indexes; // Select best action for each head

                    // Evaluating the selected actions' Q-value distributions using targetNet
                    var allNextStateDistributions = targetNet.forward(nonFinalNextStates);

                    // Gather the distributions corresponding to the selected actions
                    nextStateDistributions = allNextStateDistributions.gather(2, nextActions.unsqueeze(3).expand(-1, -1, -1, _numAtoms)).squeeze(2);
                }
                else
                {
                    // Compute the Q-value distributions for all actions in the next states using targetNet
                    nextStateDistributions = targetNet.forward(nonFinalNextStates);

                    // Select the Q-value distributions corresponding to the actions with the highest mean Q-value
                    nextStateDistributions = nextStateDistributions.max(2).values;
                }
            }
            else
            {
                nextStateDistributions = torch.zeros(new long[] { opts.BatchSize, _numAtoms }, device: device);
            }
        }

        return nextStateDistributions;
    }
}