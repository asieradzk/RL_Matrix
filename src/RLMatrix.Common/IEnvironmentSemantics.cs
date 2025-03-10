namespace RLMatrix.Common;

public interface IEnvironmentSemantics
{
    string EnvironmentDescription { get; }
    IReadOnlyDictionary<string, string> ObservationDescriptions { get; }
    IReadOnlyDictionary<string, string> ActionDescriptions { get; }
}

//Preparing for LLM integration in the future
//example how it may be used in the future
//[RLMatrixEnvironment("Cart-pole balancing environment")]
//public partial class CartPoleToolTest
//{
//    [RLMatrixDescription("Cart position on x-axis")]
//    [RLMatrixObservation]
//    public float GetCartPosition() => myState[0];

//    [RLMatrixDescription("Apply force left (0) or right (1)")]
//    [RLMatrixActionDiscrete(2)]
//    public void ApplyForce(int action) { }
//}