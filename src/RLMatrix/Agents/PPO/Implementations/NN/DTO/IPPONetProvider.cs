namespace RLMatrix;

public interface IPPONetProvider
{
    public PPOActorNet CreateActorNet(EnvironmentSizeDTO env);
    public PPOCriticNet CreateCriticNet(EnvironmentSizeDTO env);
}