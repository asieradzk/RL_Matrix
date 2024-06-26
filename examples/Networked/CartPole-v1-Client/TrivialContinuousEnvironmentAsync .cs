using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using OneOf;
using RLMatrix;

public class TrivialContinuousEnvironmentAsync : IContinuousEnvironmentAsync<float[]>
{
    public const float CorrectAnswerReward = 1;
    public const float WrongAnswerPenalty = -1;
    public OneOf<int, (int, int)> StateSize { get; set; } = 2;
    public int[] DiscreteActionSize { get; set; } = new int[] { 2 };
    public (float min, float max)[] ContinuousActionBounds { get; set; } = new (float, float)[] { (0f, 10f) };
    public bool isDone { get; set; }

    private float continousActionToGuess;
    private int discreteActionToGuess;

    public TrivialContinuousEnvironmentAsync()
    {
       // continousActionToGuess = 20f;
        
        Reset();
    }

    public Task<float[]> GetCurrentState()
    {
        if (isDone)
            Reset().Wait(); // Reset if done

        return Task.FromResult(new float[] { continousActionToGuess, discreteActionToGuess});
    }

    Random sharedRandom = new Random();
    public Task Reset()
    {
        return Task.Run(() =>
        {
            isDone = false;
            continousActionToGuess = 10 * (float)sharedRandom.NextDouble();
            discreteActionToGuess = Random.Shared.Next(2);
        });
        
    }

    public Task<(float reward, bool done)> Step(int[] discreteActions, float[] continuousActions)
    {
        return Task.Run(() =>
        {
            if (isDone)
                Reset().Wait(); // Reset if done

            float discreteOutput = discreteActions[0];
            float continuousOutput = continuousActions[0];


            float discreteReward = discreteActionToGuess == discreteOutput ? CorrectAnswerReward : WrongAnswerPenalty;
            float continuousReward = 1 - Math.Abs((continousActionToGuess - continuousOutput));
            Console.WriteLine($"Randomised num: {continousActionToGuess}, output num: {continuousOutput}");

            float totalReward = 0 + continuousReward;
            bool done = true;
            isDone = true;
            // Console.WriteLine($"Total reward: {totalReward}");

            return (totalReward, done);
        });
       
    }

    private static float RandomValue()
    {
        return Random.Shared.Next(2);
    }
}