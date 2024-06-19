using System;
using System.Threading.Tasks;
using OneOf;
using RLMatrix;

public class SequenceGuessingEnv : IEnvironmentAsync<float[]>
{
    public int stepCounter { get; set; }
    public int maxSteps { get; set; }
    public bool isDone { get; set; }
    public OneOf<int, (int, int)> stateSize { get; set; }
    public int[] actionSize { get; set; }

    float state;
    int randomLength;
    Random random = new Random();

    public SequenceGuessingEnv()
    {
        InitialiseAsync();
    }

    public Task<float[]> GetCurrentState()
    {
        if (isDone)
            Reset().Wait(); // Reset if done

        return Task.FromResult(new float[] { state, stepCounter });
    }

    public void InitialiseAsync()
    {
        maxSteps = 50;
        isDone = false;
        stateSize = 2;
        actionSize = new int[] { 5 };
        randomLength = random.Next(1, 47);
        stepCounter = 1;
        state = randomLength;
    }

    public Task Reset()
    {
        InitialiseAsync();
        return Task.CompletedTask;
    }

    public Task<(float, bool)> Step(int[] actionsIds)
    {
        if (isDone)
            Reset().Wait(); // Reset if done

        stepCounter++;
        if (actionsIds[0] == 1)
        {
            var reward = 100 - 20 * Math.Abs(stepCounter - randomLength);
            Console.WriteLine($"Finished with reward {reward} at {stepCounter} steps");
            isDone = true;


            return Task.FromResult(((float)reward, isDone));
        }
        else
        {
            if (stepCounter > 3)
            {
                state = stepCounter;
            }

            if (stepCounter >= maxSteps)
            {
                var reward = -20 * Math.Abs(stepCounter - randomLength);
                Console.WriteLine($"Finished with reward {reward} at {stepCounter} steps");
                isDone = true;
                return Task.FromResult(((float)reward, isDone));
            }

            return Task.FromResult((0.1f, isDone));
        }
    }
}