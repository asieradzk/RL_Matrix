using System;
using System.Threading.Tasks;
using OneOf;
using RLMatrix;

public class SequencePushEnv : IEnvironmentAsync<float[]>
{
    public int stepCounter { get; set; }
    public int maxSteps { get; set; }
    public bool isDone { get; set; }
    public OneOf<int, (int, int)> stateSize { get; set; }
    public int[] actionSize { get; set; }

    float state;
    float[] directions;
    int direction;
    Random random = new Random();
    float previousStep;

    public SequencePushEnv()
    {
       InitialiseAsync();
    }

    public Task<float[]> GetCurrentState()
    {
        if (isDone)
            Reset().Wait(); // Reset if done

        return Task.FromResult(new float[] { stepCounter, directions[0], directions[1], directions[2], directions[3] });
    }

    public void InitialiseAsync()
    {
        directions = new float[] { 0f, 0f, 0f, 0f };
        maxSteps = 20;
        isDone = false;
        stateSize = 5;
        actionSize = new int[] { 4 };
        direction = random.Next(0, 3);
        //set direction to 1 depending on index
        directions[direction] = 1f;
        stepCounter = 1;
    }

    public Task Reset()
    {
        InitialiseAsync();
        return Task.CompletedTask;
    }

    public Task<(float, bool)> Step(int[] actionsIds)
    {
        if(isDone)
            Reset().Wait(); // Reset if done

        stepCounter++;

        if (stepCounter > 4)
        {
            directions = new float[] { 0f, 0f, 0f, 0f };
        }

        if (stepCounter > maxSteps)
        {
            isDone = true;
        }

        float reward = actionsIds[0] == direction ? 1f : -2f;
        return Task.FromResult((reward, isDone));
    }
}