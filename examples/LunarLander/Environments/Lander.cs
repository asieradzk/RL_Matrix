using OneOf;
using Gym.Rendering.WinForm;
using Gym.Environments.Envs.Aether;
using RLMatrix;

public class Lander : IEnvironment<float[]>
{
    public int stepCounter { get; set; }
    public int maxSteps { get; set; }
    public bool isDone { get; set; }
    public OneOf<int, (int, int)> stateSize { get; set;  }
    public int actionSize { get; set; }


    public Lander()
    {
        Initialise();
    }

    private float[] myState;
    public float[] GetCurrentState()
    {
        if(myState == null)
            myState = new float[8];

        return myState;
    }

    private LunarLanderEnv myEnv;
    public void Initialise()
    {
        myEnv = new LunarLanderEnv(WinFormEnvViewer.Factory);
        myEnv.Reset();
        stepCounter = 0;
        maxSteps = 1000;
        isDone = false;
        actionSize = myEnv.ActionSpace.Shape.Size;
        stateSize = myEnv.ObservationSpace.Shape.Size;
    }

    public void Reset()
    {
        myEnv.Reset();
        isDone = false;
        stepCounter = 0;

    }

    public float Step(int actionId)
    {
        var (observation, reward, _done, information) = myEnv.Step(actionId);

        var img = myEnv.Render();


        myState = observation.ToFloatArray();
        isDone = _done;

        if (stepCounter > maxSteps)
            isDone = true;

        return reward;
    }
}