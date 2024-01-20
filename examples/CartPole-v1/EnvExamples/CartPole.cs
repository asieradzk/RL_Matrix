using Gym.Environments.Envs.Classic;
using Gym.Rendering.WinForm;
using OneOf;
using RLMatrix;

public class CartPole : IEnvironment<float[]>
{
    public int stepCounter { get; set; }
    public int maxSteps { get; set; }
    public bool isDone { get; set; }
    public OneOf<int, (int, int)> stateSize { get; set; }
    public int[] actionSize { get; set; }

    CartPoleEnv myEnv;

    private float[] myState;

    public CartPole()
    {
        //TODO: it should be more obvious to put Initalise method in constructor or something since its not called later. 
        //Create base constructor that calls initialise method if not initialised???
        Initialise();
    }

    public float[] GetCurrentState()
    {
        if (myState == null)
            myState = new float[4] { 0, 0, 0, 0 };
        return myState;
    }

    public void Initialise()
    {
        myEnv = new CartPoleEnv(WinFormEnvViewer.Factory);
        stepCounter = 0;
        maxSteps = 100000;
        stateSize = myEnv.ObservationSpace.Shape.Size;
        actionSize = new int[] { myEnv.ActionSpace.Shape.Size };
        myEnv.Reset();
        isDone = false;
    }

    public void Reset()
    {
        myEnv.Reset();
        isDone = false;
        stepCounter = 0;
    }

    public float Step(int[] actionsIds)
    {
        var actionId = actionsIds[0];

        var (observation, reward, _done, information) = myEnv.Step(actionId);
        SixLabors.ImageSharp.Image img = myEnv.Render();
        myState = observation.ToFloatArray();
        isDone = _done;

        if (stepCounter > maxSteps)
            isDone = true;

        if (isDone)
        {
            reward = 0;
        }

        return reward;
    }
}
