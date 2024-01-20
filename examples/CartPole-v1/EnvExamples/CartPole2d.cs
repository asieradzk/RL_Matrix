using Gym.Environments.Envs.Classic;
using Gym.Rendering.WinForm;
using OneOf;
using RLMatrix;

public class CartPole2d : IEnvironment<float[,]>
{
    public int stepCounter { get; set; }
    public int maxSteps { get; set; }
    public bool isDone { get; set; }
    public OneOf<int, (int, int)> stateSize { get; set; }
    public int[] actionSize { get; set; }

    CartPoleEnv myEnv;

    private float[,] myState;

    public CartPole2d()
    {
        //TODO: it should be more obvious to put Initalise method in constructor or something since its not called later. 
        //Create base constructor that calls initialise method if not initialised???
        Initialise();
    }

    public float[,] GetCurrentState()
    {
        if (myState == null)
            myState = new float[2, 2];

        var myStateCopy = new float[2, 2];

        for (int i = 0; i < myState.GetLength(0); i++)
        {
            for (int j = 0; j < myState.GetLength(1); j++)
            {
                myStateCopy[i, j] = myState[i, j];
            }
        }

        return myStateCopy;
    }

    public void Initialise()
    {
        myEnv = new CartPoleEnv(WinFormEnvViewer.Factory);
        stepCounter = 0;
        maxSteps = 100000;
        stateSize = (2, 2);
        actionSize = new int[] { myEnv.ActionSpace.Shape.Size };
        myEnv.Reset();
        isDone = false;

    }

    public void Reset()
    {
        myState = new float[2, 2];
        myEnv.Reset();
        isDone = false;
        stepCounter = 0;
    }

    public float Step(int[] actionsIds)
    {
        var actionId = actionsIds[0];

        var (observation, reward, _done, information) = myEnv.Step(actionId);

        SixLabors.ImageSharp.Image img = myEnv.Render(); //returns the image that was rendered.
                                                         // form.Invoke(new Action(() => ChangeImage(img)));


        // Thread.Sleep(1); //this is to prevent it from finishing instantly !

        float[] observationArray = observation.ToFloatArray();
        myState[0, 0] = observationArray[0];
        myState[0, 1] = observationArray[1];
        myState[1, 0] = observationArray[2];
        myState[1, 1] = observationArray[3];
        isDone = _done;

        if (stepCounter > maxSteps)
            isDone = true;

        return reward;


    }


    ~CartPole2d()
    {
        myEnv.CloseEnvironment();
        myEnv.Dispose();
    }
}
