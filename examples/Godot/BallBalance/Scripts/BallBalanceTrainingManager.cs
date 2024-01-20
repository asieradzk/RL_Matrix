using Godot;
using System;
using System.Collections.Generic;
using RLMatrix;
using RLMatrix.Godot;



//The inheritance is explained:
//Always inherit the TrainingManagerBase - this has all the guts wiring RLMatrix to Godot
//First generic parameter <T> is the environment type - this is the type of the environment you are training.
//Training manger will find all children of the T and step on them independently collecting experiences
//Second generic parameter <TState> is the shape of your observation vector.
//As reminder if you use float[,] then by default we use CNN, if you use float[] then we use feed forward network
public partial class BallBalanceTrainingManager : TrainingManagerBaseDiscrete<BallBalanceEnv, float[]>
{
    protected override IDiscreteAgent<float[]> CreateAgent(List<IEnvironment<float[]>> environments)
    {
        //you can of course load any options you want from a file if you do not wish to re-compile every time you change something
        var opts = new PPOAgentOptions(
            batchSize: 1,           // Number of EPISODES agent interacts with environment before learning from its experience
            memorySize: 10000,       // Size of the replay buffer
            gamma: 0.99f,          // Discount factor for rewards
            gaeLambda: 0.95f,      // Lambda factor for Generalized Advantage Estimation
            lr: 1e-5f,            // Learning rate
            clipEpsilon: 0.2f,     // Clipping factor for PPO's objective function
            vClipRange: 0.2f,      // Clipping range for value loss
            cValue: 0.5f,          // Coefficient for value loss
            ppoEpochs: 1,            // Number of PPO epochs
            clipGradNorm: 0.5f,    // Maximum allowed gradient norm
            displayPlot: null
        );
        
        return new PPOAgent<float[]>(opts, environments);
    }
    
    
    //Alternatively you can do the same to use DQN:
    /*
    protected override IDiscreteAgent<float[]> CreateAgent(List<IEnvironment<float[]>> enviroments)
    {
        var opts = new DQNAgentOptions();
        return new DQNAgent<float[]>(opts, enviroments);
    }
    */
}
