using RLMatrix;
using RLMatrix.WinformsChart;
// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");

var myChart = new WinformsChart();

//PPO
var optsppo = new PPOAgentOptions(
    batchSize: 32,           // Number of steps agent interacts with environment before learning from its experience
    memorySize: 10000,       // Size of the replay buffer
    gamma: 0.99f,          // Discount factor for rewards
    gaeLambda: 0.95f,      // Lambda factor for Generalized Advantage Estimation
    lr: 3e-6f,            // Learning rate
    clipEpsilon: 0.2f,     // Clipping factor for PPO's objective function
    vClipRange: 0.2f,      // Clipping range for value loss
    cValue: 0.5f,          // Coefficient for value loss
    ppoEpochs: 4,            // Number of PPO epochs
    clipGradNorm: 0.5f,    // Maximum allowed gradient norm
    displayPlot: myChart
   );

var envppo = new CartPole();
var myAgentppo = new PPOAgent<float[]>(optsppo, envppo);

for (int i = 0; i < 400; i++)
{
   // myAgentppo.TrainEpisode();
}






//DQN
var opts = new DQNAgentOptions(batchSize: 64, memorySize: 10000, gamma: 0.99f, epsStart: 1f, epsEnd: 0.05f, epsDecay: 50f, tau: 0.005f, lr: 1e-4f, displayPlot: myChart);
var env = new CartPole();
var myAgent = new D2QNAgent<float[]>(opts, env);



for (int i = 0; i < 400; i++)
{
    myAgent.TrainEpisode();
}


Console.ReadLine();