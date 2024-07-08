// See https://aka.ms/new-console-template for more information
using RLMatrix;
using RLMatrix.Agents.Common;
using RLMatrix.Agents.SignalR;
using RLMatrix.WinformsChart;

Console.WriteLine("Hello, World!");

var myChart = new WinformsChart();
var myChart2 = new WinformsChart();

var optsppo = new PPOAgentOptions(
    batchSize: 32,           // Nu8mber of EPISODES agent interacts with environment before learning from its experience
    memorySize: 10000,       // Size of the replay buffer
    gamma: 0.99f,          // Discount factor for rewards
    gaeLambda: 0.95f,      // Lambda factor for Generalized Advantage Estimation
    lr: 1e-3f,            // Learning rate
    width: 512,
    depth: 2,
    clipEpsilon: 0.2f,     // Clipping factor for PPO's objective function
    vClipRange: 0.2f,      // Clipping range for value loss
    cValue: 0.5f,          // Coefficient for value loss
    ppoEpochs: 7,            // Number of PPO epoch
    clipGradNorm: 0.5f,
    entropyCoefficient: 0.005f,
    useRNN: false
   );

var optsdqn = new DQNAgentOptions(numAtoms: 51,
    batchedActionProcessing: true,
    boltzmannExploration: true,
    prioritizedExperienceReplay: true, 
    nStepReturn: 200, duelingDQN: true, 
    doubleDQN: true, noisyLayers: true, 
    noisyLayersScale: 0.02f, 
    categoricalDQN: true, 
    batchSize: 128,
    memorySize: 10000, 
    gamma: 0.99f, 
    epsStart: 1f, 
    epsEnd: 0.05f, 
    epsDecay: 150f, 
    tau: 0.005f, 
    lr: 5e-3f, 
    width: 512,
    depth: 2);



var env = new List<IContinuousEnvironmentAsync<float[]>> { new TrivialContinuousEnvironmentAsync(), };
var env2 = new List<IContinuousEnvironmentAsync<float[]>> { new TrivialContinuousEnvironmentAsync()};

var agent = new RemoteContinuousRolloutAgent<float[]>("http://127.0.0.1:5006/rlmatrixhub", optsppo, env, myChart);
var agent2 = new RemoteContinuousRolloutAgent<float[]>("http://127.0.0.1:5006/rlmatrixhub", optsppo, env2, myChart2);


for (int i = 0; i < 20000; i++)
{
    await agent.Step();
    await agent2.Step();
}