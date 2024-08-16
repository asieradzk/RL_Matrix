![RLMatrixBanner](https://i.imgur.com/x5x6Ph8.png)

# 🚀 RL Matrix - Pure C# Deep Reinforcement Learning Experience with TorchSharp!

Dive into the future of type-safe Deep Reinforcement Learning with .NET & RL Matrix, powered by the might of TorchSharp. RL Matrix stands out as a user-friendly toolkit offering a collection of RL algorithms—primed for plug, play, and prosper!
[![NuGet](https://img.shields.io/nuget/v/RLMatrix.svg?label=RLMatrix-nuget)](https://www.nuget.org/packages/RLMatrix/)
[![NuGet](https://img.shields.io/nuget/v/RLMatrix.WinformsChart.svg?label=WinformsChart-nuget)](https://www.nuget.org/packages/RLMatrix.WinformsChart/)
[![NuGet](https://img.shields.io/nuget/v/RLMatrix.Godot.svg?label=RLMatrix.Godot-nuget)](https://www.nuget.org/packages/RLMatrix.Godot/)[
![TorchSharp](https://img.shields.io/badge/Made%20With-TorchSharp-8A2BE2)](https://github.com/dotnet/TorchSharp)
![discord](https://img.shields.io/badge/-discord-link?label=join&link=https%3A%2F%2Fdiscord.gg%2FRYnyUQGd)


## 🌟 Featured Algorithms:

 - PPO 
 - DQN
 - Both have 1D (Feed forward) and 2D (CNN) variants
 - 0.1.2 Adds multi-head continous (PPO) discrete (PPO, DQN) and mixed (PPO) actions. See IEnvironment and IContinousEnvironment.
 - 0.2.0 Adds working-ish PPO GAIL. And overhauls training method for stepwise
 - 0.2.0 Adds multi-environment training
 - 0.2.0 Includes Godot examples and RLMatrix.Godot nuget package for easy setup
 - Only tested single-head discrete output so please open issue if it doesnt work.

## 🎯 What Sparks RL Matrix?

While embarking on my RL journey, I sensed a gap in the reinforcement learning world even with TorchSharp's solid foundation. It struck me—C# is the ideal choice for RL outside research circles, thanks to its pristine and intuitive coding experience. No more guessing games in environment or agent building!

With RL Matrix, our vision is to offer a seamless experience. By simply incorporating the IEnvironment interface, you're equipped to rapidly craft and unleash Reinforcement Learning Agents (RL Agents). Switching between algorithms? A breeze! It’s our nod to the elegance of Matlab's toolkit methodology.

![enter image description here](https://i.imgur.com/jTKghOP.gif)
![enter image description here](https://s13.gifyu.com/images/S0a0u.gif)

## 🚀 Getting Started:

Peek into the `/examples/` directory for illustrative code. But to give you a quick start:

1. **Craft an IEnvironment class:** Comply with reinforcement learning guidelines, defining your observation shapes and action count:

  ```cs
  public class CartPole : IEnvironment<float[]>
{
    public int stepCounter { get; set; }
    public int maxSteps { get; set; }
    public bool isDone { get; set; }
    public OneOf<int, (int, int)> stateSize { get; set; }
    public int actionSize { get; set; }

    CartPoleEnv myEnv;

    private float[] myState;

    public CartPole()
    {
        Initialise();
    }

    public float[] GetCurrentState()
    {
        if (myState == null)
            myState = new float[4] {0,0,0,0};
        return myState;
    }

    public void Initialise()
    {
        myEnv = new CartPoleEnv(WinFormEnvViewer.Factory);
        stepCounter = 0;
        maxSteps = 100000;
        stateSize = myEnv.ObservationSpace.Shape.Size;
        actionSize = myEnv.ActionSpace.Shape.Size;
        myEnv.Reset();
        isDone = false; 
    }

    public void Reset()
    {
    //For instance:
        myEnv.Reset();
        isDone = false;
        stepCounter = 0;
    }

    public float Step(int actionId)
    {
    //Whatever step logic, returns reward
        return reward;
    }
}
```

2. Agent Instance & Training: Spawn an agent for your environment and ignite the Step method:
```cs
var opts = new DQNAgentOptions(batchSize: 64, memorySize: 10000, gamma: 0.99f, epsStart: 1f, epsEnd: 0.05f, epsDecay: 50f, tau: 0.005f, lr: 1e-4f, displayPlot: myChart);
var env = new List<IEnvironment<float[]>> { new CartPole(), new CartPole() };
var myAgent = new DQNAgent<float[]>(opts, env);

for (int i = 0; i < 10000; i++)
{
    myAgent.Step();
}
```
Notice that TrainEpisode method was removed. 
## 📌 Current Roadmap:
-Add RNN support for PPO and DQN
-Add variations for multi-head output for PPO and DQN
-More Godot examples testing multi-head continous+discrete action spaces
-Create Godot plugin
-Fully develop workflow for Gail and imitation learning
As we innovate, anticipate breaking changes. We'll keep you in the loop!
## 💌 Get in Touch:
Questions? Ideas? Collaborations? Drop a line at:
📧 contact@exmachinasoft.com

## 🤝 License:
RLMatrix is licensed under the [RLMatrix Comprehensive Dual License Agreement](https://github.com/asieradzk/RL_Matrix/blob/master/LICENSE.txt)

