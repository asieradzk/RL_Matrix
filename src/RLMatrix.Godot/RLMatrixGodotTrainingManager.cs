using Godot;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using RLMatrix;
using RLMatrix.Agents.PPO.Variants;
using RLMatrix.Agents;

namespace RLMatrix.Godot
{

    public abstract partial class TrainingManagerBaseDiscrete<T, TState> : Node where T : GodotEnvironmentDiscrete<TState>
    {

        internal List<IEnvironment<TState>> myEnvironments = new();
        public IDiscreteAgent<TState> myAgent { get; set; }
        protected abstract IDiscreteAgent<TState> CreateAgent(List<IEnvironment<TState>> environments);

        public override void _Ready()
        {
            AppDomain.CurrentDomain.AssemblyResolve += CurrentDomain_AssemblyResolve;

            var children = GetChildren();
            foreach (var child in children)
            {
                if (child is T)
                {
                    myEnvironments.Add((T)child);
                }
            }

            Debug.WriteLine(myEnvironments.Count);

            /* GAIL EXAMPLE
            var recorder = new Recorder<TState>();
            recorder.Load(pathToGAILDemo);
            var mem = recorder.myMemory;
            var gailOptions = new GAILOptions();
            gailOptions.rewardFactor = 0.01f;
            gailOptions.BatchSize = 128;
            var myGAIL = new GAIL<TState>(mem, gailOptions);
            */

            myAgent = CreateAgent(myEnvironments);
        }

        public override void _Process(double delta)
        {
            myAgent.Step();
        }




        //This is here because some relection stuff was breaking, I think binary serialiser related.
        System.Reflection.Assembly CurrentDomain_AssemblyResolve(object sender, ResolveEventArgs args)
        {
            string assemblyName = new System.Reflection.AssemblyName(args.Name).Name;
            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                if (assembly.GetName().Name == assemblyName)
                {
                    return assembly;
                }
            }
            return null;
        }


    }
}


