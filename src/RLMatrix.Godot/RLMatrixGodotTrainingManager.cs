using Godot;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using RLMatrix;
using RLMatrix.Agents;
using RLMatrix.Godot.RLMatrix.Godot;
using RLMatrix.Agents.Common;

namespace RLMatrix.Godot
{

    public abstract partial class TrainingManagerBaseDiscrete<T, TState> : Node where T : GodotEnvironmentDiscrete<TState>
    {

        internal List<IEnvironmentAsync<TState>> myEnvironments = new();
        public LocalDiscreteRolloutAgent<TState> myAgent { get; set; }
        protected abstract LocalDiscreteRolloutAgent<TState> CreateAgent(List<IEnvironmentAsync<TState>> environments);

        public void SaveModel(string path)
        {
            myAgent.Save(path);
        }
        public void LoadModel(string path)
        {
            myAgent.Load(path);
        }

        public override void _Ready()
        {
            AppDomain.CurrentDomain.AssemblyResolve += CurrentDomain_AssemblyResolve;

            myEnvironments = GetAllChildrenOfType<IEnvironmentAsync<TState>>(this); // Get all children of type IEnvironment<TState>

            Console.WriteLine($"Training with envs: {myEnvironments.Count}");

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
        private List<T> GetAllChildrenOfType<T>(Node parentNode) where T : class
        {
            List<T> resultList = new List<T>();
            AddChildrenOfType(parentNode, resultList);
            return resultList;
        }

        private void AddChildrenOfType<T>(Node node, List<T> resultList) where T : class
        {
            foreach (Node child in node.GetChildren())
            {
                if (child is T typedChild)
                {
                    resultList.Add(typedChild);
                }
                AddChildrenOfType(child, resultList); // Recursive call to check the children of the current child
            }
        }
        int trainingStepCounter = 100000;
        int testingStepCounter = 20000;
        public override void _Process(double delta)
        {
            if (trainingStepCounter > 0)
            {
                myAgent.Step(true);
                trainingStepCounter -= 10;
                if (trainingStepCounter % 250 == 0)
                    Console.WriteLine($"Training steps remaining: {trainingStepCounter}");
            }
            else if (testingStepCounter > 0)
            {
                myAgent.Step(false);
                testingStepCounter -= 10;
            }
            else
            {
               Console.WriteLine("Training and testing complete");
                GetTree().Quit();
            }
        }




        //TODO: This is here because some reflection stuff was breaking, I think binary serialiser related.
        //Maybe can be removed after binary serialiser is gone
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


