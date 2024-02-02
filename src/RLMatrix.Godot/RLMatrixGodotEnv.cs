using Godot;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using RLMatrix;
using OneOf;



namespace RLMatrix.Godot
{
    public abstract partial class GodotEnvironmentDiscrete<TState> : Node, IEnvironment<TState>
    {

        public abstract List<List<Action>> myHeads { get; set; }
        public abstract Action resetProvider { get; }
        public abstract Func<TState> observationProvider { get; }
        public abstract Func<bool> isDoneProvider { get; }
        public float rewardRepository = 0f;

        public void AddReward(float reward)
        {
            rewardRepository += reward;
        }

        public override void _Ready()
        {
            base._Ready();
            Reset();

            
        }

        public void Initialise()
        {
          
            this.actionSize = new int[myHeads.Count];
        
            for (int i = 0; i < myHeads.Count; i++)
            {
                this.actionSize[i] = myHeads[i].Count;
            }


            //here assign stateSize
            if (typeof(TState) == typeof(float[]))
            {

                var probeObs = observationProvider() as float[];
                stateSize = (probeObs).Length;
            }
            else if (typeof(TState) == typeof(float[,]))
            {
                var probeObs = observationProvider() as float[,];
                stateSize = (probeObs.GetLength(0), probeObs.GetLength(1));
            }
            stepCounter = 0;
        }

        public TState GetCurrentState()
        {
            return observationProvider();
        }

        public void Reset()
        {
            resetProvider();
            Initialise();
        }

        public float Step(int[] actionsIds)
        {
            for (int i = 0; i < actionsIds.Length; i++)
            {

                myHeads[i][actionsIds[i]]();
            }
            stepCounter++;

            var reward = rewardRepository;
            rewardRepository = 0f;
            return reward;
        }

        public int stepCounter { get; set; }
        public abstract int maxSteps { get; set; }
        public bool isDone
        {
            get
            {
                if (stepCounter >= maxSteps)
                {
                    return true;
                }

                return isDoneProvider();
            }

            set
            {

            }
        }
        public OneOf<int, (int, int)> stateSize { get; set; }
        public int[] actionSize { get; set; }
    }
}