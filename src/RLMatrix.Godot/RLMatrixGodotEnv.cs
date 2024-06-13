using Godot;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using RLMatrix;
using OneOf;



namespace RLMatrix.Godot
{
    using Godot;
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Threading.Tasks;
    using RLMatrix;
    using OneOf;

    namespace RLMatrix.Godot
    {
        public abstract partial class GodotEnvironmentDiscrete<TState> : Node, IEnvironmentAsync<TState>
        {
            public abstract List<List<Func<ValueTask>>> myHeads { get; set; }
            public abstract Func<ValueTask> resetProvider { get; }
            public abstract Func<ValueTask<TState>> observationProvider { get; }
            public abstract Func<ValueTask<bool>> isDoneProvider { get; }

            public float rewardRepository = 0f;

            public void AddReward(float reward)
            {
                rewardRepository += reward;
            }

            public override void _Ready()
            {
                base._Ready();
                Reset().GetAwaiter().GetResult();
            }

            public async Task Initialise()
            {
                this.actionSize = new int[myHeads.Count];
                for (int i = 0; i < myHeads.Count; i++)
                {
                    this.actionSize[i] = myHeads[i].Count;
                }

                //here assign stateSize
                if (typeof(TState) == typeof(float[]))
                {
                    var probeObs = await observationProvider() as float[];
                    stateSize = (probeObs).Length;
                }
                else if (typeof(TState) == typeof(float[,]))
                {
                    var probeObs = await observationProvider() as float[,];
                    stateSize = (probeObs.GetLength(0), probeObs.GetLength(1));
                }
            }

            public async Task<TState> GetCurrentState()
            {
                if (await isDoneProvider())
                {
                    await Reset();
                }
                
                return await observationProvider();
            }

            public async Task Reset()
            {
                await resetProvider();
                await Initialise();
            }

            public async Task<(float, bool)> Step(int[] actionsIds)
            {
                for (int i = 0; i < actionsIds.Length; i++)
                {
                    await myHeads[i][actionsIds[i]]();
                }

                var reward = rewardRepository;
                rewardRepository = 0f;

                var done = await isDoneProvider();
                
                if (done)
                {
                    await Reset();
                }

                return (reward, done);
            }
            public OneOf<int, (int, int)> stateSize { get; set; }
            public int[] actionSize { get; set; }
        }
    }
}