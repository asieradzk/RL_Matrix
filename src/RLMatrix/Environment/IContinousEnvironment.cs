using OneOf;
using System;

namespace RLMatrix
{
    /// <summary>
    /// This interface defines the structure of a continuous environment in which the reinforcement learning algorithm operates.
    /// It accommodates scenarios where agents can produce both continuous and discrete actions simultaneously.
    /// </summary>
    /// <typeparam name="TState">The type that describes the state of the environment.</typeparam>
    public interface IContinuousEnvironment<TState> : IEnvironment<TState>
    {

        /// <summary>
        /// Represents the dimensionality of the environment's continuous action space.
        /// Each element indicates the range for each continuous action.
        /// For instance, if a continuous action can range between -1 and 1, the tuple will be (-1, 1).
        /// </summary>
        (float min, float max)[] continuousActionBounds { get; set; }

        float Step(int[] discreteActions, float[] continuousActions);
    }

    //Can this be done without adapter? 
    public static class ContinuousEnvironmentFactory
    {
        public static IContinuousEnvironment<TState> Create<TState>(IEnvironment<TState> env)
        {
            if (env is IContinuousEnvironment<TState> continuousEnv)
            {
                // If the environment is already continuous, just return it as-is.
                return continuousEnv;
            }

            return new ContinuousEnvironmentAdapter<TState>(env);
        }

        private class ContinuousEnvironmentAdapter<TState> : IContinuousEnvironment<TState>
        {
            private readonly IEnvironment<TState> _env;

            public ContinuousEnvironmentAdapter(IEnvironment<TState> env)
            {
                _env = env;
            }

            public int stepCounter { get => _env.stepCounter; set => _env.stepCounter = value; }
            public int maxSteps { get => _env.maxSteps; set => _env.maxSteps = value; }
            public bool isDone { get => _env.isDone; set => _env.isDone = value; }
            public OneOf<int, (int, int)> stateSize { get => _env.stateSize; set => _env.stateSize = value; }
            public int[] actionSize { get => _env.actionSize; set => _env.actionSize = value; }

            public (float min, float max)[] continuousActionBounds { get; set; } = new (float, float)[0];

            public void Initialise() => _env.Initialise();
            public TState GetCurrentState() => _env.GetCurrentState();
            public void Reset() => _env.Reset();
            public float Step(int[] actionsIds) => _env.Step(actionsIds);
            //??
            public float Step(int[] discreteActions, float[] continuousActions) => _env.Step(discreteActions);
        }
    }




}
