using System;

namespace RLMatrix.Toolkit
{
    internal static class AdditionalFiles
    {
        internal const string RLMatrixPoolingHelper = @"
using System;
using System.Collections.Generic;
namespace RLMatrix.Toolkit
{
    public class RLMatrixPoolingHelper
    {
        private int poolingRate;
        private Queue<float[]> observationBuffer;
        private float[] lastAction;
        private Func<float[]> getObservationFunc;
        private int singleObservationSize;
        private float accumulatedReward;
        public bool HasAction { get; private set; }
        public RLMatrixPoolingHelper(int rate, int actionSize, Func<float[]> getObservation)
        {
            poolingRate = rate;
            lastAction = new float[actionSize];
            getObservationFunc = getObservation;
            HasAction = false;
            singleObservationSize = getObservation().Length;
            accumulatedReward = 0f;
            observationBuffer = new Queue<float[]>(poolingRate);
            InitializeObservations();
        }
        private void InitializeObservations()
        {
            for (int i = 0; i < poolingRate; i++)
            {
                CollectObservation(0f);
            }
        }
        public void SetAction(float[] action)
        {
            Array.Copy(action, lastAction, action.Length);
            HasAction = true;
        }
        public float[] GetLastAction()
        {
            return lastAction;
        }
        public void CollectObservation(float reward)
        {
            float[] currentObservation = getObservationFunc();
            if (observationBuffer.Count >= poolingRate)
            {
                observationBuffer.Dequeue();
            }
            observationBuffer.Enqueue(currentObservation);
            accumulatedReward += reward;
        }
        public float[] GetPooledObservations()
        {
            float[] pooledObservations = new float[singleObservationSize * poolingRate];
            int index = 0;
            foreach (var observation in observationBuffer)
            {
                Array.Copy(observation, 0, pooledObservations, index, singleObservationSize);
                index += singleObservationSize;
            }
            return pooledObservations;
        }
        public float GetAndResetAccumulatedReward()
        {
            float reward = accumulatedReward;
            accumulatedReward = 0f;
            return reward;
        }
        public void Reset()
        {
            observationBuffer.Clear();
            HasAction = false;
            accumulatedReward = 0f;
            InitializeObservations();
        }
        public void HardReset(Func<float[]> getInitialObservation)
        {
            observationBuffer.Clear();
            HasAction = false;
            accumulatedReward = 0f;
            lastAction = new float[lastAction.Length];  // Reset the last action
            // Fill the buffer with new observations
            for (int i = 0; i < poolingRate; i++)
            {
                float[] newObservation = getInitialObservation();
                observationBuffer.Enqueue(newObservation);
            }
        }
    }
}";

        internal const string IRLMatrixExtraObservationSource = @"
using System.Text;
namespace RLMatrix.Toolkit
{
    public interface IRLMatrixExtraObservationSource
    {
        float[] GetObservations();
        int GetObservationSize();
    }
}";

        internal const string RLMatrixAttributes = @"
using System;
namespace RLMatrix.Toolkit
{
    [AttributeUsage(AttributeTargets.Class)]
    public class RLMatrixEnvironmentAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Method)]
    public class RLMatrixObservationAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Method)]
    public class RLMatrixActionDiscreteAttribute : Attribute
    {
        public int ActionSize { get; }
        public RLMatrixActionDiscreteAttribute(int actionSize)
        {
            ActionSize = actionSize;
        }
    }
    [AttributeUsage(AttributeTargets.Method)]
    public class RLMatrixActionContinuousAttribute : Attribute
    {
        public float Min { get; }
        public float Max { get; }
        public RLMatrixActionContinuousAttribute(float min = -1, float max = 1)
        {
            Min = min;
            Max = max;
        }
    }
    [AttributeUsage(AttributeTargets.Method)]
    public class RLMatrixRewardAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Method)]
    public class RLMatrixDoneAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Method)]
    public class RLMatrixResetAttribute : Attribute { }
}";
    }
}