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
}