using System;

namespace RLMatrix.Toolkit;

[AttributeUsage(AttributeTargets.Class)]
public class RLMatrixEnvironmentAttribute : Attribute;

[AttributeUsage(AttributeTargets.Method)]
public class RLMatrixObservationAttribute : Attribute;

[AttributeUsage(AttributeTargets.Method)]
public class RLMatrixActionDiscreteAttribute(int actionSize) : Attribute
{
    public int ActionSize { get; } = actionSize;
}

[AttributeUsage(AttributeTargets.Method)]
public class RLMatrixActionContinuousAttribute(float min = -1, float max = 1) : Attribute
{
    public float Min { get; } = min;
    public float Max { get; } = max;
}

[AttributeUsage(AttributeTargets.Method)]
public class RLMatrixRewardAttribute : Attribute;

[AttributeUsage(AttributeTargets.Method)]
public class RLMatrixDoneAttribute : Attribute;

[AttributeUsage(AttributeTargets.Method)]
public class RLMatrixResetAttribute : Attribute;

//----------------------------------------------Stubs for future semantics integration----------------------------------------------
[AttributeUsage(AttributeTargets.Class)]
public class RLMatrixEnvironmentDescriptionAttribute(string description) : Attribute
{
    public string Description { get; } = description;
}

[AttributeUsage(AttributeTargets.Method)]
public class RLMatrixObservationDescriptionAttribute(string description) : Attribute
{
    public string Description { get; } = description;
}

[AttributeUsage(AttributeTargets.Method)]
public class RLMatrixActionDescriptionAttribute(string description) : Attribute
{
    public string Description { get; } = description;
}