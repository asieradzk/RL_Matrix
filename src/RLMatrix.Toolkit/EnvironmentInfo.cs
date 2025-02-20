﻿using Microsoft.CodeAnalysis;
using System.Linq;
using RLMatrix.Common;

namespace RLMatrix.Toolkit;

public abstract class EnvironmentInfo
{
    public INamedTypeSymbol EnvironmentType { get; }
    public IMethodSymbol[] ObservationMethods { get; }
    public IMethodSymbol[] DiscreteActionMethods { get; }
    public IMethodSymbol[] RewardMethods { get; }
    public IMethodSymbol? DoneMethod { get; }
    public IMethodSymbol? ResetMethod { get; }
    public string? DoneMethodName => DoneMethod?.Name;
    public string? ResetMethodName => ResetMethod?.Name;

    protected EnvironmentInfo(INamedTypeSymbol environmentType)
    {
        EnvironmentType = environmentType;
        ObservationMethods = GetMethodsWithAttribute<RLMatrixObservationAttribute>();
        DiscreteActionMethods = GetMethodsWithAttribute<RLMatrixActionDiscreteAttribute>();
        RewardMethods = GetMethodsWithAttribute<RLMatrixRewardAttribute>();
        DoneMethod = GetMethodWithAttribute<RLMatrixDoneAttribute>();
        ResetMethod = GetMethodWithAttribute<RLMatrixResetAttribute>();
    }

    protected IMethodSymbol[] GetMethodsWithAttribute<T>() where T : System.Attribute
    {
        return EnvironmentType.GetMembers()
            .OfType<IMethodSymbol>()
            .Where(m => m.GetAttributes().Any(a => a.AttributeClass?.Name == typeof(T).Name))
            .ToArray();
    }

    protected IMethodSymbol? GetMethodWithAttribute<T>() where T : System.Attribute
    {
        return EnvironmentType.GetMembers()
            .OfType<IMethodSymbol>()
            .FirstOrDefault(m => m.GetAttributes().Any(a => a.AttributeClass?.Name == typeof(T).Name));
    }
}

public class DiscreteEnvironmentInfo : EnvironmentInfo
{
    public int[] ActionSizes { get; }

    public DiscreteEnvironmentInfo(INamedTypeSymbol environmentType) : base(environmentType)
    {
        ActionSizes = DiscreteActionMethods
            .Select(m => m.GetAttributes()
                .First(a => a.AttributeClass?.Name == nameof(RLMatrixActionDiscreteAttribute))
                .ConstructorArguments[0].Value)
            .Cast<int>()
            .ToArray();
    }
}

public class ContinuousEnvironmentInfo : EnvironmentInfo
{
    public IMethodSymbol[] ContinuousActionMethods { get; }
    public int[] DiscreteDimensions { get; }
    public ContinuousActionDimensions[] ContinuousActionDimensions { get; }

    public ContinuousEnvironmentInfo(INamedTypeSymbol environmentType) : base(environmentType)
    {
        ContinuousActionMethods = GetMethodsWithAttribute<RLMatrixActionContinuousAttribute>();
        DiscreteDimensions = DiscreteActionMethods.Length > 0
            ? DiscreteActionMethods
                .Select(m => m.GetAttributes()
                    .FirstOrDefault(a => a.AttributeClass?.Name == nameof(RLMatrixActionDiscreteAttribute))
                    ?.ConstructorArguments.FirstOrDefault().Value)
                .Where(v => v != null)
                .Cast<int>()
                .ToArray()
            : [];
        ContinuousActionDimensions = ContinuousActionMethods
            .Select(m =>
            {
                var attribute = m.GetAttributes().First(a => a.AttributeClass?.Name == nameof(RLMatrixActionContinuousAttribute));
                var (min, max) = attribute.ConstructorArguments.Length switch
                {
                    1 when attribute.ConstructorArguments[0].Value is float mx => (-1f, mx),
                    2 when (attribute.ConstructorArguments[0].Value, attribute.ConstructorArguments[1].Value) is (float mn, float mx) => (mn, mx),
                    _ => (-1f, 1f)
                };
                
                //var min = attribute.ConstructorArguments.Length > 0 ? (float)attribute.ConstructorArguments[0].Value! : -1f;
                //var max = attribute.ConstructorArguments.Length > 1 ? (float)attribute.ConstructorArguments[1].Value! : 1f;
                return new ContinuousActionDimensions(min, max);
            })
            .ToArray();
    }
}