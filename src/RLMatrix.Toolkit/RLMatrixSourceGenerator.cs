using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;

namespace RLMatrix.Toolkit
{
    [Generator]
    public class RLMatrixSourceGenerator : IIncrementalGenerator
    {
        public void Initialize(IncrementalGeneratorInitializationContext context)
        {
            context.RegisterPostInitializationOutput(ctx =>
            {
                ctx.AddSource("RLMatrixPoolingHelper.g.cs", SourceText.From(AdditionalFiles.RLMatrixPoolingHelper, Encoding.UTF8));
                ctx.AddSource("IRLMatrixExtraObservationSource.g.cs", SourceText.From(AdditionalFiles.IRLMatrixExtraObservationSource, Encoding.UTF8));
                ctx.AddSource("RLMatrixAttributes.g.cs", SourceText.From(AdditionalFiles.RLMatrixAttributes, Encoding.UTF8));
            });

            var environmentClasses = context.SyntaxProvider
                .ForAttributeWithMetadataName(
                    "RLMatrix.Toolkit.RLMatrixEnvironmentAttribute",
                    predicate: static (node, _) => node is ClassDeclarationSyntax,
                    transform: static (ctx, _) => GetEnvironmentModel(ctx))
                .Where(static m => m is not null);

            context.RegisterSourceOutput(environmentClasses,
                static (spc, model) => GenerateEnvironmentClass(spc, model!));
        }

        private static EnvironmentModel? GetEnvironmentModel(GeneratorAttributeSyntaxContext context)
        {
            if (context.TargetSymbol is not INamedTypeSymbol classSymbol)
                return null;

            var className = classSymbol.Name;
            var namespaceName = classSymbol.ContainingNamespace?.ToDisplayString() ?? "global";

            var observationMethods = GetMethodsWithAttribute(classSymbol, "RLMatrixObservationAttribute");
            var discreteActionMethods = GetMethodsWithAttribute(classSymbol, "RLMatrixActionDiscreteAttribute");
            var continuousActionMethods = GetMethodsWithAttribute(classSymbol, "RLMatrixActionContinuousAttribute");
            var rewardMethods = GetMethodsWithAttribute(classSymbol, "RLMatrixRewardAttribute");
            var doneMethod = GetMethodWithAttribute(classSymbol, "RLMatrixDoneAttribute");
            var resetMethod = GetMethodWithAttribute(classSymbol, "RLMatrixResetAttribute");

            if (resetMethod == null || doneMethod == null)
                return null;

            var discreteActionInfos = discreteActionMethods
                .Select(m => new ActionInfo(
                    m.Name,
                    GetAttributeArgument<int>(m, "RLMatrixActionDiscreteAttribute", 0)))
                .ToArray();

            var continuousActionInfos = continuousActionMethods
                .Select(m => new ContinuousActionInfo(
                    m.Name,
                    GetAttributeArgument<float>(m, "RLMatrixActionContinuousAttribute", 0, -1f),
                    GetAttributeArgument<float>(m, "RLMatrixActionContinuousAttribute", 1, 1f)))
                .ToArray();

            var observationInfos = observationMethods
                .Select(m => new ObservationInfo(
                    m.Name,
                    m.ReturnType.SpecialType == SpecialType.System_Single))
                .ToArray();

            return new EnvironmentModel(
                className,
                namespaceName,
                observationInfos,
                discreteActionInfos,
                continuousActionInfos,
                rewardMethods.Select(m => m.Name).ToArray(),
                doneMethod.Name,
                resetMethod.Name,
                continuousActionMethods.Any());
        }

        private static IMethodSymbol[] GetMethodsWithAttribute(INamedTypeSymbol type, string attributeName)
        {
            return type.GetMembers()
                .OfType<IMethodSymbol>()
                .Where(m => m.GetAttributes().Any(a => a.AttributeClass?.Name == attributeName))
                .ToArray();
        }

        private static IMethodSymbol? GetMethodWithAttribute(INamedTypeSymbol type, string attributeName)
        {
            return type.GetMembers()
                .OfType<IMethodSymbol>()
                .FirstOrDefault(m => m.GetAttributes().Any(a => a.AttributeClass?.Name == attributeName));
        }

        private static T GetAttributeArgument<T>(IMethodSymbol method, string attributeName, int index, T defaultValue = default!)
        {
            var attr = method.GetAttributes().FirstOrDefault(a => a.AttributeClass?.Name == attributeName);
            if (attr?.ConstructorArguments.Length > index)
                return (T)attr.ConstructorArguments[index].Value!;
            return defaultValue;
        }

        private static void GenerateEnvironmentClass(SourceProductionContext context, EnvironmentModel model)
        {
            var source = GenerateSource(model);
            context.AddSource($"{model.ClassName}.g.cs", SourceText.From(source, Encoding.UTF8));
        }

        private static string GenerateSource(EnvironmentModel model)
        {
            var interfaceName = model.IsContinuous ? "IContinuousEnvironmentAsync<float[]>" : "IEnvironmentAsync<float[]>";
            var actionProperties = model.IsContinuous ? GenerateContinuousProperties(model) : GenerateDiscreteProperties(model);
            var stepMethod = model.IsContinuous ? GenerateContinuousStepMethod(model) : GenerateDiscreteStepMethod(model);
            var actionSizeCalc = model.IsContinuous ? "DiscreteActionSize.Length + ContinuousActionBounds.Length" : "actionSize.Length";
            var stateSizeProp = model.IsContinuous ? "StateSize" : "stateSize";
            var rewardSum = model.RewardMethods.Any() ? string.Join(" + ", model.RewardMethods.Select(m => $"{m}()")) : "0f";
            var actionMethodsInit = GenerateActionMethodsInitialization(model);
            var observationCollection = GenerateObservationCollection(model);
            var ghostStepActions = GenerateGhostStepActions(model);

            return $$"""
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using OneOf;
using RLMatrix;
using RLMatrix.Toolkit;

namespace {{model.NamespaceName}}
{
    public partial class {{model.ClassName}} : {{interfaceName}}
    {
        private int _poolingRate;
        private RLMatrixPoolingHelper _poolingHelper;
        private List<IRLMatrixExtraObservationSource> _extraObservationSources;
        private int _stepsSoft;
        private int _stepsHard;
        private int _maxStepsHard;
        private int _maxStepsSoft;
        private bool _rlMatrixEpisodeTerminated;
        private bool _rlMatrixEpisodeTruncated;
        private (Action<int> method, int maxValue)[] _actionMethodsWithCaps;
        private Action<float>[] _continuousActionMethods;

{{actionProperties}}

        public virtual {{model.ClassName}} RLInit(int poolingRate = 1, int maxStepsHard = 1000, int maxStepsSoft = 100, List<IRLMatrixExtraObservationSource> extraObservationSources = null)
        {
            if (poolingRate != 1)
            {
                throw new NotImplementedException("Pooling is currently broken due to observation size inconsistencies. Use poolingRate = 1 for now.");
            }

            _poolingRate = poolingRate;
            _maxStepsHard = maxStepsHard / poolingRate;
            _maxStepsSoft = maxStepsSoft / poolingRate;
            _extraObservationSources = extraObservationSources ?? new List<IRLMatrixExtraObservationSource>();

            _poolingHelper = new RLMatrixPoolingHelper(_poolingRate, {{actionSizeCalc}}, _GetAllObservations);

            int baseObservationSize = _GetBaseObservationSize();
            int extraObservationSize = _extraObservationSources.Sum(source => source.GetObservationSize());
            {{stateSizeProp}} = _poolingRate * (1 + baseObservationSize + extraObservationSize);

            _rlMatrixEpisodeTerminated = true;
            _rlMatrixEpisodeTruncated = false;
            _InitializeObservations();

{{actionMethodsInit}}

            return this;
        }

        private void _InitializeObservations()
        {
            for (int i = 0; i < _poolingRate; i++)
            {
                float reward = {{rewardSum}};
                _poolingHelper.CollectObservation(reward);
            }
        }

        public Task<float[]> GetCurrentState()
        {
            if (_rlMatrixEpisodeTerminated && _IsHardDone())
            {
                Reset();
                _poolingHelper.HardReset(_GetAllObservations);
                _rlMatrixEpisodeTerminated = false;
                _rlMatrixEpisodeTruncated = false;
            }
            else if (_rlMatrixEpisodeTerminated && _IsSoftDone())
            {
                _stepsSoft = 0;
                _rlMatrixEpisodeTerminated = false;
                _rlMatrixEpisodeTruncated = false;
            }

            return Task.FromResult(_GetPooledObservationsWithTruncationFlag());
        }

        public Task Reset()
        {
            _stepsSoft = 0;
            _stepsHard = 0;
            {{model.ResetMethodName}}();
            _rlMatrixEpisodeTerminated = false;
            _rlMatrixEpisodeTruncated = false;
            _poolingHelper.HardReset(_GetAllObservations);

            if ({{model.DoneMethodName}}())
            {
                throw new Exception("Done flag still raised after reset - did you intend to reset?");
            }

            return Task.CompletedTask;
        }

{{stepMethod}}

        private bool _IsHardDone()
        {
            return (_stepsHard >= _maxStepsHard || {{model.DoneMethodName}}());
        }

        private bool _IsSoftDone()
        {
            return (_stepsSoft >= _maxStepsSoft);
        }

        public void GhostStep()
        {
            if (_IsHardDone() || _IsSoftDone())
                return;

            if (_poolingHelper.HasAction)
            {
                var actions = _poolingHelper.GetLastAction();
                int actionIndex = 0;
{{ghostStepActions}}
            }
            float reward = {{rewardSum}};
            _poolingHelper.CollectObservation(reward);
        }

        private float[] _GetAllObservations()
        {
            var baseObservations = _GetBaseObservations();
            var extraObservations = _extraObservationSources.SelectMany(source => source.GetObservations()).ToArray();
            return baseObservations.Concat(extraObservations).ToArray();
        }

        private float[] _GetPooledObservationsWithTruncationFlag()
        {
            var pooledObservations = _poolingHelper.GetPooledObservations();
            var singleObservationSize = pooledObservations.Length / _poolingRate;
            var result = new float[pooledObservations.Length + _poolingRate];

            for (int i = 0; i < _poolingRate; i++)
            {
                result[i * (singleObservationSize + 1)] = _rlMatrixEpisodeTruncated ? 1f : 0f;
                Array.Copy(pooledObservations, i * singleObservationSize, result, i * (singleObservationSize + 1) + 1, singleObservationSize);
            }

            return result;
        }

        private int _GetBaseObservationSize()
        {
            return _GetBaseObservations().Length;
        }

        private float[] _GetBaseObservations()
        {
            var observations = new List<float>();
{{observationCollection}}
            return observations.ToArray();
        }
    }
}
""";
        }

        private static string GenerateDiscreteProperties(EnvironmentModel model)
        {
            int maxSize = model.DiscreteActions.Any() ? model.DiscreteActions.Max(a => a.Size) : 0;
            string sizes = string.Join(", ", Enumerable.Repeat(maxSize, model.DiscreteActions.Length));
            return $$"""
        public OneOf<int, (int, int)> stateSize { get; set; }
        public int[] actionSize { get; set; } = new int[] { {{sizes}} };
""";
        }

        private static string GenerateContinuousProperties(EnvironmentModel model)
        {
            string discreteSizes = model.DiscreteActions.Any()
                ? string.Join(", ", Enumerable.Repeat(model.DiscreteActions.Max(a => a.Size), model.DiscreteActions.Length))
                : "";

            string bounds = string.Join(", ", model.ContinuousActions.Select(a => $"({a.Min}f, {a.Max}f)"));

            return $$"""
        public OneOf<int, (int, int)> StateSize { get; set; }
        public int[] DiscreteActionSize { get; set; } = new int[] { {{discreteSizes}} };
        public (float min, float max)[] ContinuousActionBounds { get; set; } = new (float min, float max)[] { {{bounds}} };
""";
        }

        private static string GenerateActionMethodsInitialization(EnvironmentModel model)
        {
            var sb = new StringBuilder();

            if (model.DiscreteActions.Any())
            {
                sb.AppendLine("            _actionMethodsWithCaps = new (Action<int>, int)[]");
                sb.AppendLine("            {");
                foreach (var action in model.DiscreteActions)
                {
                    sb.AppendLine($"                ({action.MethodName}, {action.Size}),");
                }
                sb.AppendLine("            };");
            }
            else
            {
                sb.AppendLine("            _actionMethodsWithCaps = new (Action<int>, int)[0];");
            }

            if (model.IsContinuous && model.ContinuousActions.Any())
            {
                sb.AppendLine();
                sb.AppendLine("            _continuousActionMethods = new Action<float>[]");
                sb.AppendLine("            {");
                foreach (var action in model.ContinuousActions)
                {
                    sb.AppendLine($"                {action.MethodName},");
                }
                sb.AppendLine("            };");
            }

            return sb.ToString();
        }

        private static string GenerateDiscreteStepMethod(EnvironmentModel model)
        {
            var rewardSum = model.RewardMethods.Any() ? string.Join(" + ", model.RewardMethods.Select(m => $"{m}()")) : "0f";

            return $$"""
        public Task<(float, bool)> Step(int[] actionsIds)
        {
            _stepsSoft++;
            _stepsHard++;

            for (int i = 0; i < _actionMethodsWithCaps.Length; i++)
            {
                int cappedAction = Math.Min(actionsIds[i], _actionMethodsWithCaps[i].maxValue - 1);
                _actionMethodsWithCaps[i].method(cappedAction);
            }
            _poolingHelper.SetAction(actionsIds.Select(a => (float)a).ToArray());

            float stepReward = {{rewardSum}};
            _poolingHelper.CollectObservation(stepReward);

            float totalReward = _poolingHelper.GetAndResetAccumulatedReward();

            bool hardDone = _IsHardDone();
            bool softDone = _IsSoftDone();
            _rlMatrixEpisodeTerminated = hardDone || softDone;
            _rlMatrixEpisodeTruncated = !hardDone && softDone;

            return Task.FromResult((totalReward, _rlMatrixEpisodeTerminated));
        }
""";
        }

        private static string GenerateContinuousStepMethod(EnvironmentModel model)
        {
            var rewardSum = model.RewardMethods.Any() ? string.Join(" + ", model.RewardMethods.Select(m => $"{m}()")) : "0f";

            return $$"""
        public Task<(float, bool)> Step(int[] discreteActions, float[] continuousActions)
        {
            _stepsSoft++;
            _stepsHard++;

            for (int i = 0; i < _actionMethodsWithCaps.Length; i++)
            {
                int cappedAction = Math.Min(discreteActions[i], _actionMethodsWithCaps[i].maxValue - 1);
                _actionMethodsWithCaps[i].method(cappedAction);
            }

            for (int i = 0; i < _continuousActionMethods.Length; i++)
            {
                _continuousActionMethods[i](continuousActions[i]);
            }

            _poolingHelper.SetAction(discreteActions.Select(a => (float)a).Concat(continuousActions).ToArray());

            float stepReward = {{rewardSum}};
            _poolingHelper.CollectObservation(stepReward);

            float totalReward = _poolingHelper.GetAndResetAccumulatedReward();

            bool hardDone = _IsHardDone();
            bool softDone = _IsSoftDone();
            _rlMatrixEpisodeTerminated = hardDone || softDone;
            _rlMatrixEpisodeTruncated = !hardDone && softDone;

            return Task.FromResult((totalReward, _rlMatrixEpisodeTerminated));
        }
""";
        }

        private static string GenerateGhostStepActions(EnvironmentModel model)
        {
            var sb = new StringBuilder();

            if (model.DiscreteActions.Any())
            {
                sb.AppendLine("                for (int i = 0; i < _actionMethodsWithCaps.Length; i++)");
                sb.AppendLine("                {");
                sb.AppendLine("                    int cappedAction = Math.Min((int)actions[actionIndex], _actionMethodsWithCaps[i].maxValue - 1);");
                sb.AppendLine("                    _actionMethodsWithCaps[i].method(cappedAction);");
                sb.AppendLine("                    actionIndex++;");
                sb.AppendLine("                }");
            }

            if (model.IsContinuous && model.ContinuousActions.Any())
            {
                sb.AppendLine("                for (int i = 0; i < _continuousActionMethods.Length; i++)");
                sb.AppendLine("                {");
                sb.AppendLine("                    _continuousActionMethods[i](actions[actionIndex]);");
                sb.AppendLine("                    actionIndex++;");
                sb.AppendLine("                }");
            }

            return sb.ToString();
        }

        private static string GenerateObservationCollection(EnvironmentModel model)
        {
            var sb = new StringBuilder();
            foreach (var obs in model.Observations)
            {
                if (obs.IsSingle)
                    sb.AppendLine($"            observations.Add({obs.MethodName}());");
                else
                    sb.AppendLine($"            observations.AddRange({obs.MethodName}());");
            }
            return sb.ToString();
        }

        public class EnvironmentModel
        {
            public string ClassName { get; }
            public string NamespaceName { get; }
            public ObservationInfo[] Observations { get; }
            public ActionInfo[] DiscreteActions { get; }
            public ContinuousActionInfo[] ContinuousActions { get; }
            public string[] RewardMethods { get; }
            public string DoneMethodName { get; }
            public string ResetMethodName { get; }
            public bool IsContinuous { get; }

            public EnvironmentModel(string className, string namespaceName, ObservationInfo[] observations,
                ActionInfo[] discreteActions, ContinuousActionInfo[] continuousActions, string[] rewardMethods,
                string doneMethodName, string resetMethodName, bool isContinuous)
            {
                ClassName = className;
                NamespaceName = namespaceName;
                Observations = observations;
                DiscreteActions = discreteActions;
                ContinuousActions = continuousActions;
                RewardMethods = rewardMethods;
                DoneMethodName = doneMethodName;
                ResetMethodName = resetMethodName;
                IsContinuous = isContinuous;
            }
        }

        public class ObservationInfo
        {
            public string MethodName { get; }
            public bool IsSingle { get; }

            public ObservationInfo(string methodName, bool isSingle)
            {
                MethodName = methodName;
                IsSingle = isSingle;
            }
        }

        public class ActionInfo
        {
            public string MethodName { get; }
            public int Size { get; }

            public ActionInfo(string methodName, int size)
            {
                MethodName = methodName;
                Size = size;
            }
        }

        public class ContinuousActionInfo
        {
            public string MethodName { get; }
            public float Min { get; }
            public float Max { get; }

            public ContinuousActionInfo(string methodName, float min, float max)
            {
                MethodName = methodName;
                Min = min;
                Max = max;
            }
        }
    }
    }