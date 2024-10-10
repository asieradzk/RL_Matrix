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
            var candidateClasses = context.SyntaxProvider
                .CreateSyntaxProvider(
                    predicate: static (s, _) => s is ClassDeclarationSyntax cds &&
                        cds.AttributeLists.Any(al => al.Attributes.Any(a => a.Name.ToString() == "RLMatrixEnvironment")),
                    transform: static (ctx, _) => (ClassDeclarationSyntax)ctx.Node)
                .Where(static m => m is not null);

            IncrementalValueProvider<(Compilation, ImmutableArray<ClassDeclarationSyntax>)> compilationAndClasses
                = context.CompilationProvider.Combine(candidateClasses.Collect());

            context.RegisterSourceOutput(compilationAndClasses,
                (spc, source) => Execute(source.Item1, source.Item2, spc));

            context.RegisterPostInitializationOutput(ctx =>
            {
                ctx.AddSource("RLMatrixPoolingHelper.g.cs", SourceText.From(AdditionalFiles.RLMatrixPoolingHelper, Encoding.UTF8));
                ctx.AddSource("IRLMatrixExtraObservationSource.g.cs", SourceText.From(AdditionalFiles.IRLMatrixExtraObservationSource, Encoding.UTF8));
                ctx.AddSource("RLMatrixAttributes.g.cs", SourceText.From(AdditionalFiles.RLMatrixAttributes, Encoding.UTF8));
            });
        }

        private void Execute(Compilation compilation, ImmutableArray<ClassDeclarationSyntax> classes, SourceProductionContext context)
        {
            foreach (var classDeclaration in classes)
            {
                var semanticModel = compilation.GetSemanticModel(classDeclaration.SyntaxTree);
                var classSymbol = semanticModel.GetDeclaredSymbol(classDeclaration) as INamedTypeSymbol;
                if (classSymbol == null) continue;

                EnvironmentInfo environmentInfo;
                if (HasContinuousActions(classSymbol))
                {
                    environmentInfo = new ContinuousEnvironmentInfo(classSymbol);
                }
                else
                {
                    environmentInfo = new DiscreteEnvironmentInfo(classSymbol);
                }

                if (!ValidateEnvironment(environmentInfo, context, classDeclaration))
                    continue;

                var generatedCode = GeneratePartialClass(environmentInfo);
                context.AddSource($"{classSymbol.Name}.g.cs", SourceText.From(generatedCode, Encoding.UTF8));
            }
        }

        private bool HasContinuousActions(INamedTypeSymbol classSymbol)
        {
            return classSymbol.GetMembers()
                .OfType<IMethodSymbol>()
                .Any(m => m.GetAttributes()
                    .Any(a => a.AttributeClass.Name == nameof(RLMatrixActionContinuousAttribute)));
        }

        private bool ValidateEnvironment(EnvironmentInfo info, SourceProductionContext context, ClassDeclarationSyntax classDeclaration)
        {
            if (info.ResetMethod == null)
            {
                ReportDiagnostic(context, classDeclaration, "RLMatrix001", "No Reset method found. Please add a method with [RLMatrixReset] attribute.");
                return false;
            }
            if (info.DoneMethod == null)
            {
                ReportDiagnostic(context, classDeclaration, "RLMatrix002", "No Done method found. Please add a method with [RLMatrixDone] attribute.");
                return false;
            }
            return true;
        }

        private void ReportDiagnostic(SourceProductionContext context, ClassDeclarationSyntax classDeclaration, string id, string message)
        {
            var diagnostic = Diagnostic.Create(new DiagnosticDescriptor(id, "RLMatrix Environment Error", message, "RLMatrix", DiagnosticSeverity.Error, isEnabledByDefault: true), classDeclaration.GetLocation());
            context.ReportDiagnostic(diagnostic);
        }

        private string GeneratePartialClass(EnvironmentInfo info)
        {
            var sb = new StringBuilder();

            sb.AppendLine("using System;");
            sb.AppendLine("using System.Collections.Generic;");
            sb.AppendLine("using System.Linq;");
            sb.AppendLine("using System.Threading.Tasks;");
            sb.AppendLine("using OneOf;");
            sb.AppendLine("using RLMatrix;");
            sb.AppendLine("using RLMatrix.Toolkit;");
            sb.AppendLine();

            sb.AppendLine($"namespace {info.EnvironmentType.ContainingNamespace}");
            sb.AppendLine("{");

            string interfaceName = info is ContinuousEnvironmentInfo ? "IContinuousEnvironmentAsync<float[]>" : "IEnvironmentAsync<float[]>";
            sb.AppendLine($"    public partial class {info.EnvironmentType.Name} : {interfaceName}");
            sb.AppendLine("    {");

            GenerateFields(sb);
            GenerateProperties(sb, info);
            GenerateRLInitMethod(sb, info);
            GenerateInitializeObservationsMethod(sb, info);
            GenerateGetCurrentStateMethod(sb);
            GenerateResetMethod(sb, info);
            GenerateStepMethod(sb, info);
            GenerateIsHardDoneMethod(sb, info);
            GenerateIsSoftDoneMethod(sb);
            GenerateGhostStepMethod(sb, info);
            GenerateGetAllObservationsMethod(sb, info);
            GenerateGetBaseObservationSizeMethod(sb, info);
            GenerateGetBaseObservationsMethod(sb, info);

            sb.AppendLine("    }");
            sb.AppendLine("}");

            return sb.ToString();
        }


        private void GenerateFields(StringBuilder sb)
        {
            sb.AppendLine("        private int _poolingRate;");
            sb.AppendLine("        private RLMatrixPoolingHelper _poolingHelper;");
            sb.AppendLine("        private List<IRLMatrixExtraObservationSource> _extraObservationSources;");
            sb.AppendLine("        private int _stepsSoft;");
            sb.AppendLine("        private int _stepsHard;");
            sb.AppendLine("        private int _maxStepsHard;");
            sb.AppendLine("        private int _maxStepsSoft;");
            sb.AppendLine("        private bool _rlMatrixEpisodeTerminated;");
            sb.AppendLine("        private (Action<int> method, int maxValue)[] _actionMethodsWithCaps;");
            sb.AppendLine("        private Action<float>[] _continuousActionMethods;");
        }


        private void GenerateProperties(StringBuilder sb, EnvironmentInfo info)
        {
            if (info is ContinuousEnvironmentInfo continuousInfo)
            {
                sb.AppendLine("        public OneOf<int, (int, int)> StateSize { get; set; }");

                if (continuousInfo.DiscreteDimensions.Length > 0)
                {
                    int maxDiscreteActionSize = continuousInfo.DiscreteDimensions.Max();
                    string discreteActionSizes = string.Join(", ", Enumerable.Repeat(maxDiscreteActionSize, continuousInfo.DiscreteDimensions.Length));
                    sb.AppendLine($"        public int[] DiscreteActionSize {{ get; set; }} = new int[] {{ {discreteActionSizes} }};");
                }
                else
                {
                    sb.AppendLine("        public int[] DiscreteActionSize { get; set; } = new int[0];");
                }

                sb.AppendLine($"        public (float min, float max)[] ContinuousActionBounds {{ get; set; }} = new (float min, float max)[] {{ {string.Join(", ", continuousInfo.ContinuousActionBounds.Select(b => $"({b.min}f, {b.max}f)"))} }};");
            }
            else
            {
                sb.AppendLine("        public OneOf<int, (int, int)> stateSize { get; set; }");

                int maxDiscreteActionSize = info.DiscreteActionMethods.Length > 0
                    ? info.DiscreteActionMethods
                        .Max(m => (int)m.GetAttributes()
                            .First(a => a.AttributeClass.Name == nameof(RLMatrixActionDiscreteAttribute))
                            .ConstructorArguments[0].Value)
                    : 0;

                int discreteActionCount = info.DiscreteActionMethods.Length;
                string discreteActionSizes = string.Join(", ", Enumerable.Repeat(maxDiscreteActionSize, discreteActionCount));

                sb.AppendLine($"        public int[] actionSize {{ get; set; }} = new int[] {{ {discreteActionSizes} }};");
            }
        }

        private void GenerateRLInitMethod(StringBuilder sb, EnvironmentInfo info)
        {
            string returnType = info.EnvironmentType.Name;
            sb.AppendLine($"        public virtual {returnType} RLInit(int poolingRate = 1, int maxStepsHard = 1000, int maxStepsSoft = 100, List<IRLMatrixExtraObservationSource> extraObservationSources = null)");
            sb.AppendLine("        {");
            sb.AppendLine("            _poolingRate = poolingRate;");
            sb.AppendLine("            _maxStepsHard = maxStepsHard / poolingRate;");
            sb.AppendLine("            _maxStepsSoft = maxStepsSoft / poolingRate;");
            sb.AppendLine("            _extraObservationSources = extraObservationSources ?? new List<IRLMatrixExtraObservationSource>();");
            sb.AppendLine();

            if (info is ContinuousEnvironmentInfo continuousInfo)
            {
                sb.AppendLine($"            _poolingHelper = new RLMatrixPoolingHelper(_poolingRate, DiscreteActionSize.Length + ContinuousActionBounds.Length, _GetAllObservations);");
                sb.AppendLine();
                sb.AppendLine("            int baseObservationSize = _GetBaseObservationSize();");
                sb.AppendLine("            int extraObservationSize = _extraObservationSources.Sum(source => source.GetObservationSize());");
                sb.AppendLine("            StateSize = _poolingRate * (baseObservationSize + extraObservationSize);");
                sb.AppendLine();
                sb.AppendLine("            _rlMatrixEpisodeTerminated = true;");
                sb.AppendLine("            _InitializeObservations();");
                sb.AppendLine();

                if (continuousInfo.DiscreteDimensions.Length > 0)
                {
                    sb.AppendLine("            _actionMethodsWithCaps = new (Action<int>, int)[]");
                    sb.AppendLine("            {");
                    foreach (var method in info.DiscreteActionMethods)
                    {
                        var attr = method.GetAttributes().First(a => a.AttributeClass.Name == nameof(RLMatrixActionDiscreteAttribute));
                        var maxValue = attr.ConstructorArguments[0].Value;
                        sb.AppendLine($"                ({method.Name}, {maxValue}),");
                    }
                    sb.AppendLine("            };");
                }
                else
                {
                    sb.AppendLine("            _actionMethodsWithCaps = new (Action<int>, int)[0];");
                }

                sb.AppendLine();
                sb.AppendLine("            _continuousActionMethods = new Action<float>[]");
                sb.AppendLine("            {");
                foreach (var method in continuousInfo.ContinuousActionMethods)
                {
                    sb.AppendLine($"                {method.Name},");
                }
                sb.AppendLine("            };");
            }
            else
            {
                sb.AppendLine($"            _poolingHelper = new RLMatrixPoolingHelper(_poolingRate, actionSize.Length, _GetAllObservations);");
                sb.AppendLine();
                sb.AppendLine("            int baseObservationSize = _GetBaseObservationSize();");
                sb.AppendLine("            int extraObservationSize = _extraObservationSources.Sum(source => source.GetObservationSize());");
                sb.AppendLine("            stateSize = _poolingRate * (baseObservationSize + extraObservationSize);");
                sb.AppendLine();
                sb.AppendLine("            _rlMatrixEpisodeTerminated = true;");
                sb.AppendLine("            _InitializeObservations();");
                sb.AppendLine();

                sb.AppendLine("            _actionMethodsWithCaps = new (Action<int>, int)[]");
                sb.AppendLine("            {");
                foreach (var method in info.DiscreteActionMethods)
                {
                    var attr = method.GetAttributes().First(a => a.AttributeClass.Name == nameof(RLMatrixActionDiscreteAttribute));
                    var maxValue = attr.ConstructorArguments[0].Value;
                    sb.AppendLine($"                ({method.Name}, {maxValue}),");
                }
                sb.AppendLine("            };");
            }

            sb.AppendLine("            return this;");
            sb.AppendLine("        }");
        }


        private void GenerateInitializeObservationsMethod(StringBuilder sb, EnvironmentInfo info)
        {
            sb.AppendLine("        private void _InitializeObservations()");
            sb.AppendLine("        {");
            sb.AppendLine("            for (int i = 0; i < _poolingRate; i++)");
            sb.AppendLine("            {");
            sb.AppendLine($"                float reward = {string.Join(" + ", info.RewardMethods.Select(m => $"{m.Name}()"))};");
            sb.AppendLine("                _poolingHelper.CollectObservation(reward);");
            sb.AppendLine("            }");
            sb.AppendLine("        }");
        }

        private void GenerateGetCurrentStateMethod(StringBuilder sb)
        {
            sb.AppendLine("        public Task<float[]> GetCurrentState()");
            sb.AppendLine("        {");
            sb.AppendLine("            if (_rlMatrixEpisodeTerminated && _IsHardDone())");
            sb.AppendLine("            {");
            sb.AppendLine("                Reset();");
            sb.AppendLine("                _poolingHelper.HardReset(_GetAllObservations);");
            sb.AppendLine("                _rlMatrixEpisodeTerminated = false;");
            sb.AppendLine("            }");
            sb.AppendLine("            else if (_rlMatrixEpisodeTerminated && _IsSoftDone())");
            sb.AppendLine("            {");
            sb.AppendLine("                _stepsSoft = 0;");
            sb.AppendLine("                _rlMatrixEpisodeTerminated = false;");
            sb.AppendLine("            }");
            sb.AppendLine();
            sb.AppendLine("            return Task.FromResult(_poolingHelper.GetPooledObservations());");
            sb.AppendLine("        }");
        }

        private void GenerateResetMethod(StringBuilder sb, EnvironmentInfo info)
        {
            sb.AppendLine("        public Task Reset()");
            sb.AppendLine("        {");
            sb.AppendLine("            _stepsSoft = 0;");
            sb.AppendLine("            _stepsHard = 0;");
            sb.AppendLine($"            {info.ResetMethod.Name}();");
            sb.AppendLine("            _rlMatrixEpisodeTerminated = false;");
            sb.AppendLine("            _poolingHelper.HardReset(_GetAllObservations);");
            sb.AppendLine();
            sb.AppendLine($"            if ({info.DoneMethod.Name}())");
            sb.AppendLine("            {");
            sb.AppendLine("                throw new Exception(\"Done flag still raised after reset - did you intend to reset?\");");
            sb.AppendLine("            }");
            sb.AppendLine();
            sb.AppendLine("            return Task.CompletedTask;");
            sb.AppendLine("        }");
        }

        private void GenerateStepMethod(StringBuilder sb, EnvironmentInfo info)
        {
            string methodSignature = info is ContinuousEnvironmentInfo
                ? "public Task<(float, bool)> Step(int[] discreteActions, float[] continuousActions)"
                : "public Task<(float, bool)> Step(int[] actionsIds)";

            sb.AppendLine($"        {methodSignature}");
            sb.AppendLine("        {");
            sb.AppendLine("            _stepsSoft++;");
            sb.AppendLine("            _stepsHard++;");
            sb.AppendLine();

            if (info is ContinuousEnvironmentInfo continuousInfo)
            {
                if (continuousInfo.DiscreteDimensions.Length > 0)
                {
                    sb.AppendLine("            for (int i = 0; i < _actionMethodsWithCaps.Length; i++)");
                    sb.AppendLine("            {");
                    sb.AppendLine("                int cappedAction = Math.Min(discreteActions[i], _actionMethodsWithCaps[i].maxValue - 1);");
                    sb.AppendLine("                _actionMethodsWithCaps[i].method(cappedAction);");
                    sb.AppendLine("            }");
                    sb.AppendLine();
                }

                sb.AppendLine("            for (int i = 0; i < _continuousActionMethods.Length; i++)");
                sb.AppendLine("            {");
                sb.AppendLine("                _continuousActionMethods[i](continuousActions[i]);");
                sb.AppendLine("            }");

                if (continuousInfo.DiscreteDimensions.Length > 0)
                {
                    sb.AppendLine("            _poolingHelper.SetAction(discreteActions.Select(a => (float)a).Concat(continuousActions).ToArray());");
                }
                else
                {
                    sb.AppendLine("            _poolingHelper.SetAction(continuousActions);");
                }
            }
            else
            {
                sb.AppendLine("            for (int i = 0; i < _actionMethodsWithCaps.Length; i++)");
                sb.AppendLine("            {");
                sb.AppendLine("                int cappedAction = Math.Min(actionsIds[i], _actionMethodsWithCaps[i].maxValue - 1);");
                sb.AppendLine("                _actionMethodsWithCaps[i].method(cappedAction);");
                sb.AppendLine("            }");
                sb.AppendLine("            _poolingHelper.SetAction(actionsIds.Select(a => (float)a).ToArray());");
            }

            sb.AppendLine();
            sb.AppendLine($"            float stepReward = {string.Join(" + ", info.RewardMethods.Select(m => $"{m.Name}()"))};");
            sb.AppendLine("            _poolingHelper.CollectObservation(stepReward);");
            sb.AppendLine();
            sb.AppendLine("            float totalReward = _poolingHelper.GetAndResetAccumulatedReward();");
            sb.AppendLine();
            sb.AppendLine("            _rlMatrixEpisodeTerminated = _IsHardDone() || _IsSoftDone();");
            sb.AppendLine();
            sb.AppendLine("            return Task.FromResult((totalReward, _rlMatrixEpisodeTerminated));");
            sb.AppendLine("        }");
        }


        private void GenerateIsHardDoneMethod(StringBuilder sb, EnvironmentInfo info)
        {
            sb.AppendLine("        private bool _IsHardDone()");
            sb.AppendLine("        {");
            sb.AppendLine($"            return (_stepsHard >= _maxStepsHard || {info.DoneMethod.Name}());");
            sb.AppendLine("        }");
        }

        private void GenerateIsSoftDoneMethod(StringBuilder sb)
        {
            sb.AppendLine("        private bool _IsSoftDone()");
            sb.AppendLine("        {");
            sb.AppendLine("            return (_stepsSoft >= _maxStepsSoft);");
            sb.AppendLine("        }");
        }

        private void GenerateGhostStepMethod(StringBuilder sb, EnvironmentInfo info)
        {
            sb.AppendLine("        public void GhostStep()");
            sb.AppendLine("        {");
            sb.AppendLine("            if (_IsHardDone() || _IsSoftDone())");
            sb.AppendLine("                return;");
            sb.AppendLine();
            sb.AppendLine("            if (_poolingHelper.HasAction)");
            sb.AppendLine("            {");
            sb.AppendLine("                var actions = _poolingHelper.GetLastAction();");
            sb.AppendLine("                int actionIndex = 0;");

            if (info is ContinuousEnvironmentInfo continuousInfo)
            {
                if (continuousInfo.DiscreteDimensions.Length > 0)
                {
                    sb.AppendLine("                for (int i = 0; i < _actionMethodsWithCaps.Length; i++)");
                    sb.AppendLine("                {");
                    sb.AppendLine("                    int cappedAction = Math.Min((int)actions[actionIndex], _actionMethodsWithCaps[i].maxValue - 1);");
                    sb.AppendLine("                    _actionMethodsWithCaps[i].method(cappedAction);");
                    sb.AppendLine("                    actionIndex++;");
                    sb.AppendLine("                }");
                }

                sb.AppendLine("                for (int i = 0; i < _continuousActionMethods.Length; i++)");
                sb.AppendLine("                {");
                sb.AppendLine("                    _continuousActionMethods[i](actions[actionIndex]);");
                sb.AppendLine("                    actionIndex++;");
                sb.AppendLine("                }");
            }
            else
            {
                sb.AppendLine("                for (int i = 0; i < _actionMethodsWithCaps.Length; i++)");
                sb.AppendLine("                {");
                sb.AppendLine("                    int cappedAction = Math.Min((int)actions[i], _actionMethodsWithCaps[i].maxValue - 1);");
                sb.AppendLine("                    _actionMethodsWithCaps[i].method(cappedAction);");
                sb.AppendLine("                }");
            }

            sb.AppendLine("            }");
            sb.AppendLine($"            float reward = {string.Join(" + ", info.RewardMethods.Select(m => $"{m.Name}()"))};");
            sb.AppendLine("            _poolingHelper.CollectObservation(reward);");
            sb.AppendLine("        }");
        }
        private void GenerateGetAllObservationsMethod(StringBuilder sb, EnvironmentInfo info)
        {
            sb.AppendLine("        private float[] _GetAllObservations()");
            sb.AppendLine("        {");
            sb.AppendLine("            var baseObservations = _GetBaseObservations();");
            sb.AppendLine("            var extraObservations = _extraObservationSources.SelectMany(source => source.GetObservations()).ToArray();");
            sb.AppendLine("            return baseObservations.Concat(extraObservations).ToArray();");
            sb.AppendLine("        }");
        }

        private void GenerateGetBaseObservationsMethod(StringBuilder sb, EnvironmentInfo info)
        {
            sb.AppendLine("        private float[] _GetBaseObservations()");
            sb.AppendLine("        {");
            sb.AppendLine("            var observations = new List<float>();");

            foreach (var method in info.ObservationMethods)
            {
                if (method.ReturnType.SpecialType == SpecialType.System_Single)
                {
                    sb.AppendLine($"            observations.Add({method.Name}());");
                }
                else if (method.ReturnType is IArrayTypeSymbol arrayType &&
                         arrayType.ElementType.SpecialType == SpecialType.System_Single)
                {
                    sb.AppendLine($"            observations.AddRange({method.Name}());");
                }
            }

            sb.AppendLine("            return observations.ToArray();");
            sb.AppendLine("        }");
        }
        private void GenerateGetBaseObservationSizeMethod(StringBuilder sb, EnvironmentInfo info)
        {
            sb.AppendLine("        private int _GetBaseObservationSize()");
            sb.AppendLine("        {");
            sb.AppendLine("            return _GetBaseObservations().Length;");
            sb.AppendLine("        }");
        }
    }

    public class RLMatrixSyntaxReceiver : ISyntaxContextReceiver
    {
        public List<ClassDeclarationSyntax> CandidateClasses { get; } = new List<ClassDeclarationSyntax>();

        public void OnVisitSyntaxNode(GeneratorSyntaxContext context)
        {
            if (context.Node is ClassDeclarationSyntax classDeclaration &&
                classDeclaration.AttributeLists.Any(al => al.Attributes.Any(a => a.Name.ToString() == "RLMatrixEnvironment")))
            {
                CandidateClasses.Add(classDeclaration);
            }
        }
    }
}