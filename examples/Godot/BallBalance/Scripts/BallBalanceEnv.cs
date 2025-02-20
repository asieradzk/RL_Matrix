using Godot;
using System;
using System.Threading.Tasks;
using RLMatrix;
using OneOf;

public partial class BallBalanceEnv : Node3D, IContinuousEnvironmentAsync<float[]>
{
    [Export] public RigidBody3D Ball { get; set; }
    [Export] public RigidBody3D Head { get; set; }
    [Export] public float HeadRadius { get; set; } = 5f;

    private int poolingRate = 1;
    private RLMatrixPoolingHelper poolingHelper;
    private int stepsSoft = 0;
    private int stepsHard = 0;

    private int _maxStepsHard = 5000;
    private int maxStepsHard => _maxStepsHard / poolingRate;

    private int _maxStepsSoft = 1000;
    private int maxStepsSoft => _maxStepsSoft / poolingRate;

    public OneOf<int, (int, int)> StateSize { get; set; }
    public int[] DiscreteActionSize { get; set; } = Array.Empty<int>();
    public (float min, float max)[] ContinuousActionBounds { get; set; } = new[]
    {
        (-1f, 1f),
        (-1f, 1f),
    };

    private bool isDone;
    private const float MaxAngularVelocity = 3f;

    public void Initialize(int poolingRate = 1)
    {
        this.poolingRate = poolingRate;
        poolingHelper = new RLMatrixPoolingHelper(poolingRate, ContinuousActionBounds.Length, GetObservations);
        StateSize = poolingRate * 8;
        isDone = true;
        InitializeObservations();
    }

    private void InitializeObservations()
    {
        for (var i = 0; i < poolingRate; i++)
        {
            var reward = CalculateReward();
            poolingHelper.CollectObservation(reward);
        }
    }

    public Task<float[]> GetCurrentState()
    {
        if (isDone && IsHardDone())
        {
            Reset();
            poolingHelper.HardReset(GetObservations);
            isDone = false;
        }
        else if (isDone && IsSoftDone())
        {
            stepsSoft = 0;
            isDone = false;
        }

        return Task.FromResult(poolingHelper.GetPooledObservations());
    }

    public Task Reset()
    {
        stepsSoft = 0;
        stepsHard = 0;
        ResetEnvironment();
        isDone = false;
        poolingHelper.HardReset(GetObservations);

        if (IsDone())
        {
            throw new Exception("Done flag still raised after reset - did you intend to reset?");
        }

        return Task.CompletedTask;
    }

    public Task<(float reward, bool done)> Step(int[] discreteActions, float[] continuousActions)
    {
        stepsSoft++;
        stepsHard++;

        ApplyActions(continuousActions);

        var stepReward = CalculateReward();
        poolingHelper.CollectObservation(stepReward);

        var totalReward = poolingHelper.GetAndResetAccumulatedReward();
        isDone = IsHardDone() || IsSoftDone();

        poolingHelper.SetAction(continuousActions);

        return Task.FromResult((totalReward, isDone));
    }

    private bool IsHardDone() => stepsHard >= maxStepsHard || IsDone();

    private bool IsSoftDone() => stepsSoft >= maxStepsSoft;

    public void GhostStep()
    {
        if (IsHardDone() || IsSoftDone())
            return;

        if (poolingHelper.HasAction)
        {
            ApplyActions(poolingHelper.GetLastAction());
        }
        var reward = CalculateReward();
        poolingHelper.CollectObservation(reward);
    }

    private float[] GetObservations()
    {
        return new float[]
        {
            Head.Rotation.X,
            Head.Rotation.Z,
            BallOffsetObservation().X / HeadRadius,
            BallOffsetObservation().Y / HeadRadius,
            BallOffsetObservation().Z / HeadRadius,
            Ball.LinearVelocity.X / 10f,
            Ball.LinearVelocity.Y / 10f,
            Ball.LinearVelocity.Z / 10f
        };
    }

    float modifier = 0.2f;
    private void ApplyActions(float[] actions)
    {
        var angularVelocity = new Vector3(
            actions[0] * modifier* MaxAngularVelocity,
            0,
            actions[1] * modifier* MaxAngularVelocity
        );
        Head.AngularVelocity = angularVelocity;
    }

    private Vector3 BallOffsetObservation() => Ball.GlobalPosition - Head.GlobalPosition;

    private float CalculateReward()
    {
        var ballOffset = BallOffsetObservation();
        if (ballOffset.Y < -2f || Mathf.Abs(ballOffset.X) > HeadRadius || Mathf.Abs(ballOffset.Z) > HeadRadius)
        {
            return -1f;
        }
        return 1f;
    }

    private bool IsDone()
    {
        var ballOffset = BallOffsetObservation();
        return ballOffset.Y < -2f || Mathf.Abs(ballOffset.X) > HeadRadius || Mathf.Abs(ballOffset.Z) > HeadRadius;
    }

    private void ResetEnvironment()
    {
        Head.Rotation = new Vector3(
            Mathf.DegToRad((float)GD.RandRange(-10.0, 10.0)),
            0,
            Mathf.DegToRad((float)GD.RandRange(-10.0, 10.0))
        );
        Head.AngularVelocity = Vector3.Zero;
        Ball.LinearVelocity = Vector3.Zero;
        Ball.AngularVelocity = Vector3.Zero;
    
        var spawnRadius = HeadRadius * 0.15f;
        var randomOffset = new Vector3(
            (float)GD.RandRange(-spawnRadius, spawnRadius),
            HeadRadius * 0.3f,
            (float)GD.RandRange(-spawnRadius, spawnRadius)
        );
        Ball.GlobalPosition = Head.GlobalPosition + randomOffset;
    }
}