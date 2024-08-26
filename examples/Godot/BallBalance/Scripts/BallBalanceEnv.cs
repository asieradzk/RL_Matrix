using Godot;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using RLMatrix;
using OneOf;

public partial class BallBalanceEnv : Node3D, IContinuousEnvironmentAsync<float[]>
{
    [Export] public RigidBody3D Ball { get; set; }
    [Export] public Node3D Head { get; set; }
    [Export] public float HeadRadius { get; set; } = 5f;

    private int poolingRate = 1;
    private RLMatrixPoolingHelper poolingHelper;
    private int stepsSoft = 0;
    private int stepsHard = 0;

    int _maxStepsHard = 5000;
    private int maxStepsHard
    {
        get => _maxStepsHard / poolingRate;
        set => _maxStepsHard = value;
    }

    int _maxStepsSoft = 1000;
    private int maxStepsSoft
    {
        get => _maxStepsSoft / poolingRate;
        set => _maxStepsSoft = value;
    }

    public OneOf<int, (int, int)> StateSize { get; set; }
    public int[] DiscreteActionSize { get; set; } = Array.Empty<int>();
    public (float min, float max)[] ContinuousActionBounds { get; set; } = new (float min, float max)[]
    {
        (-1f, 1f),
        (-1f, 1f),
    };

    private bool isDone;

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
        for (int i = 0; i < poolingRate; i++)
        {
            float reward = Reward();
            poolingHelper.CollectObservation(reward);
        }
    }

    public Task<float[]> GetCurrentState()
    {
        if (isDone && AmIHardDone())
        {
            Reset();
            poolingHelper.HardReset(GetObservations);
            isDone = false;
        }
        else if (isDone && AmISoftDone())
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
        ResetMe();
        isDone = false;
        poolingHelper.HardReset(GetObservations);

        if (IsDoneCheck())
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

        float stepReward = Reward();
        poolingHelper.CollectObservation(stepReward);

        float totalReward = poolingHelper.GetAndResetAccumulatedReward();
        isDone = AmIHardDone() || AmISoftDone();

        poolingHelper.SetAction(continuousActions);

        return Task.FromResult((totalReward, isDone));
    }

    private bool AmIHardDone()
    {
        return (stepsHard >= maxStepsHard || IsDoneCheck());
    }

    private bool AmISoftDone()
    {
        return (stepsSoft >= maxStepsSoft);
    }

    public void GhostStep()
    {
        if(AmIHardDone() || AmISoftDone())
            return;

        if (poolingHelper.HasAction)
        {
            ApplyActions(poolingHelper.GetLastAction());
        }
        float reward = Reward();
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
            BallVelocityObservation().X,
            BallVelocityObservation().Y,
            BallVelocityObservation().Z
        };
    }

    private void ApplyActions(float[] actions)
    {
        HeadRotationActionXAxis(actions[0]);
        HeadRotationActionZAxis(actions[1]);
    }

    public void HeadRotationActionZAxis(float rotation)
    {
        if ((Head.Rotation.Z < 0.25f && rotation > 0f) ||
            (Head.Rotation.Z > -0.25f && rotation < 0f))
        {
            Head.RotateZ(2f * rotation * (float)GetProcessDeltaTime());
        }
    }

    public void HeadRotationActionXAxis(float rotation)
    {
        if ((Head.Rotation.X < 0.25f && rotation > 0f) ||
            (Head.Rotation.X > -0.25f && rotation < 0f))
        {
            Head.RotateX(2f * rotation * (float)GetProcessDeltaTime());
        }
    }

    public Vector3 BallOffsetObservation()
    {
        return Ball.GlobalPosition - Head.GlobalPosition;
    }

    public Vector3 BallVelocityObservation()
    {
        return Ball.LinearVelocity;
    }

    public float Reward()
    {
        Vector3 ballOffset = BallOffsetObservation();
        if (ballOffset.Y < -2f || Mathf.Abs(ballOffset.X) > HeadRadius || Mathf.Abs(ballOffset.Z) > HeadRadius)
        {
            return -1f;
        }
        else
        {
            return 0.1f;
        }
    }

    public bool IsDoneCheck()
    {
        Vector3 ballOffset = BallOffsetObservation();
        return ballOffset.Y < -2f || Mathf.Abs(ballOffset.X) > HeadRadius || Mathf.Abs(ballOffset.Z) > HeadRadius;
    }

    public void ResetMe()
    {
        Head.Rotation = new Vector3(
            Mathf.DegToRad((float)GD.RandRange(-10.0, 10.0)),
            0,
            Mathf.DegToRad((float)GD.RandRange(-10.0, 10.0))
        );
        Ball.LinearVelocity = Vector3.Zero;
        Ball.AngularVelocity = Vector3.Zero;
        
        float spawnRadius = HeadRadius * 0.5f;
        Vector3 randomOffset = new Vector3(
            (float)GD.RandRange(-spawnRadius, spawnRadius),
            HeadRadius * 0.8f,
            (float)GD.RandRange(-spawnRadius, spawnRadius)
        );
        Ball.GlobalPosition = Head.GlobalPosition + randomOffset;
    }
}