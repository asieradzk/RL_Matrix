using Godot;
using System;
using System.Collections.Generic;
using RLMatrix.Godot;


//TState is the shape of input vector, alternative is float[,] for 2d inputs like images.
//If you use float[,] then RLMatrix uses CNN as default.
public partial class BallBalanceEnv : GodotEnvironmentDiscrete<float[]>
{
	//--------------------------------------v---------These are are required properties from the base class
	
	public override List<List<Action>> myHeads { get; set; }
	public override Action resetProvider => Reset;
	public override Func<float[]> observationProvider => Observations;
	public override Func<bool> isDoneProvider => IsDone;
	public override int maxSteps { get; set; }
	
	//_________________________________________________^___ These are are required properties from the base class
	
	
	
	
	//--------------------------------v user code v--------------------------------
    
	//User needs to specify RBs that we will be watching
	[Export] public RigidBody3D head; //platform on which ball is balanced
	[Export] public RigidBody3D ball;
	
	//lets create a reset method that moves ball above balancing platform & reset platform's rotation
	//also need to reset forces on the ball
	void Reset()
	{
		//local scale by default (nice)
		//create some offset in the x between -0.5 and 0.5
		var xOffset = (float)GD.RandRange(-0.5f, 0.5f);
		ball.Position = new Vector3(xOffset, 3, 0);
		ball.LinearVelocity = new Vector3(0,0,0);
		ball.AngularVelocity = new Vector3(0,0,0);
		//reset spinning platform (head)
		head.Rotation = new Vector3(0,0,0);
		head.AngularVelocity = new Vector3(0,0,0);
		head.Position = new Vector3(0,0,0);
	}
	
	//Create a method for finding if ball fell off te platform (you can also implement all this directly in properties above)
	bool IsDone()
	{
		return ball.Position.Y < -2f;
	}

	
	//RL agent will figure out the size of inputs for neural network based on this
	//It is up to user to determine how to format and fold observations into a single array
	//its also possible to use 2D float float[,] for observations and RLMatrix will use CNN as first layer for this by default
	float[] Observations()
	{
		//lets observe head rotation in the z (1) axis as well as ball x and y position (2) and linear velocity (2) and angular velocity (3)
		//this is a sum of 9 floats
		
		//if you dont feel like counting you can do this roundabout way:
		List<float> myObservations = new();
		myObservations.Add(head.Rotation.Z/10f);
		myObservations.Add(ball.Position.X);
		myObservations.Add(ball.Position.Y);
		myObservations.Add(ball.LinearVelocity.X);
		myObservations.Add(ball.LinearVelocity.Y);

		return myObservations.ToArray();
	}
	
	
	//Finally we need to make some actions (outputs from neural network are mapped to these)
	//This is also a good place to add rewards
	void RotateLeft()
	{
		head.AngularVelocity = new Vector3(0,0,3);
		AddReward(0.2f);
	}
	void RotateRight()
	{
		head.AngularVelocity = new Vector3(0,0,-3);
		AddReward(0.2f);
	}


	//For clarity I assign these methods here but they can be also written straight into properties above
	public override void _Ready()
	{
		
		myHeads = new List<List<Action>> { new List<Action> { RotateLeft, RotateRight } };
		maxSteps = 10000;
		
		
		//the base Ready must be called after heads exist!
		base._Ready();
	}
	//Now we loop back up and assign these to properties
	//In the future after dev is more stable it should be done with attributes
}
