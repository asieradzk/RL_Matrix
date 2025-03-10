namespace RLMatrix;

public abstract class PPOCriticNet(string name) : TensorModule(name)
{
    public abstract override Tensor forward(Tensor x);

    public abstract Tensor forward(PackedSequence x); 
}