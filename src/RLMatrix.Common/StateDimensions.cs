namespace RLMatrix.Common;

public sealed record StateDimensions
{
    private StateDimensions(params int[] dimensions)
    {
        if (dimensions.Length == 0)
            throw new InvalidOperationException("State must have at least one dimension.");
        
        Dimensions = dimensions;
    }
    
    public int[] Dimensions { get; }

    public static StateDimensions Create1D(int dimension) => dimension;
    public static StateDimensions Create2D(int dimension1, int dimension2) => (dimension1, dimension2);
    
    public static implicit operator StateDimensions(int dimension1) => new(dimension1);
    public static implicit operator StateDimensions((int Dimension1, int Dimension2) tuple) => new(tuple.Dimension1, tuple.Dimension2);
}