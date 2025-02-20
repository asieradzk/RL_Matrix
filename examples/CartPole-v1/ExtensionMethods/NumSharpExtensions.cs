using NumSharp;

public static class NumSharpExtensions
{
    public static float[] ToFloatArray(this NDArray npArray)
    {
        var doubleArray = npArray.ToArray<double>();
        return Array.ConvertAll(doubleArray, item => (float)item);
    }
    public static float[,] ToFloat2DArray(this NDArray ndArray)
    {
        var multidimArray = ndArray.ToMuliDimArray<float>();
        var floatArray = new float[multidimArray.GetLength(0), multidimArray.GetLength(1)];

        for (var i = 0; i < multidimArray.GetLength(0); i++)
        {
            for (var j = 0; j < multidimArray.GetLength(1); j++)
            {
                floatArray[i, j] = (float)multidimArray.GetValue(i, j);
            }
        }

        return floatArray;
    }
}