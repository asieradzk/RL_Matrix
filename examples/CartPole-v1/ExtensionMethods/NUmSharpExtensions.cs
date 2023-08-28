using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public static class NumSharpExtensions
{
    public static float[] ToFloatArray(this NDArray npArray)
    {
        double[] doubleArray = npArray.ToArray<double>();
        return Array.ConvertAll(doubleArray, item => (float)item);
    }
    public static float[,] ToFloat2DArray(this NDArray ndArray)
    {
        var multidimArray = ndArray.ToMuliDimArray<float>();
        var floatArray = new float[multidimArray.GetLength(0), multidimArray.GetLength(1)];

        for (int i = 0; i < multidimArray.GetLength(0); i++)
        {
            for (int j = 0; j < multidimArray.GetLength(1); j++)
            {
                floatArray[i, j] = (float)multidimArray.GetValue(i, j);
            }
        }

        return floatArray;
    }
}