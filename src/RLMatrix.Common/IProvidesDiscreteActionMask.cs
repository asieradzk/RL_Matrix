using System;

namespace RLMatrix.Common
{
    // Optional interface implemented by generated discrete environments that provide action masks
    public interface IProvidesDiscreteActionMask
    {
        // Returns per-head masks; for H discrete heads returns H arrays of length actionSize[head]
        int[][] GetDiscreteActionMasks();
    }
}

