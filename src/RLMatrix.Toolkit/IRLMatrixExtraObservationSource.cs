﻿namespace RLMatrix.Toolkit;

public interface IRLMatrixExtraObservationSource
{
    float[] GetObservations();
    int GetObservationSize();
}