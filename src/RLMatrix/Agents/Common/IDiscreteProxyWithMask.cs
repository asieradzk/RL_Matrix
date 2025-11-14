using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace RLMatrix.Agents.Common
{
    // Optional extension for proxies that support action masks
    public interface IDiscreteProxyWithMask<T>
    {
#if NET8_0_OR_GREATER
        ValueTask<Dictionary<Guid, int[]>> SelectActionsBatchWithMaskAsync(List<(Guid environmentId, T state, int[][] actionMasks)> stateInfos, bool isTraining);
#else
        Task<Dictionary<Guid, int[]>> SelectActionsBatchWithMaskAsync(List<(Guid environmentId, T state, int[][] actionMasks)> stateInfos, bool isTraining);
#endif
    }
}

