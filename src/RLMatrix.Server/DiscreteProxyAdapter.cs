using RLMatrix.Agents.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLMatrix.Server
{
    public interface IDiscreteProxyAdapter
    {
        ValueTask<Dictionary<Guid, int[]>> SelectActionsBatchAsync(List<(Guid environmentId, object state)> stateInfos);
        ValueTask ResetStates(List<(Guid environmentId, bool dones)> environmentIds);
        ValueTask UploadTransitionsAsync(IEnumerable<object> transitions);
        ValueTask OptimizeModelAsync();
        ValueTask SaveAsync(string path);
        ValueTask LoadAsync(string path);
    }


    public static class DiscreteProxyAdapter
    {
        public static IDiscreteProxyAdapter CreateAdapterFromTState<T>()
        {
            return null;
        }

        private class DiscreteProxyAdapterImpl<T> : IDiscreteProxyAdapter
        {
            private readonly IDiscreteProxy<T> _proxy;

            public DiscreteProxyAdapterImpl(IDiscreteProxy<T> proxy)
            {
                _proxy = proxy;
            }

            public async ValueTask<Dictionary<Guid, int[]>> SelectActionsBatchAsync(List<(Guid environmentId, object state)> stateInfos)
            {
                var typedStateInfos = stateInfos.Select(si => (si.environmentId, (T)si.state)).ToList();
                return await _proxy.SelectActionsBatchAsync(typedStateInfos);
            }

            public ValueTask ResetStates(List<(Guid environmentId, bool dones)> environmentIds)
            {
                return _proxy.ResetStates(environmentIds);
            }

            public ValueTask UploadTransitionsAsync(IEnumerable<object> transitions)
            {
                var typedTransitions = transitions.Cast<TransitionPortable<T>>();
                return _proxy.UploadTransitionsAsync(typedTransitions);
            }

            public ValueTask OptimizeModelAsync()
            {
                return _proxy.OptimizeModelAsync();
            }

            public ValueTask SaveAsync(string path)
            {
                return _proxy.SaveAsync(path);
            }

            public ValueTask LoadAsync(string path)
            {
                return _proxy.LoadAsync(path);
            }
        }
    }
}
