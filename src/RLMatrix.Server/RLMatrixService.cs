using OneOf;
using RLMatrix.Agents.Common;
using RLMatrix.Agents.PPO.Implementations;
using RLMatrix.Common.Remote;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace RLMatrix.Server
{
    public interface IRLMatrixProxyAdapter
    {
        ValueTask<ActionResponseDTO> SelectActionsBatchAsync(List<StateInfoDTO> stateInfos, bool isTraining);
        ValueTask ResetStates(List<(Guid environmentId, bool dones)> environmentIds);
        ValueTask UploadTransitionsAsync(List<TransitionPortableDTO> transitions);
        ValueTask OptimizeModelAsync();
        ValueTask SaveAsync();
        ValueTask LoadAsync();
    }

    public interface IRLMatrixService : IRLMatrixProxyAdapter
    {
        void Initialize(OneOf<DQNAgentOptions, PPOAgentOptions> opts, int[] discreteActionSizes, (float min, float max)[] continuousActionBounds, OneOf<int, (int, int)> stateSizes);
    }

    public class RLMatrixService : IRLMatrixService, IDisposable
    {
        private IRLMatrixProxyAdapter? _proxyAdapter = null;
        private readonly string _savePath;
        private readonly SemaphoreSlim _semaphore = new SemaphoreSlim(1, 1);
        private readonly CancellationTokenSource _optimizationCts = new CancellationTokenSource();
        private Task? _continuousOptimizationTask;
        private bool _isDqnAgent = false;

        public RLMatrixService(string savePath)
        {
            _savePath = savePath;
        }

        public void Initialize(OneOf<DQNAgentOptions, PPOAgentOptions> opts, int[] discreteActionSizes, (float min, float max)[] continuousActionBounds, OneOf<int, (int, int)> stateSizes)
        {
            _semaphore.Wait();
            try
            {
                if (_proxyAdapter != null)
                {
                    throw new InvalidOperationException("Service already initialized");
                }

                opts.Switch(
                    dqnOpts =>
                    {
                        _isDqnAgent = true;
                        _proxyAdapter = CreateDQNAdapter(dqnOpts, discreteActionSizes, stateSizes);
                        StartContinuousOptimization();
                    },
                    ppoOpts =>
                    {
                        _proxyAdapter = CreatePPOAdapter(ppoOpts, discreteActionSizes, continuousActionBounds, stateSizes);
                    }
                );
            }
            finally
            {
                _semaphore.Release();
            }
        }

        public async ValueTask<ActionResponseDTO> SelectActionsBatchAsync(List<StateInfoDTO> stateInfos, bool isTraining)
        {
            await _semaphore.WaitAsync();
            try
            {
                var _ = _proxyAdapter ?? throw new InvalidOperationException("Service not initialized");
                return await _proxyAdapter.SelectActionsBatchAsync(stateInfos, isTraining);
            }
            finally
            {
                _semaphore.Release();
            }
        }

        public async ValueTask ResetStates(List<(Guid environmentId, bool dones)> environmentIds)
        {
            await _semaphore.WaitAsync();
            try
            {
                var _ = _proxyAdapter ?? throw new InvalidOperationException("Service not initialized");
                await _proxyAdapter.ResetStates(environmentIds);
            }
            finally
            {
                _semaphore.Release();
            }
        }

        public async ValueTask UploadTransitionsAsync(List<TransitionPortableDTO> transitions)
        {
            await _semaphore.WaitAsync();
            try
            {
                var _ = _proxyAdapter ?? throw new InvalidOperationException("Service not initialized");
                await _proxyAdapter.UploadTransitionsAsync(transitions);
            }
            finally
            {
                _semaphore.Release();
            }
        }

        public async ValueTask OptimizeModelAsync()
        {
            if (!_isDqnAgent)
            {
                await _semaphore.WaitAsync();
                try
                {
                    var _ = _proxyAdapter ?? throw new InvalidOperationException("Service not initialized");
                    await _proxyAdapter.OptimizeModelAsync();
                }
                finally
                {
                    _semaphore.Release();
                }
            }
        }

        public async ValueTask SaveAsync()
        {
            await _semaphore.WaitAsync();
            try
            {
                var _ = _proxyAdapter ?? throw new InvalidOperationException("Service not initialized");
                await _proxyAdapter.SaveAsync();
            }
            finally
            {
                _semaphore.Release();
            }
        }

        public async ValueTask LoadAsync()
        {
            await _semaphore.WaitAsync();
            try
            {
                var _ = _proxyAdapter ?? throw new InvalidOperationException("Service not initialized");
                await _proxyAdapter.LoadAsync();
            }
            finally
            {
                _semaphore.Release();
            }
        }

        private void StartContinuousOptimization()
        {
            _continuousOptimizationTask = Task.Run(async () =>
            {
                while (!_optimizationCts.IsCancellationRequested)
                {
                    await _semaphore.WaitAsync();
                    try
                    {
                        if (_proxyAdapter != null)
                        {
                            await _proxyAdapter.OptimizeModelAsync();
                        }
                    }
                    finally
                    {
                        _semaphore.Release();
                    }
                    await Task.Delay(100);
                }
            });
        }

        public void Dispose()
        {
            _optimizationCts.Cancel();
            _continuousOptimizationTask?.Wait();
            _optimizationCts.Dispose();
            _semaphore.Dispose();
        }

        private IRLMatrixProxyAdapter CreateDQNAdapter(DQNAgentOptions opts, int[] discreteActionSizes, OneOf<int, (int, int)> stateSizes)
        {
            return stateSizes.Match<IRLMatrixProxyAdapter>(
                stateSize => new DiscreteProxyAdapterImpl<float[]>(
                    new LocalDiscreteQAgent<float[]>(opts, discreteActionSizes, stateSize),
                    _savePath),
                sizes => new DiscreteProxyAdapterImpl<float[,]>(
                    new LocalDiscreteQAgent<float[,]>(opts, discreteActionSizes, sizes),
                    _savePath)
            );
        }

        private IRLMatrixProxyAdapter CreatePPOAdapter(PPOAgentOptions opts, int[] discreteActionSizes, (float min, float max)[] continuousActionBounds, OneOf<int, (int, int)> stateSizes)
        {
            return stateSizes.Match<IRLMatrixProxyAdapter>(
                stateSize => continuousActionBounds.Length == 0
                    ? new DiscreteProxyAdapterImpl<float[]>(
                        new LocalDiscretePPOAgent<float[]>(opts, discreteActionSizes, stateSize),
                        _savePath)
                    : new ContinuousProxyAdapterImpl<float[]>(
                        new LocalContinuousPPOAgent<float[]>(opts, discreteActionSizes, stateSize, continuousActionBounds),
                        _savePath),
                sizes => continuousActionBounds.Length == 0
                    ? new DiscreteProxyAdapterImpl<float[,]>(
                        new LocalDiscretePPOAgent<float[,]>(opts, discreteActionSizes, sizes),
                        _savePath)
                    : new ContinuousProxyAdapterImpl<float[,]>(
                        new LocalContinuousPPOAgent<float[,]>(opts, discreteActionSizes, sizes, continuousActionBounds),
                        _savePath)
            );
        }
    }

    class DiscreteProxyAdapterImpl<T> : IRLMatrixProxyAdapter
    {
        private readonly IDiscreteProxy<T> _proxy;
        private readonly string _savePath;

        public DiscreteProxyAdapterImpl(IDiscreteProxy<T> proxy, string savePath)
        {
            _proxy = proxy;
            _savePath = savePath;
        }

        public async ValueTask<ActionResponseDTO> SelectActionsBatchAsync(List<StateInfoDTO> stateInfosDTOs, bool isTraining)
        {
            var stateInfos = stateInfosDTOs.UnpackList<T>();
            var actions = await _proxy.SelectActionsBatchAsync(stateInfos, isTraining);
            var actionDTOs = actions.ToDictionary(kvp => kvp.Key, kvp => new ActionDTO(kvp.Value));
            var resp = new ActionResponseDTO(actionDTOs);
            return resp;
        }

        public ValueTask ResetStates(List<(Guid environmentId, bool dones)> environmentIds)
        {
            return _proxy.ResetStates(environmentIds);
        }

        public ValueTask UploadTransitionsAsync(List<TransitionPortableDTO> transitions)
        {
            var typedTransitions = transitions.FromDTOList<T>();
            return _proxy.UploadTransitionsAsync(typedTransitions);
        }

        public ValueTask OptimizeModelAsync()
        {
            return _proxy.OptimizeModelAsync();
        }

        public ValueTask SaveAsync()
        {
            return _proxy.SaveAsync(_savePath);
        }

        public ValueTask LoadAsync()
        {
            return _proxy.LoadAsync(_savePath);
        }
    }

    class ContinuousProxyAdapterImpl<T> : IRLMatrixProxyAdapter
    {
        private readonly IContinuousProxy<T> _proxy;
        private readonly string _savePath;

        public ContinuousProxyAdapterImpl(IContinuousProxy<T> proxy, string savePath)
        {
            _proxy = proxy;
            _savePath = savePath;
        }

        public async ValueTask<ActionResponseDTO> SelectActionsBatchAsync(List<StateInfoDTO> stateInfosDTOs, bool isTraining)
        {
            var stateInfos = stateInfosDTOs.UnpackList<T>();
            var actions = await _proxy.SelectActionsBatchAsync(stateInfos, isTraining);
            var actionDTOs = actions.ToDictionary(kvp => kvp.Key, kvp => new ActionDTO(kvp.Value.discreteActions, kvp.Value.continuousActions));
            return new ActionResponseDTO(actionDTOs);
        }

        public ValueTask ResetStates(List<(Guid environmentId, bool dones)> environmentIds)
        {
            return _proxy.ResetStates(environmentIds);
        }

        public ValueTask UploadTransitionsAsync(List<TransitionPortableDTO> transitions)
        {
            var typedTransitions = transitions.FromDTOList<T>();
            return _proxy.UploadTransitionsAsync(typedTransitions);
        }

        public ValueTask OptimizeModelAsync()
        {
            return _proxy.OptimizeModelAsync();
        }

        public ValueTask SaveAsync()
        {
            return _proxy.SaveAsync(_savePath);
        }

        public ValueTask LoadAsync()
        {
            return _proxy.LoadAsync(_savePath);
        }
    }
}