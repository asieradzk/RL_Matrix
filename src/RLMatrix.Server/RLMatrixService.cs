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

    public class RLMatrixService : IRLMatrixService
    {
        private IRLMatrixProxyAdapter? _proxyAdapter = null;
        private readonly string _savePath;
        private readonly object _lock = new object();
        private readonly Queue<Func<ValueTask>> _requestQueue = new Queue<Func<ValueTask>>();
        private bool _isOptimizing = false;

        public RLMatrixService(string savePath)
        {
            _savePath = savePath;
        }

        public void Initialize(OneOf<DQNAgentOptions, PPOAgentOptions> opts, int[] discreteActionSizes, (float min, float max)[] continuousActionBounds, OneOf<int, (int, int)> stateSizes)
        {
            if (_proxyAdapter != null)
            {
                throw new InvalidOperationException("Service already initialized");
            }

            opts.Switch(
                dqnOpts => throw new NotSupportedException("DQN is not supported for continuous actions"),
                ppoOpts =>
                {
                    stateSizes.Switch(
                        stateSize =>
                        {
                            if (continuousActionBounds.Length == 0)
                            {
                                _proxyAdapter = new DiscreteProxyAdapterImpl<float[]>(
                                    new LocalDiscretePPOAgent<float[]>(ppoOpts, discreteActionSizes, stateSize),
                                    _savePath);
                            }
                            else
                            {
                                _proxyAdapter = new ContinuousProxyAdapterImpl<float[]>(
                                    new LocalContinuousPPOAgent<float[]>(ppoOpts, discreteActionSizes, stateSize, continuousActionBounds),
                                    _savePath);
                            }
                        },
                        sizes =>
                        {
                            if (continuousActionBounds.Length == 0)
                            {
                                _proxyAdapter = new DiscreteProxyAdapterImpl<float[,]>(
                                    new LocalDiscretePPOAgent<float[,]>(ppoOpts, discreteActionSizes, sizes),
                                    _savePath);
                            }
                            else
                            {
                                _proxyAdapter = new ContinuousProxyAdapterImpl<float[,]>(
                                    new LocalContinuousPPOAgent<float[,]>(ppoOpts, discreteActionSizes, sizes, continuousActionBounds),
                                    _savePath);
                            }
                        }
                    );
                }
            );
        }

        public async ValueTask<ActionResponseDTO> SelectActionsBatchAsync(List<StateInfoDTO> stateInfos, bool isTraining)
        {
            ActionResponseDTO result = null;
            await EnqueueRequest(async () =>
            {
                var _ = _proxyAdapter ?? throw new InvalidOperationException("Service not initialized");
                result = await _proxyAdapter.SelectActionsBatchAsync(stateInfos, isTraining);
            });
            return result;
        }

        public async ValueTask ResetStates(List<(Guid environmentId, bool dones)> environmentIds)
        {
            await EnqueueRequest(async () =>
            {
                var _ = _proxyAdapter ?? throw new InvalidOperationException("Service not initialized");
                await _proxyAdapter.ResetStates(environmentIds);
            });
        }

        public async ValueTask UploadTransitionsAsync(List<TransitionPortableDTO> transitions)
        {
            await EnqueueRequest(async () =>
            {
                var _ = _proxyAdapter ?? throw new InvalidOperationException("Service not initialized");
                await _proxyAdapter.UploadTransitionsAsync(transitions);
            });
        }

        public async ValueTask OptimizeModelAsync()
        {
            lock (_lock)
            {
                _isOptimizing = true;
            }

            var _ = _proxyAdapter ?? throw new InvalidOperationException("Service not initialized");
            await _proxyAdapter.OptimizeModelAsync();

            lock (_lock)
            {
                _isOptimizing = false;
            }

            await ProcessQueuedRequests();
        }

        public ValueTask SaveAsync()
        {
            var _ = _proxyAdapter ?? throw new InvalidOperationException("Service not initialized");
            return _proxyAdapter.SaveAsync();
        }

        public ValueTask LoadAsync()
        {
            var _ = _proxyAdapter ?? throw new InvalidOperationException("Service not initialized");
            return _proxyAdapter.LoadAsync();
        }

        private async ValueTask EnqueueRequest(Func<ValueTask> request)
        {
            lock (_lock)
            {
                if (_isOptimizing)
                {
                    _requestQueue.Enqueue(request);
                    return;
                }
            }

            await request();
        }

        private async ValueTask ProcessQueuedRequests()
        {
            while (true)
            {
                Func<ValueTask> request;
                lock (_lock)
                {
                    if (_requestQueue.Count == 0)
                        break;

                    request = _requestQueue.Dequeue();
                }

                await request();
            }
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