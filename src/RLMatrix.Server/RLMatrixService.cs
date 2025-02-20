using RLMatrix.Common;
using RLMatrix.Remote;

namespace RLMatrix.Server;

public interface IRLMatrixProxyAdapter : ISavable
{
    ValueTask<ActionResponseDTO> SelectBatchActionsAsync(List<StateInfoDTO> stateDTOs, bool isTraining);
    ValueTask ResetStatesAsync(List<(Guid EnvironmentId, bool IsDone)> environmentIds);
    ValueTask UploadTransitionsAsync(List<TransitionPortableDTO> transitions);
    ValueTask OptimizeModelAsync();
}

public interface IRLMatrixService : IRLMatrixProxyAdapter
{
    Task InitializeAsync(IAgentOptions options, int[] discreteActionDimensions, ContinuousActionDimensions[] continuousActionDimensions, StateDimensions stateDimensions);
    string SavePath { get; }
}

public class RLMatrixService(string savePath) : IRLMatrixService, IDisposable
{
    private IRLMatrixProxyAdapter? _proxyAdapter;
    private readonly SemaphoreSlim _semaphore = new(1, 1);
    private readonly CancellationTokenSource _optimizationCts = new();
    //private Task? _continuousOptimizationTask; TODO : unused
    private bool _isDqnAgent;

    public string SavePath { get; } = savePath;

    public async Task InitializeAsync(IAgentOptions options, int[] discreteActionDimensions, ContinuousActionDimensions[] continuousActionDimensions, StateDimensions stateDimensions)
    {
        await _semaphore.WaitAsync();
        
        try
        {
            if (_proxyAdapter != null)
            {
                throw new InvalidOperationException("Service already initialized.");
            }

            if (options is DQNAgentOptions)
                _isDqnAgent = true;

            _proxyAdapter = CreateProxyAdapter(options, discreteActionDimensions, continuousActionDimensions, stateDimensions);
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public async ValueTask<ActionResponseDTO> SelectBatchActionsAsync(List<StateInfoDTO> stateDTOs, bool isTraining)
    {
        await _semaphore.WaitAsync();

        ThrowIfNotInitialized();
        
        try
        {
            return await _proxyAdapter!.SelectBatchActionsAsync(stateDTOs, isTraining);
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public async ValueTask ResetStatesAsync(List<(Guid EnvironmentId, bool IsDone)> environmentIds)
    {
        await _semaphore.WaitAsync();
        
        ThrowIfNotInitialized();
        
        try
        {
            await _proxyAdapter!.ResetStatesAsync(environmentIds);
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public async ValueTask UploadTransitionsAsync(List<TransitionPortableDTO> transitions)
    {
        await _semaphore.WaitAsync();
        
        ThrowIfNotInitialized();
        
        try
        {
            await _proxyAdapter!.UploadTransitionsAsync(transitions);
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public async ValueTask OptimizeModelAsync()
    {
        if (_isDqnAgent)
            return;
        
        await _semaphore.WaitAsync();
            
        ThrowIfNotInitialized();
            
        try
        {
            await _proxyAdapter!.OptimizeModelAsync();
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public async ValueTask SaveAsync()
    {
        await _semaphore.WaitAsync();
        
        ThrowIfNotInitialized();
        
        try
        {
            await _proxyAdapter!.SaveAsync(SavePath);
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public async ValueTask LoadAsync()
    {
        await _semaphore.WaitAsync();
        
        ThrowIfNotInitialized();
        
        try
        {
            await _proxyAdapter!.LoadAsync(SavePath);
        }
        finally
        {
            _semaphore.Release();
        }
    }

    /* TODO: unused?
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
    */

    public void Dispose()
    {
        _optimizationCts.Cancel();
        //_continuousOptimizationTask?.Wait(); TODO: unused
        _optimizationCts.Dispose();
        _semaphore.Dispose();
    }

    private IRLMatrixProxyAdapter CreateProxyAdapter(IAgentOptions opts, int[] discreteActionDimensions, ContinuousActionDimensions[]? continuousActionDimensions, StateDimensions stateDimensions)
    {
        return opts switch
        {
            DQNAgentOptions dqn => stateDimensions.Dimensions.Length switch
            {
                1 => new DiscreteProxyAdapterImpl<float[]>(new LocalDiscreteQAgent<float[]>(dqn, discreteActionDimensions, stateDimensions),
                    SavePath),
                2 => new DiscreteProxyAdapterImpl<float[,]>(new LocalDiscreteQAgent<float[,]>(dqn, discreteActionDimensions, stateDimensions),
                    SavePath),
                _ => throw new ArgumentOutOfRangeException(nameof(stateDimensions.Dimensions))
            },
            PPOAgentOptions ppo when continuousActionDimensions is not null => stateDimensions.Dimensions.Length switch
            {
                1 => new ContinuousProxyAdapterImpl<float[]>(new LocalContinuousPPOAgent<float[]>(ppo, discreteActionDimensions, continuousActionDimensions, stateDimensions), SavePath),
                2 => new ContinuousProxyAdapterImpl<float[,]>(new LocalContinuousPPOAgent<float[,]>(ppo, discreteActionDimensions, continuousActionDimensions, stateDimensions), SavePath),
                _ => throw new ArgumentOutOfRangeException(nameof(stateDimensions.Dimensions))
            },
            PPOAgentOptions ppo => stateDimensions.Dimensions.Length switch
            {
                1 => new DiscreteProxyAdapterImpl<float[]>(new LocalDiscretePPOAgent<float[]>(ppo, discreteActionDimensions, stateDimensions), SavePath),
                2 => new DiscreteProxyAdapterImpl<float[,]>(new LocalDiscretePPOAgent<float[,]>(ppo, discreteActionDimensions, stateDimensions), SavePath),
                _ => throw new ArgumentOutOfRangeException(nameof(stateDimensions.Dimensions))
            },
            _ => throw new ArgumentOutOfRangeException(nameof(opts))
        };
    }

    private void ThrowIfNotInitialized()
    {
        if (_proxyAdapter is null)
            throw new InvalidOperationException("Service not initialized.");
    }
    
    ValueTask ISavable.SaveAsync(string path) => SaveAsync();
    ValueTask ISavable.LoadAsync(string path) => LoadAsync();
}

internal class DiscreteProxyAdapterImpl<TState>(IDiscreteProxy<TState> proxy, string savePath) : IRLMatrixProxyAdapter
    where TState : notnull
{
    public async ValueTask<ActionResponseDTO> SelectBatchActionsAsync(List<StateInfoDTO> stateDTOs, bool isTraining)
    {
        var stateInfos = stateDTOs.FromDTOList<TState>();
        var actions = await proxy.SelectBatchActionsAsync(stateInfos, isTraining);
        var actionDTOs = actions.ToDictionary(kvp => kvp.Key, kvp => new ActionDTO(kvp.Value.DiscreteActions));
        return new ActionResponseDTO(actionDTOs);
    }

    public ValueTask ResetStatesAsync(List<(Guid EnvironmentId, bool IsDone)> environmentIds)
    {
        return proxy.ResetStatesAsync(environmentIds);
    }

    public ValueTask UploadTransitionsAsync(List<TransitionPortableDTO> transitions)
    {
        var typedTransitions = transitions.FromDTOList<TState>();
        return proxy.UploadTransitionsAsync(typedTransitions);
    }

    public ValueTask OptimizeModelAsync()
    {
        return proxy.OptimizeModelAsync();
    }

    public ValueTask SaveAsync()
    {
        return proxy.SaveAsync(savePath);
    }

    public ValueTask LoadAsync()
    {
        return proxy.LoadAsync(savePath);
    }
    
    ValueTask ISavable.SaveAsync(string path) => SaveAsync();
    ValueTask ISavable.LoadAsync(string path) => LoadAsync();
}

internal class ContinuousProxyAdapterImpl<TState>(IContinuousProxy<TState> proxy, string savePath) : IRLMatrixProxyAdapter
    where TState : notnull
{
    public async ValueTask<ActionResponseDTO> SelectBatchActionsAsync(List<StateInfoDTO> stateDTOs, bool isTraining)
    {
        var stateInfos = stateDTOs.FromDTOList<TState>();
        var actions = await proxy.SelectBatchActionsAsync(stateInfos, isTraining);
        var actionDTOs = actions.ToDictionary(kvp => kvp.Key, kvp => new ActionDTO(kvp.Value.DiscreteActions, kvp.Value.ContinuousActions));
        return new ActionResponseDTO(actionDTOs);
    }

    public ValueTask ResetStatesAsync(List<(Guid EnvironmentId, bool IsDone)> environmentIds)
    {
        return proxy.ResetStatesAsync(environmentIds);
    }

    public ValueTask UploadTransitionsAsync(List<TransitionPortableDTO> transitions)
    {
        var typedTransitions = transitions.FromDTOList<TState>();
        return proxy.UploadTransitionsAsync(typedTransitions);
    }

    public ValueTask OptimizeModelAsync()
    {
        return proxy.OptimizeModelAsync();
    }

    public ValueTask SaveAsync()
    {
        return proxy.SaveAsync(savePath);
    }

    public ValueTask LoadAsync()
    {
        return proxy.LoadAsync(savePath);
    }

    ValueTask ISavable.SaveAsync(string path) => SaveAsync();
    ValueTask ISavable.LoadAsync(string path) => LoadAsync();
}