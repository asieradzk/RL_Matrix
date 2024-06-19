using RLMatrix.Common.Remote;
using RLMatrix.Server;
using Microsoft.AspNetCore.SignalR;
public class RLMatrixHub : Hub
{
    private readonly IDiscreteRLMatrixService _proxyAdapter;

    public RLMatrixHub(IDiscreteRLMatrixService proxyAdapter)
    {
        _proxyAdapter = proxyAdapter;
    }

    public async Task Initialize(AgentOptionsDTO opts, int[] actionSizes, StateSizesDTO stateSizes)
    {
        Console.WriteLine($"Initializing... request from {Context.ConnectionId}");
        try
        {
            _proxyAdapter.Initialize(opts.ToAgentOptions(), actionSizes, stateSizes.ToOneOf());
            Console.WriteLine("Initialized");
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to initialize the service.");
            Console.WriteLine(ex);
        }
    }

    public async Task<Dictionary<Guid, int[]>> SelectActions(List<StateInfoDTO> stateInfosDTOs)
    {
        try
        {
            return await _proxyAdapter.SelectActionsBatchAsync(stateInfosDTOs);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to select actions.");
            Console.WriteLine(ex);
            throw;
        }
    }

    public async Task ResetStates(List<(Guid environmentId, bool dones)> environmentIds)
    {
        try
        {
            await _proxyAdapter.ResetStates(environmentIds);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to reset states.");
            Console.WriteLine(ex);
            throw;
        }
    }

    public async Task UploadTransitions(List<TransitionPortableDTO> transitions)
    {
        try
        {
            await _proxyAdapter.UploadTransitionsAsync(transitions);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to upload transitions.");
            Console.WriteLine(ex);
            throw;
        }
    }

    public async Task OptimizeModel()
    {
        try
        {
            await _proxyAdapter.OptimizeModelAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to optimize model.");
            Console.WriteLine(ex);
            throw;
        }
    }

    public async Task Save()
    {
        try
        {
            await _proxyAdapter.SaveAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to save.");
            Console.WriteLine(ex);
            throw;
        }
    }

    public async Task Load()
    {
        try
        {
            await _proxyAdapter.LoadAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to load.");
            Console.WriteLine(ex);
            throw;
        }
    }
}
