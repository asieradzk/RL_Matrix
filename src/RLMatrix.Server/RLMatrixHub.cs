using Microsoft.AspNetCore.SignalR;
using RLMatrix.Common;
using RLMatrix.Remote;

namespace RLMatrix.Server;

public class RLMatrixHub(IRLMatrixService service) : Hub
{
    public Task InitializeAsync(AgentOptionsDTO opts, int[] discreteActionDimensions, ContinuousActionDimensions[] continuousActionDimensions, StateSizesDTO stateSizes)
    {
        Console.WriteLine($"Initializing... request from {Context.ConnectionId}");
        
        try
        {
            Console.WriteLine("Initialized");
            return service.InitializeAsync(opts.ToAgentOptions(), discreteActionDimensions, continuousActionDimensions, stateSizes.ToStateDimensions());
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to initialize the service.");
            Console.WriteLine(ex);
            throw;
        }
    }

    public ValueTask<ActionResponseDTO> SelectBatchActionsAsync(List<StateInfoDTO> stateInfosDTOs, bool isTraining)
    {
        try
        {
            return service.SelectBatchActionsAsync(stateInfosDTOs, isTraining);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to select actions.");
            Console.WriteLine(ex);
            throw;
        }
    }

    public ValueTask ResetStatesAsync(List<(Guid EnvironmentId, bool IsDone)> environments)
    {
        try
        {
            return service.ResetStatesAsync(environments);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to reset states.");
            Console.WriteLine(ex);
            throw;
        }
    }

    public ValueTask UploadTransitionsAsync(List<TransitionPortableDTO> transitions)
    {
        try
        {
            return service.UploadTransitionsAsync(transitions);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to upload transitions.");
            Console.WriteLine(ex);
            throw;
        }
    }

    public ValueTask OptimizeModelAsync()
    {
        try
        {
            return service.OptimizeModelAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to optimize model.");
            Console.WriteLine(ex);
            throw;
        }
    }

    public ValueTask SaveAsync()
    {
        try
        {
            return service.SaveAsync(service.SavePath);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to save.");
            Console.WriteLine(ex);
            throw;
        }
    }

    public ValueTask LoadAsync()
    {
        try
        {
            return service.LoadAsync(service.SavePath);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to load.");
            Console.WriteLine(ex);
            throw;
        }
    }
}