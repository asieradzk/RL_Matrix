using RLMatrix.Common.Remote;
using RLMatrix.Server;
using Microsoft.AspNetCore.SignalR;
using OneOf;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

public class RLMatrixHub : Hub
{
    private readonly IRLMatrixService _service;

    public RLMatrixHub(IRLMatrixService service)
    {
        _service = service;
    }

    public async Task Initialize(AgentOptionsDTO opts, int[] discreteActionSizes, (float min, float max)[] continuousActionBounds, StateSizesDTO stateSizes)
    {
        Console.WriteLine($"Initializing... request from {Context.ConnectionId}");
        try
        {
            _service.Initialize(opts.ToAgentOptions(), discreteActionSizes, continuousActionBounds, stateSizes.ToOneOf());
            Console.WriteLine("Initialized");
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to initialize the service.");
            Console.WriteLine(ex);
        }
    }

    public async Task<ActionResponseDTO> SelectActions(List<StateInfoDTO> stateInfosDTOs, bool isTraining)
    {
        try
        {
            return await _service.SelectActionsBatchAsync(stateInfosDTOs, isTraining);
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
            await _service.ResetStates(environmentIds);
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
            await _service.UploadTransitionsAsync(transitions);
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
            await _service.OptimizeModelAsync();
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
            await _service.SaveAsync();
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
            await _service.LoadAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine("Failed to load.");
            Console.WriteLine(ex);
            throw;
        }
    }
}