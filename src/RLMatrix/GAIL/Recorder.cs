using RLMatrix.Common;

namespace RLMatrix;

// TODO: not implemented - still gonna touch it up :)
public class Recorder<TState>
    where TState : notnull
{
    private readonly IMemory<TState> _memory;
    private MemoryTransition<TState>? _previousTransition;

    public Recorder()
    {
        throw new NotImplementedException();
        //_memory = new TransitionReplayMemory<T>(10000, 512);
    }

    public async ValueTask AddStepAsync(TState observation, int[] discreteActions, float[] continuousActions, float reward)
    {
        var newTransition = new MemoryTransition<TState>(Utilities<TState>.DeepCopy(observation), RLActions.Continuous(discreteActions, continuousActions), reward, default, null, null);
           
        if(_previousTransition != null)
        {
            var transition = new MemoryTransition<TState>(
                _previousTransition.State, _previousTransition.Actions, _previousTransition.Reward, newTransition.State, _previousTransition, null);
            
            await _memory.PushAsync(transition);
        }
        
        _previousTransition = newTransition;
    }

    public async Task EndEpisodeAsync()
    {
        if(_previousTransition != null)
        {
            await _memory.PushAsync(_previousTransition);
        }
        
        _previousTransition = null;
    }

    public ValueTask SaveAsync(string path)
    {
        throw new NotImplementedException();
        //  myMemory.Save(pathToFile);
    }

    public ValueTask LoadAsync(string path)
    {
        throw new NotImplementedException();
        //  myMemory.Load(pathToFile);
    }

}